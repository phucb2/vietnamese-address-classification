import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from address_classification import (
    ACAutomaton,
    Province,
    Street,
    TrigramIndex,
    Ward,
    basic_cleanup,
    levenshtein_cutoff,
    load_provinces,
    load_streets,
    load_wards,
    make_aliases,
    normalize_text,
    strip_diacritics_keep_d,
)

_SPACES_RE = re.compile(r"\s+")
_BEAM_WIDTH = 5
_TOP_K_FUZZY = 150
_N_PROVINCE = 4
_N_WARD = 5
_N_STREET = 3


def _nodiac_collapsed(s: str) -> str:
    """Strip diacritics + collapse spaces — no abbreviation expansion."""
    return basic_cleanup(strip_diacritics_keep_d(s)).replace(" ", "")


def _nodiac_spaced(s: str) -> str:
    """Strip diacritics only — no abbreviation expansion."""
    return basic_cleanup(strip_diacritics_keep_d(s))


@dataclass
class Candidate:
    level: str
    entity_id: Optional[str]
    alias: str
    source: str
    score: float
    dist: int = 0
    overlap: int = 0


@dataclass
class BeamState:
    province_id: Optional[str]
    ward_id: Optional[str]
    street_id: Optional[str]
    score: float
    trace: List[str] = field(default_factory=list)


def _split_raw_segments(raw_text: str) -> List[str]:
    return [p.strip() for p in raw_text.split(",") if p.strip()]


class BeamSearchAddressCorrector:
    def __init__(
        self,
        provinces: Dict[str, Province],
        wards: Dict[str, Ward],
        streets: Optional[Dict[str, Street]] = None,
    ):
        self.provinces = provinces
        self.wards = wards
        self.streets = streets or {}
        self.beam_width = _BEAM_WIDTH

        self.province_to_wards: Dict[str, List[str]] = {}
        for pid in self.provinces:
            self.province_to_wards[pid] = []
        for wid, ward in self.wards.items():
            self.province_to_wards.setdefault(ward.province_id, []).append(wid)

        self._ward_admin_cache: Dict[str, Optional[str]] = {}

        self.ac = {
            "province": ACAutomaton(),
            "ward": ACAutomaton(),
            "street": ACAutomaton() if self.streets else None,
        }
        self.fuzzy_idx = {
            "province": TrigramIndex(),
            "ward": TrigramIndex(),
            "street": TrigramIndex() if self.streets else None,
        }
        self._build_indexes()

    def _build_indexes(self) -> None:
        alias_id = 1
        for pid, p in self.provinces.items():
            for a in make_aliases(p.name):
                a_norm = basic_cleanup(strip_diacritics_keep_d(a)).replace(" ", "")
                if len(a_norm) >= 3:
                    self.ac["province"].add(a_norm, ("province", pid, a_norm))
                a_f = basic_cleanup(strip_diacritics_keep_d(a))
                self.fuzzy_idx["province"].add(alias_id, a_f, meta=("province", pid, a_f))
                alias_id += 1

        for wid, w in self.wards.items():
            for a in make_aliases(w.name):
                a_norm = basic_cleanup(strip_diacritics_keep_d(a)).replace(" ", "")
                if len(a_norm) >= 3:
                    self.ac["ward"].add(a_norm, ("ward", wid, a_norm))
                a_f = basic_cleanup(strip_diacritics_keep_d(a))
                self.fuzzy_idx["ward"].add(alias_id, a_f, meta=("ward", wid, a_f))
                alias_id += 1

        if self.streets:
            for sid, s in self.streets.items():
                for a in make_aliases(s.name):
                    a_norm = basic_cleanup(strip_diacritics_keep_d(a)).replace(" ", "")
                    if len(a_norm) >= 3:
                        self.ac["street"].add(a_norm, ("street", sid, a_norm))
                    a_f = basic_cleanup(strip_diacritics_keep_d(a))
                    self.fuzzy_idx["street"].add(alias_id, a_f, meta=("street", sid, a_f))
                    alias_id += 1

        self.ac["province"].build()
        self.ac["ward"].build()
        if self.ac["street"]:
            self.ac["street"].build()
        self.fuzzy_idx["province"].build()
        self.fuzzy_idx["ward"].build()
        if self.fuzzy_idx["street"]:
            self.fuzzy_idx["street"].build()

    def _admin_type_of_ward(self, wid: str) -> Optional[str]:
        cached = self._ward_admin_cache.get(wid)
        if cached is not None:
            return cached
        w = self.wards.get(wid)
        if w is None:
            return None
        nm = basic_cleanup(strip_diacritics_keep_d(w.name))
        for prefix, atype in [
            ("dac khu ", "dac_khu"),
            ("thi tran ", "thi_tran"),
            ("thi xa ", "thi_xa"),
            ("phuong ", "phuong"),
            ("xa ", "xa"),
        ]:
            if nm.startswith(prefix):
                self._ward_admin_cache[wid] = atype
                return atype
        self._ward_admin_cache[wid] = None
        return None

    def _admin_type_from_text(self, text_nodiac: str) -> Optional[str]:
        t = f" {text_nodiac} "
        for token, atype in [
            ("dac khu", "dac_khu"),
            ("thi tran", "thi_tran"),
            ("thi xa", "thi_xa"),
            ("phuong", "phuong"),
            ("xa", "xa"),
        ]:
            if f" {token} " in t:
                return atype
        return None

    # ------------------------------------------------------------------
    # Candidate generators
    # ------------------------------------------------------------------

    def _exact_candidates_full(
        self,
        lvl: str,
        full_collapsed: str,
        top_n: int,
        restrict_ids: Optional[set] = None,
    ) -> List[Candidate]:
        ac = self.ac[lvl]
        if ac is None:
            return []
        found = ac.find(full_collapsed)
        best_by_id: Dict[str, Candidate] = {}
        text_len = max(1, len(full_collapsed))
        for end, length, _, payload in found:
            _, eid, alias = payload
            if restrict_ids is not None and eid not in restrict_ids:
                continue
            pos = end / text_len
            score = 4.0 + 0.15 * length
            if lvl in ("ward", "province"):
                score += 0.6 * pos
            else:
                score += 0.4 * (1 - pos)
            prev = best_by_id.get(eid)
            if prev is None or score > prev.score:
                best_by_id[eid] = Candidate(lvl, eid, alias, "exact", score=score)
        return sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)[:top_n]

    def _fuzzy_candidates(
        self,
        lvl: str,
        query_nodiac: str,
        top_n: int,
        max_dist: int = 3,
        restrict_ids: Optional[set] = None,
    ) -> List[Candidate]:
        idx = self.fuzzy_idx[lvl]
        if idx is None:
            return []
        raw = idx.candidates(query_nodiac, top_k=_TOP_K_FUZZY)
        best_by_id: Dict[str, Candidate] = {}
        for aid, overlap in raw:
            _, eid, alias = idx.alias_meta[aid]
            if restrict_ids is not None and eid not in restrict_ids:
                continue
            d = levenshtein_cutoff(query_nodiac, idx.alias_text[aid], max_dist)
            if d > max_dist:
                continue
            score = 2.5 + 0.35 * overlap - 1.1 * d + 0.08 * len(alias.replace(" ", ""))
            prev = best_by_id.get(eid)
            if prev is None or score > prev.score:
                best_by_id[eid] = Candidate(lvl, eid, alias, "fuzzy", score=score, dist=d, overlap=overlap)
        return sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)[:top_n]

    def _merge_candidates(self, *lists: List[Candidate], top_n: int) -> List[Candidate]:
        best: Dict[Optional[str], Candidate] = {}
        for lst in lists:
            for cand in lst:
                prev = best.get(cand.entity_id)
                if prev is None or cand.score > prev.score:
                    best[cand.entity_id] = cand
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_n]

    def _province_candidates(self, s_nodiac: str, s_collapsed: str) -> List[Candidate]:
        exact = self._exact_candidates_full("province", s_collapsed, top_n=_N_PROVINCE * 3)
        fuzzy = self._fuzzy_candidates("province", s_nodiac, top_n=_N_PROVINCE * 2, max_dist=2)
        merged = self._merge_candidates(exact, fuzzy, top_n=_N_PROVINCE)
        if not merged:
            merged = [Candidate("province", None, "", "none", score=0.0)]
        return merged

    def _ward_candidates_for_province(
        self,
        full_collapsed: str,
        ward_seg_nodiac: str,
        pid: Optional[str],
    ) -> List[Candidate]:
        restrict = set(self.province_to_wards.get(pid, [])) if pid is not None else None
        exact = self._exact_candidates_full("ward", full_collapsed, top_n=_N_WARD * 2, restrict_ids=restrict)
        if len(exact) >= 2 and exact[0].score >= 5.0:
            return exact[:_N_WARD]
        fuzzy = self._fuzzy_candidates(
            "ward",
            ward_seg_nodiac,
            top_n=_N_WARD * 2,
            max_dist=3 if pid is not None else 2,
            restrict_ids=restrict,
        )
        merged = self._merge_candidates(exact, fuzzy, top_n=_N_WARD)
        if not merged:
            merged = [Candidate("ward", None, "", "none", score=-0.2)]
        return merged

    def _street_candidates(self, full_collapsed: str, street_seg_nodiac: str) -> List[Candidate]:
        if not self.streets:
            return [Candidate("street", None, "", "none", score=0.0)]
        exact = self._exact_candidates_full("street", full_collapsed, top_n=_N_STREET * 2)
        if len(exact) >= 2 and exact[0].score >= 5.0:
            return exact[:_N_STREET]
        fuzzy = self._fuzzy_candidates("street", street_seg_nodiac, top_n=_N_STREET * 2, max_dist=3)
        merged = self._merge_candidates(exact, fuzzy, top_n=_N_STREET)
        if not merged:
            merged = [Candidate("street", None, "", "none", score=-0.3)]
        return merged

    # ------------------------------------------------------------------
    # Beam helpers
    # ------------------------------------------------------------------

    def _expand_beam(self, beam: List[BeamState], choices: List[Candidate], level: str) -> List[BeamState]:
        expanded: List[BeamState] = []
        for st in beam:
            for cand in choices:
                pid = st.province_id
                wid = st.ward_id
                sid = st.street_id
                if level == "province":
                    pid = cand.entity_id
                elif level == "ward":
                    wid = cand.entity_id
                elif level == "street":
                    sid = cand.entity_id
                score = st.score + cand.score
                trace = st.trace + [f"{level}:{cand.entity_id or 'none'}:{cand.source}:{cand.score:.2f}"]
                expanded.append(BeamState(pid, wid, sid, score, trace))
        expanded.sort(key=lambda x: x.score, reverse=True)
        return expanded[: self.beam_width]

    def _apply_consistency_rerank(self, beam: List[BeamState], q_type: Optional[str]) -> List[BeamState]:
        for st in beam:
            if st.province_id and st.ward_id:
                w = self.wards.get(st.ward_id)
                if w is not None and w.province_id == st.province_id:
                    st.score += 1.8
                elif w is not None:
                    st.score -= 2.5

                if q_type and w is not None:
                    w_type = self._admin_type_of_ward(st.ward_id)
                    if w_type == q_type:
                        st.score += 0.8
                    elif w_type is not None:
                        st.score -= 0.6
        beam.sort(key=lambda x: x.score, reverse=True)
        return beam[: self.beam_width]

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def correct(self, raw_text: str) -> dict:
        raw_segments = _split_raw_segments(raw_text)

        norm = normalize_text(raw_text)
        s_nodiac = norm["nodiac"]
        s_collapsed = norm["nodiac_collapsed"]

        if len(raw_segments) >= 3:
            ward_seg_nodiac = _nodiac_spaced(raw_segments[-2])
            street_seg_nodiac = _nodiac_spaced(raw_segments[0])
        elif len(raw_segments) == 2:
            ward_seg_nodiac = _nodiac_spaced(raw_segments[0])
            street_seg_nodiac = ""
        else:
            ward_seg_nodiac = s_nodiac
            street_seg_nodiac = s_nodiac

        q_type = self._admin_type_from_text(ward_seg_nodiac)

        # --- Province ---
        provinces = self._province_candidates(s_nodiac, s_collapsed)
        beam: List[BeamState] = [BeamState(None, None, None, 0.0, ["root"])]
        beam = self._expand_beam(beam, provinces, "province")

        # --- Ward (cached per province_id) ---
        ward_cache: Dict[Optional[str], List[Candidate]] = {}
        expanded_ward: List[BeamState] = []
        for st in beam:
            pid = st.province_id
            if pid not in ward_cache:
                ward_cache[pid] = self._ward_candidates_for_province(s_collapsed, ward_seg_nodiac, pid)
            expanded_ward.extend(self._expand_beam([st], ward_cache[pid], "ward"))
        expanded_ward.sort(key=lambda x: x.score, reverse=True)
        beam = expanded_ward[: self.beam_width]
        beam = self._apply_consistency_rerank(beam, q_type)

        # --- Street (cached per (province_id, ward_id)) ---
        street_cache: Dict[tuple, List[Candidate]] = {}
        expanded_street: List[BeamState] = []
        for st in beam:
            cache_key = (st.province_id, st.ward_id)
            if cache_key not in street_cache:
                province_name = self.provinces[st.province_id].name if st.province_id in self.provinces else None
                ward_name = self.wards[st.ward_id].name if st.ward_id in self.wards else None
                cleaned = self._remove_found_tokens_for_street(s_nodiac, ward_name, province_name)
                cleaned_collapsed = cleaned.replace(" ", "")
                street_cache[cache_key] = self._street_candidates(cleaned_collapsed, street_seg_nodiac)
            expanded_street.extend(self._expand_beam([st], street_cache[cache_key], "street"))

        expanded_street.sort(key=lambda x: x.score, reverse=True)
        final_beam = expanded_street[: self.beam_width] if expanded_street else beam
        best = final_beam[0]

        # --- Province fallback: infer from ward ---
        pid_out = best.province_id
        wid_out = best.ward_id
        if pid_out is None and wid_out is not None:
            w = self.wards.get(wid_out)
            if w is not None:
                pid_out = w.province_id

        province_name = self.provinces[pid_out].name if pid_out in self.provinces else None
        ward_name = self.wards[wid_out].name if wid_out in self.wards else None
        street_name = self.streets[best.street_id].name if best.street_id in self.streets else None

        return {
            "input": raw_text,
            "street": street_name,
            "ward": ward_name,
            "province": province_name,
            "ids": {
                "street_id": best.street_id,
                "ward_id": wid_out,
                "province_id": pid_out,
            },
            "debug_matches": [{"beam_score": best.score, "trace": " | ".join(best.trace)}],
        }

    def _remove_found_tokens_for_street(
        self,
        nodiac_spaced: str,
        ward_name: Optional[str],
        province_name: Optional[str],
    ) -> str:
        s = f" {nodiac_spaced} "
        for name in (ward_name, province_name):
            if not name:
                continue
            nm = basic_cleanup(strip_diacritics_keep_d(name))
            if nm:
                s = re.sub(rf"\b{re.escape(nm)}\b", " ", s)
        return _SPACES_RE.sub(" ", s).strip()


def build_beam_corrector(data_dir: str) -> BeamSearchAddressCorrector:
    province_path = os.path.join(data_dir, "list_province.csv")
    ward_path = os.path.join(data_dir, "list_ward.csv")
    street_path = os.path.join(data_dir, "list_street.csv")
    provinces = load_provinces(province_path)
    wards = load_wards(ward_path)
    streets = load_streets(street_path)
    return BeamSearchAddressCorrector(provinces, wards, streets)
