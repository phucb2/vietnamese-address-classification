import re
import unicodedata
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple

# =========================================================
# 1) NORMALIZE
# =========================================================

#_PUNCT_RE = re.compile(r"[,\.;:\(\)\[\]\{\}/\\\-\_]+")
_PUNCT_RE = re.compile(r"[,\.;:\(\)\[\]\{\}/\\\-\_]+")
_SPACES_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\b\d+\b")

# Expand/normalize common abbreviations (non-ML, rule-based)
_ABBR = {
    # cities
    "tphcm": "thanh pho ho chi minh",
    "tp.hcm": "thanh pho ho chi minh",
    "tp hcm": "thanh pho ho chi minh",
    "hcm": "ho chi minh",
    "hn": "ha noi",
    "dn": "da nang",

    # admin prefixes
    "tp": "thanh pho",
    "t.p": "thanh pho",
    "tỉnh": "tinh",
    "t.": "tinh",
    "p": "phuong",
    "p.": "phuong",
    "x": "xa",
    "x.": "xa",

    # street
    "đ": "duong",
    "đ.": "duong",
    "d.": "duong",
    "dg": "duong",
    "dg.": "duong",
    "đường": "duong",
}

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def basic_cleanup(s: str) -> str:
    s = nfc(s).lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACES_RE.sub(" ", s).strip()
    return s

def strip_diacritics_keep_d(s: str) -> str:
    """
    Bỏ dấu tiếng Việt. Map đ/Đ -> d để so khớp không dấu.
    """
    s = nfc(s)
    s = s.replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

def expand_abbr(s: str) -> str:
    s = basic_cleanup(s)
    # replace longer keys first
    for k, v in sorted(_ABBR.items(), key=lambda x: -len(x[0])):
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    s = _SPACES_RE.sub(" ", s).strip()
    return s
    
def fix_missing_space(s: str) -> str:
    # thêm space giữa chữ thường và chữ hoa (Việt/Latin)
    return re.sub(r'([a-zà-ỹ])([A-ZÀ-Ỹ])', r'\1 \2', s)
    
def normalize_text(s: str, remove_numbers: bool = False) -> Dict[str, str]:
    """
    Return:
      spaced: expanded + cleaned
      nodiac: no diacritics
      nodiac_collapsed: no diacritics + no spaces
    """
    s = fix_missing_space(s)   
    s = expand_abbr(s)

    if remove_numbers:
        s = _NUM_RE.sub(" ", s)
        s = _SPACES_RE.sub(" ", s).strip()

    nodiac = basic_cleanup(strip_diacritics_keep_d(s))
    return {
        "spaced": s,
        "nodiac": nodiac,
        "nodiac_collapsed": nodiac.replace(" ", ""),
    }

def drop_admin_prefix(name: str) -> str:
    """
    Drop common admin tokens at beginning to create more aliases.
    """
    s = basic_cleanup(name)
    tokens = s.split()
    while tokens and tokens[0] in {"tinh", "thanh", "pho", "phuong", "xa", "tp"}:
        first = tokens.pop(0)
        # handle "thanh pho" -> drop both
        if first == "thanh" and tokens and tokens[0] == "pho":
            tokens.pop(0)
    return " ".join(tokens).strip()

def make_aliases(name: str) -> List[str]:
    base = basic_cleanup(name)

    # remove prefix
    no_prefix = drop_admin_prefix(base) or base

    variants = set()

    for s in {base, no_prefix}:
        s = basic_cleanup(s)

        # ✅ dạng có khoảng trắng
        variants.add(s)

        # ✅ dạng không khoảng trắng
        variants.add(s.replace(" ", ""))

        # ✅ dạng không dấu
        nod = basic_cleanup(strip_diacritics_keep_d(s))
        variants.add(nod)
        variants.add(nod.replace(" ", ""))

    # ✅ QUAN TRỌNG: thêm alias remove prefix thủ công
    tokens = base.split()
    if tokens and tokens[0] in {"phuong", "xa", "thi tran"}:
        short = " ".join(tokens[1:])
        variants.add(short)
        variants.add(short.replace(" ", ""))

        nod_short = basic_cleanup(strip_diacritics_keep_d(short))
        variants.add(nod_short)
        variants.add(nod_short.replace(" ", ""))

    return [v for v in variants if len(v) >= 3]

# =========================================================
# 2) AHO-CORASICK (EXACT MULTI-PATTERN)
# =========================================================

class ACAutomaton:
    def __init__(self):
        self.next = [dict()]
        self.fail = [0]
        self.out = [[]]

    def add(self, pattern: str, payload):
        node = 0
        for ch in pattern:
            if ch not in self.next[node]:
                self.next[node][ch] = len(self.next)
                self.next.append(dict())
                self.fail.append(0)
                self.out.append([])
            node = self.next[node][ch]
        self.out[node].append((pattern, payload))

    def build(self):
        q = deque()
        # depth 1
        for ch, nxt in self.next[0].items():
            self.fail[nxt] = 0
            q.append(nxt)

        while q:
            r = q.popleft()
            for ch, u in self.next[r].items():
                q.append(u)
                f = self.fail[r]
                while f and ch not in self.next[f]:
                    f = self.fail[f]
                self.fail[u] = self.next[f].get(ch, 0)
                # merge output
                if self.out[self.fail[u]]:
                    self.out[u].extend(self.out[self.fail[u]])

    def find(self, text: str):
        """
        Return matches: (end_index, pattern_length, pattern, payload)
        """
        node = 0
        res = []
        for i, ch in enumerate(text):
            while node and ch not in self.next[node]:
                node = self.fail[node]
            node = self.next[node].get(ch, 0)
            if self.out[node]:
                for pattern, payload in self.out[node]:
                    res.append((i, len(pattern), pattern, payload))
        return res
    
# =========================================================
# 3) FUZZY (TRIGRAM INDEX + LEVENSHTEIN CUTOFF)
# =========================================================

def trigrams(s: str):
    s = s.replace(" ", "")
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s)-2)}

class TrigramIndex:
    def __init__(self):
        self.inv = defaultdict(list)  # trigram -> [alias_id,...]
        self.alias_text = {}          # alias_id -> text
        self.alias_meta = {}          # alias_id -> meta

    def add(self, alias_id: int, text: str, meta=None):
        self.alias_text[alias_id] = text
        self.alias_meta[alias_id] = meta
        for g in trigrams(text):
            self.inv[g].append(alias_id)

    def build(self):
        for g, lst in self.inv.items():
            self.inv[g] = list(set(lst))

    def candidates(self, query: str, top_k: int = 400):
        qg = trigrams(query)
        if not qg:
            return []
        counts = defaultdict(int)
        for g in qg:
            for aid in self.inv.get(g, []):
                counts[aid] += 1
        # top_k by overlap count
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

def levenshtein_cutoff(a: str, b: str, max_dist: int) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    if la == 0:
        return lb if lb <= max_dist else max_dist + 1
    if lb == 0:
        return la if la <= max_dist else max_dist + 1

    if lb > la:
        a, b = b, a
        la, lb = lb, la

    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)

    for i in range(1, la + 1):
        j_start = max(1, i - max_dist)
        j_end = min(lb, i + max_dist)

        cur[j_start - 1] = max_dist + 1 if j_start > 1 else i
        min_val = cur[j_start - 1]
        ai = a[i - 1]

        for j in range(j_start, j_end + 1):
            cost = 0 if ai == b[j - 1] else 1
            val = prev[j - 1] + cost
            pj1 = prev[j] + 1
            if pj1 < val:
                val = pj1
            cj1 = cur[j - 1] + 1
            if cj1 < val:
                val = cj1
            cur[j] = val
            if val < min_val:
                min_val = val

        if j_end < lb:
            cur[j_end + 1] = max_dist + 1

        if min_val > max_dist:
            return max_dist + 1

        prev, cur = cur, prev

    return prev[lb]


# =========================================================
# 4) DATA MODELS
# =========================================================

@dataclass(frozen=True)
class Province:
    id: str
    name: str

@dataclass(frozen=True)
class Ward:
    id: str
    name: str
    province_id: str  # ward -> province

@dataclass(frozen=True)
class Street:
    id: str
    name: str

@dataclass
class Match:
    level: str       # "province"|"ward"|"street"
    entity_id: str
    alias: str
    end: int
    length: int
    source: str      # "exact"|"fuzzy"
    dist: int = 0
    
# =========================================================
# 5) LOADER (TSV SIMPLE - MODIFY TO YOUR REAL FORMAT)
# =========================================================
import csv


def _detect_encoding_by_bom(path: str) -> str:
    with open(path, "rb") as bf:
        head = bf.read(4)

    # UTF-16 BOMs
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    # UTF-8 BOM
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    # No BOM: will try later
    return ""

def  _open_text_auto(path: str):
    # thử utf-8 trước
    try:
        return open(path, "r", encoding="utf-8", newline="")
    except UnicodeDecodeError:
        pass

    # thử utf-8-sig (Excel hay dùng)
    try:
        return open(path, "r", encoding="utf-8-sig", newline="")
    except UnicodeDecodeError:
        pass

    # fallback cho Windows Vietnamese
    return open(path, "r", encoding="cp1258", errors="replace", newline="")

def _read_tsv_dict(path: str, required_cols: list[str]):
    with _open_text_auto(path) as f:
        reader = csv.DictReader(f, delimiter=",")
        if not reader.fieldnames:
            raise ValueError(f"File {path} không có header.")

        # strip header spaces
        reader.fieldnames = [h.strip().lstrip("\ufeff") for h in reader.fieldnames]
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"File {path} thiếu cột {missing}. Header hiện có: {reader.fieldnames}")

        rows = []
        for row in reader:
            row = {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            if all(not v for v in row.values()):
                continue
            rows.append(row)
        return rows

def load_provinces(path: str):
    rows = _read_tsv_dict(path, ["province_id", "province_name"])
    return {
        r["province_id"]: Province(
            id=r["province_id"],
            name=r["province_name"]
        )
        for r in rows
    }

def load_wards(path: str):
    rows = _read_tsv_dict(path, ["ward_id", "ward_name", "province_id"])
    return {
        r["ward_id"]: Ward(
            id=r["ward_id"],
            name=r["ward_name"],
            province_id=r["province_id"]
        )
        for r in rows
    }

def load_streets(path: str):
    rows = _read_tsv_dict(path, ["street_id", "street_name"])
    return {
        r["street_id"]: Street(
            id=r["street_id"],
            name=r["street_name"]
        )
        for r in rows
    }
    
# =========================================================
# 6) CORRECTOR: province + ward + street
# =========================================================

class AddressCorrector:
    def _extract_ward_segment(self, raw_spaced: str) -> str:
        parts = [p.strip() for p in raw_spaced.split(",") if p.strip()]
    
        if len(parts) >= 2:
            return parts[1]   # phần sau dấu phẩy thứ 1
        
        return raw_spaced
    
    def __init__(self,
                 provinces: Dict[str, Province],
                 wards: Dict[str, Ward],
                 streets: Optional[Dict[str, Street]] = None):

        self.provinces = provinces
        self.wards = wards
        self.streets = streets or {}

        # mapping to restrict fuzzy ward by province
        self.province_to_wards = defaultdict(list)
        for wid, w in wards.items():
            self.province_to_wards[w.province_id].append(wid)

        self.ac = {
            "province": ACAutomaton(),
            "ward": ACAutomaton(),
            "street": ACAutomaton() if self.streets else None
        }

        self.fuzzy_idx = {
            "province": TrigramIndex(),
            "ward": TrigramIndex(),
            "street": TrigramIndex() if self.streets else None
        }

        self._build_indexes()

    def _build_indexes(self):
        alias_id = 1

        # province
        for pid, p in self.provinces.items():
            for a in make_aliases(p.name):
                # exact: use nodiac_collapsed
                a_norm = basic_cleanup(strip_diacritics_keep_d(a)).replace(" ", "")
                if len(a_norm) >= 3:
                    self.ac["province"].add(a_norm, ("province", pid, a_norm))

                # fuzzy: use nodiac spaced
                a_f = basic_cleanup(strip_diacritics_keep_d(a))
                self.fuzzy_idx["province"].add(alias_id, a_f, meta=("province", pid, a_f))
                alias_id += 1

        # ward
        """
        for wid, w in self.wards.items():
            for a in make_aliases(w.name):
                a_norm = basic_cleanup(strip_diacritics_keep_d(a)).replace(" ", "")
                if len(a_norm) >= 3:
                    self.ac["ward"].add(a_norm, ("ward", wid, a_norm))

                a_f = basic_cleanup(strip_diacritics_keep_d(a))
                self.fuzzy_idx["ward"].add(alias_id, a_f, meta=("ward", wid, a_f))
                alias_id += 1
        """
        # =========================
        # WARD INDEX (FIX FULL)
        # =========================
        
        SINGLE_PREFIX = {"phuong", "xa", "thi", "tran"}
        DOUBLE_PREFIX = {
            ("dac", "khu"),   # đặc khu
            ("thi", "xa"),    # thị xã
        }
        
        for wid, w in self.wards.items():
            base = basic_cleanup(w.name)
        
            # normalize không dấu để xử lý prefix
            tokens = base.split()
            tokens_norm = [basic_cleanup(strip_diacritics_keep_d(t)) for t in tokens]
        
            short_tokens = tokens  # mặc định giữ nguyên
        
            # ✅ xử lý prefix 2 từ (đặc khu, thị xã...)
            if len(tokens_norm) >= 2 and tuple(tokens_norm[:2]) in DOUBLE_PREFIX:
                short_tokens = tokens[2:]
        
            # ✅ xử lý prefix 1 từ (phường, xã...)
            elif tokens_norm and tokens_norm[0] in SINGLE_PREFIX:
                short_tokens = tokens[1:]
        
            # =========================
            # BUILD ALIASES
            # =========================
            aliases = set()
        
            # 1. full name
            variants = [" ".join(tokens)]
        
            # 2. short name (QUAN TRỌNG NHẤT)
            if short_tokens:
                variants.append(" ".join(short_tokens))
        
            for v in variants:
                v_clean = basic_cleanup(v)
                v_nod = basic_cleanup(strip_diacritics_keep_d(v_clean))
        
                # có dấu
                aliases.add(v_clean)
                aliases.add(v_clean.replace(" ", ""))
        
                # không dấu
                aliases.add(v_nod)
                aliases.add(v_nod.replace(" ", ""))
        
            # =========================
            # ADD TO INDEX
            # =========================
            for a in aliases:
                a_norm = a.replace(" ", "")
        
                if len(a_norm) >= 3:
                    self.ac["ward"].add(a_norm, ("ward", wid, a_norm))
        
                self.fuzzy_idx["ward"].add(
                    alias_id,
                    basic_cleanup(strip_diacritics_keep_d(a)),
                    meta=("ward", wid, a)
                )
        
                alias_id += 1

        # street (optional)
        if self.streets:
            for sid, s in self.streets.items():
                for a in make_aliases(s.name):
                    a_norm = basic_cleanup(strip_diacritics_keep_d(a)).replace(" ", "")
                    if len(a_norm) >= 3:
                        self.ac["street"].add(a_norm, ("street", sid, a_norm))

                    a_f = basic_cleanup(strip_diacritics_keep_d(a))
                    self.fuzzy_idx["street"].add(alias_id, a_f, meta=("street", sid, a_f))
                    alias_id += 1

        # build
        self.ac["province"].build()
        self.ac["ward"].build()
        if self.ac["street"]:
            self.ac["street"].build()

        self.fuzzy_idx["province"].build()
        self.fuzzy_idx["ward"].build()
        if self.fuzzy_idx["street"]:
            self.fuzzy_idx["street"].build()

    def _exact_matches(self, s_nodiac_collapsed: str) -> List[Match]:
        out = []
        for lvl in ("province", "ward"):
            found = self.ac[lvl].find(s_nodiac_collapsed)
            for end, length, pattern, payload in found:
                _lvl, eid, alias = payload
                out.append(Match(lvl, eid, alias, end, length, "exact", 0))

        if self.ac["street"]:
            found = self.ac["street"].find(s_nodiac_collapsed)
            for end, length, pattern, payload in found:
                _lvl, eid, alias = payload
                out.append(Match("street", eid, alias, end, length, "exact", 0))

        return out

    def _score(self, m: Match, text_len: int) -> float:
        # longer match is better
        score = 1.0 + 0.12 * m.length

        # near end bonus (province/ward often appear later; street often earlier)
        if m.source == "exact" and m.end >= 0:
            pos = m.end / max(1, text_len)
            if m.level in ("province", "ward"):
                score += 0.8 * pos
            else:  # street
                score += 0.5 * (1.0 - pos)

        # fuzzy penalty
        if m.source == "fuzzy":
            score -= 0.9 * m.dist
        return score

    def _fuzzy_pick(self, query_nodiac: str, lvl: str,
                    restrict_ids: Optional[set] = None,
                    max_dist: int = 3) -> Optional[Match]:
        idx = self.fuzzy_idx[lvl]
        if idx is None:
            return None

        cand = idx.candidates(query_nodiac, top_k=700)
        if not cand:
            return None

        best = (None, max_dist + 1, -1, None)  # aid, dist, overlap, (eid, alias)
        for aid, overlap in cand:
            _lvl, eid, alias = idx.alias_meta[aid]
            if restrict_ids is not None and eid not in restrict_ids:
                continue

            d = levenshtein_cutoff(query_nodiac, idx.alias_text[aid], max_dist=max_dist)
            if d <= max_dist:
                if (d < best[1]) or (d == best[1] and overlap > best[2]):
                    best = (aid, d, overlap, (eid, alias))

        if best[0] is None:
            return None

        eid, alias = best[3]
        return Match(lvl, eid, alias, end=-1, length=len(alias.replace(" ", "")), source="fuzzy", dist=best[1])

    def _pick_best_province_ward(self, matches: List[Match], text_len: int) -> Tuple[Optional[str], Optional[str]]:
        prov_ms = [m for m in matches if m.level == "province"]
        ward_ms = [m for m in matches if m.level == "ward"]

        best = (None, None, -1e9)

        # Prefer ward -> province
        for w in ward_ms:
            wid = w.entity_id
            wobj = self.wards.get(wid)
            if not wobj:
                continue
            pid = wobj.province_id
            s = self._score(w, text_len)

            # bonus if province matched consistently
            for p in prov_ms:
                if p.entity_id == pid:
                    s += self._score(p, text_len) + 0.5

            if s > best[2]:
                best = (pid, wid, s)

        # fallback province only
        if best[2] < -1e8 and prov_ms:
            p = max(prov_ms, key=lambda m: self._score(m, text_len))
            best = (p.entity_id, None, self._score(p, text_len))

        return best[0], best[1]

    def _extract_street_segment(self, raw_spaced: str) -> str:
        """
        Heuristic: street often before ward/province, split by comma.
        If no comma, use full string.
        """
        parts = [p.strip() for p in raw_spaced.split(",") if p.strip()]
        if not parts:
            return raw_spaced
        # commonly: "street, ward, province" => take first part as street segment
        return parts[0]

    def _remove_found_tokens_for_street(self, nodiac_spaced: str,
                                        ward_name: Optional[str],
                                        province_name: Optional[str]) -> str:
        """
        Remove ward/province from string to reduce noise when matching street.
        Work on nodiac_spaced.
        """
        s = " " + nodiac_spaced + " "
        for name in (ward_name, province_name):
            if not name:
                continue
            nm = basic_cleanup(strip_diacritics_keep_d(name))
            if not nm:
                continue
            # remove exact occurrences (spaced)
            s = re.sub(rf"\b{re.escape(nm)}\b", " ", s)
        s = _SPACES_RE.sub(" ", s).strip()
        return s

    def _pick_best_street(self, matches: List[Match], text_len: int) -> Optional[str]:
        street_ms = [m for m in matches if m.level == "street"]
        if not street_ms:
            return None
        best = max(street_ms, key=lambda m: self._score(m, text_len))
        return best.entity_id

    def correct(self, raw_text: str) -> dict:
        norm = normalize_text(raw_text)
        s_nodiac = norm["nodiac"]
        s_collapsed = norm["nodiac_collapsed"]

        # 1) Exact matches
        matches = self._exact_matches(s_collapsed)

        # 2) Pick province+ward using exact
        pid, wid = self._pick_best_province_ward(matches, text_len=len(s_collapsed))

        # 3) Fuzzy fallback for ward/province
        if wid is None:
            # Try fuzzy ward first (ward implies province)
            ward_seg = self._extract_ward_segment(norm["spaced"])
            ward_seg_norm = normalize_text(ward_seg)["nodiac"]

            # Try fuzzy ward on segment (ít noise hơn)
            m_w = self._fuzzy_pick(ward_seg_norm, "ward", max_dist=2)

            if m_w:
                matches.append(m_w)
                wid = m_w.entity_id
                if wid in self.wards:
                    pid = self.wards[wid].province_id

        if pid is None:
            m_p = self._fuzzy_pick(s_nodiac, "province", max_dist=2)
            if m_p:
                matches.append(m_p)
                pid = m_p.entity_id

            # If province found, try fuzzy ward restricted
            if wid is None and pid is not None:
                ward_seg = self._extract_ward_segment(norm["spaced"])
                ward_seg_norm = normalize_text(ward_seg)["nodiac"]

                restrict = set(self.province_to_wards.get(pid, []))

                #  EXACT match trước
                for wid_tmp in restrict:
                    w = self.wards[wid_tmp]
                    w_norm = normalize_text(w.name)["nodiac"]
                    if ward_seg_norm == w_norm:
                        wid = wid_tmp
                        matches.append(Match("ward", wid, w_norm, -1, len(w_norm), "exact", 0))
                        break
            
                # nếu chưa match → fuzzy
                if wid is None:
                    m_w2 = self._fuzzy_pick(
                        ward_seg_norm,
                        "ward",
                        restrict_ids=restrict,
                        max_dist=3
                    )
                    if m_w2:
                        matches.append(m_w2)
                        wid = m_w2.entity_id

        # repick after adding fuzzy
        pid, wid = self._pick_best_province_ward(matches, text_len=len(s_collapsed))

        province_name = self.provinces[pid].name if pid in self.provinces else None
        ward_name = self.wards[wid].name if wid in self.wards else None

        # 4) Street correction
        street_id = None
        street_name = None

        if self.streets:
            # reduce noise by removing ward/province tokens from nodiac text
            cleaned_for_street = self._remove_found_tokens_for_street(s_nodiac, ward_name, province_name)
            # use first segment primarily
            seg = self._extract_street_segment(cleaned_for_street)
            seg_norm = normalize_text(seg)["nodiac"]
            seg_collapsed = seg_norm.replace(" ", "")

            # exact street match on segment collapsed
            street_matches = []
            for end, length, pattern, payload in self.ac["street"].find(seg_collapsed):
                _lvl, sid, alias = payload
                street_matches.append(Match("street", sid, alias, end, length, "exact", 0))

            # fuzzy fallback on segment
            if not street_matches:
                m_s = self._fuzzy_pick(seg_norm, "street", max_dist=3)
                if m_s:
                    street_matches.append(m_s)

            street_id = self._pick_best_street(street_matches, text_len=len(seg_collapsed))
            if street_id and street_id in self.streets:
                street_name = self.streets[street_id].name
        else:
            # no street dictionary: just extract first segment as raw street
            street_name = self._extract_street_segment(norm["spaced"])

        return {
            "input": raw_text,
            "street": street_name,
            "ward": ward_name,
            "province": province_name,
            "ids": {"street_id": street_id, "ward_id": wid, "province_id": pid},
            "debug_matches": [
                {
                    "level": m.level, "id": m.entity_id, "alias": m.alias,
                    "source": m.source, "len": m.length, "end": m.end, "dist": m.dist
                }
                for m in matches
            ],
        }
        
# =========================================================
# 7) CLI RUNNER
# =========================================================
import os

def build_corrector(data_dir: str) -> AddressCorrector:
    
    province_path = os.path.join(data_dir, "list_province.csv")
    provinces = load_provinces(province_path)
    
    ward_path = os.path.join(data_dir, "list_ward.csv")
    wards = load_wards(ward_path)
    
    street_path = os.path.join(data_dir, "list_street.csv")
    streets = load_streets(street_path)

    return AddressCorrector(provinces, wards, streets)

if __name__ == "__main__":
    
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    text = "PhanVăn Trí, Thuần Giao, Hồ Chi Minh City"

    corr = build_corrector(data_dir)
    out = corr.correct(text)

    print("INPUT   :", out["input"])
    print("STREET  :", out["street"])
    print("WARD    :", out["ward"])
    print("PROVINCE:", out["province"])
    print("IDS     :", out["ids"])