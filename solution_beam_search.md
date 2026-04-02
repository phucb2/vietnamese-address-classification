# Purpose
Use this file when implementing or reviewing the beam-search address correction pipeline for noisy Vietnamese input.

# Beam Search Address Correction (End-to-End)

## 1) Problem and Goal
- Input is a noisy free-text address with typos, missing spaces, abbreviation, and Vietnamese diacritic/tone errors.
- Output is the most likely normalized tuple: `street`, `ward`, `province` (+ ids).
- Main goal: avoid greedy errors where one wrong early match causes a wrong final result.

## 2) Why Beam Search
- Greedy decoding picks one best candidate per stage and may lock into wrong branch.
- Beam search keeps top `k` hypotheses and delays final decision until more evidence is available.
- This is important when ward names are ambiguous across provinces.

## 3) Data Structures

### 3.1 Canonical entity storage
- `Province(id, name)`
- `Ward(id, name, province_id, admin_type, core_name)`
- `Street(id, name)`

Notes:
- `admin_type`: one of `xa`, `phuong`, `thi_tran`, `dac_khu`, `thi_xa`.
- `core_name`: ward name after dropping admin prefix (for alias matching).

### 3.2 Indexes
- `AC` exact index (nodiac + collapsed) for fast substring hit.
- `TrigramIndex` for fuzzy retrieval.
- `ward_by_province: Dict[province_id, List[ward_id]]` for hard restriction.
- `ward_core_map: Dict[core_name, List[(ward_id, province_id, admin_type)]]`.
- Optional: `char_confusion_map` for Vietnamese typo/tone confusion.

## 4) Normalization Pipeline
- Unicode NFC normalize.
- Expand abbreviations: `tp`, `t.p`, `dg`, `đ.`...
- Remove punctuation noise.
- Produce 3 variants:
  - `spaced`
  - `nodiac`
  - `nodiac_collapsed`
- Create extra robust variants for missing tone marks and keyboard mistakes (section 8).

## 5) Candidate Generation

### 5.1 Province candidates
- Use exact AC over full `nodiac_collapsed`.
- Fallback fuzzy from trigram candidates + bounded edit distance.
- Keep top `N_p` candidates with confidence score.

### 5.2 Ward candidates
- Extract ward segment (comma split heuristic + full fallback).
- If province candidates exist: generate ward candidates inside each province only.
- If province is weak/unknown: use global ward candidates but with lower prior.
- Keep top `N_w` per province branch.

### 5.3 Street candidates
- Remove selected ward/province tokens from text.
- Extract street segment and generate exact/fuzzy candidates.
- Keep top `N_s`.

## 6) Beam State and Decoding

### 6.1 Beam state
- `State = {province_id?, ward_id?, street_id?, score, trace}`
- `trace` stores why score is added/penalized.

### 6.2 Expansion order
1. Start from root state.
2. Expand province candidates.
3. Expand ward candidates conditioned on province branch.
4. Expand street candidates conditioned on current `(province, ward)`.
5. After each stage, prune to `beam_width`.

### 6.3 Recommended defaults
- `beam_width = 8`
- `N_p = 5`, `N_w = 8`, `N_s = 5`
- hard constraint: drop state if `ward.province_id != province_id`.

## 7) Scoring Function

`TotalScore = S_exact + S_fuzzy + S_position + S_type + S_consistency + S_coverage - penalties`

Suggested terms:
- `S_exact`: strong bonus for exact AC hit.
- `S_fuzzy`: inverse of edit distance and trigram overlap.
- `S_position`: mild bonus by expected position (street early, ward/province later).
- `S_type`: bonus when query contains `xã/phường/...` and matched `admin_type`.
- `S_consistency`: bonus for coherent `(ward, province)`.
- `S_coverage`: bonus when selected names explain more query tokens.
- Penalties:
  - high edit distance
  - admin type mismatch (`xã` vs `phường`)
  - cross-branch contradiction.

## 8) Vietnamese Missing/Incorrect Tone Handling (`~`, `?`, etc.)

Tone errors are common: missing dấu, wrong dấu (`hỏi/ngã` swap), mixed keyboard artifacts.

### 8.1 Practical strategy
- Always match in `nodiac` space for primary recall.
- Add a small set of tone-robust rewrite variants before fuzzy match.
- Keep candidate count bounded; do not explode all combinations.

### 8.2 Confusion classes (lightweight)
- Vowels often confused by typing:
  - `a, ă, â`
  - `e, ê`
  - `o, ô, ơ`
  - `u, ư`
  - `i, y`
  - `d, đ`
- Tone confusion:
  - missing tone completely
  - `hỏi` vs `ngã` swap (very common)

### 8.3 Implementation pattern
1. Normalize input to `nodiac`.
2. Generate up to `M` robust query variants (for example `M=4`):
   - original nodiac
   - split/merge correction variant
   - abbreviation-expanded variant
   - common confusion rewrite variant
3. Run candidate retrieval for each variant.
4. Merge candidates by entity id; keep best score.

### 8.4 Why this works
- Most tone issues disappear in nodiac matching.
- Remaining errors are handled by edit distance + controlled variant generation.
- Avoids expensive full Vietnamese phonetic generation.

## 9) Step-by-Step Example

Input:
`"chumạnhtrinh,xãtràlinh,t.pduongànẵng"`

Expected:
- street: `Chu Mạnh Trinh`
- ward: `Xã Trà Linh`
- province: `Tp Đà Nẵng`

### Step 1: Normalize
- `spaced`: `"chu mạnh trinh, xã trà linh, thanh pho duong a nang"` (example cleanup)
- `nodiac`: `"chu manh trinh xa tra linh thanh pho duong a nang"`
- `collapsed`: `"chumanhtrinhxatralinhthanhphoduonganang"`

### Step 2: Province candidate generation
- top candidates:
  - `Tp Đà Nẵng` (score 7.2)
  - `Tp Hải Phòng` (score 5.9)
  - `Tỉnh Quảng Ngãi` (score 4.3)

Beam after province (`k=2` for illustration):
- B1: `{province=Đà Nẵng, score=7.2}`
- B2: `{province=Hải Phòng, score=5.9}`

### Step 3: Ward expansion per province
- From B1 restricted wards of Đà Nẵng:
  - `Xã Trà Linh` (dist 1, +type match `xã`) -> +6.0
- From B2 restricted wards of Hải Phòng:
  - best ward unrelated, weaker -> +3.1

Beam:
- B1: `{Đà Nẵng, Trà Linh, score=13.2}`
- B2: `{Hải Phòng, Đường An, score=9.0}`

### Step 4: Street expansion
- B1 finds `Chu Mạnh Trinh` exact/fuzzy high -> +7.0
- B2 finds weaker street candidate -> +4.2

Final:
- B1 total = 20.2
- B2 total = 13.2
- Choose B1.

## 10) Another Example for Tone/Mark Noise

Input:
`"xã pà vầy sủ,tinh uy ên quang"`

Common issue:
- Province text is corrupted (`uy ên` vs `tuyên`).

How beam + robust variants help:
1. `nodiac` reduces tone dependence.
2. Province candidates include `Tỉnh Tuyên Quang` with moderate fuzzy score.
3. Ward candidate `Xã Pà Vầy Sủ` strongly supports province branch consistency.
4. Joint score of `(ward, province)` wins over unrelated high-frequency wards.

## 11) Complexity and Performance
- AC matching: near linear in text length.
- Fuzzy retrieval: controlled by trigram top-k.
- Beam decode complexity:
  - roughly `O(beam_width * (N_p + N_w + N_s) * rerank_cost)`.
- With defaults above and current dictionary size, this should remain practical for batch benchmark.

## 12) Rollout Plan
1. Add beam decoder while reusing current indexes.
2. Add `admin_type` extraction and mismatch penalty.
3. Add robust variant generator for Vietnamese tone/typing noise.
4. Benchmark and tune weights on failure subset first.
5. Re-run full benchmark and compare per-field accuracy.

## 13) Minimal Acceptance Criteria
- Province accuracy improves on current failed set.
- Ward accuracy improves significantly (primary KPI).
- No large regression in street accuracy.
- Runtime increase remains acceptable for batch benchmark.
