# XGBDimerScorer — Filter & QC Concepts

이 문서는 XGBDimerScorer에서 사용하는 모든 QC/필터 개념을 정리. H1335 분석 (HCMV Merlin Fab + gHgLgO) 에서 발견한 노하우 기반.

---

## 1. CDR-clean (Ab-Ag 전용)

**문제**: xgb_v8 reranker는 "atom contact 많을수록 좋음"으로 학습됨. Fab이 비스듬히 누워서 **framework로도 antigen에 contact**하는 비정상 pose도 xgb 점수가 높게 나옴.

**실제 사례 (H1335)**:
- xgb top-1 모델 (`af3_selftmpl/af3 s16-0`, xgb=0.407, iptm=0.70):
  - `aadars_n_framework = 230` (정상은 ~150)
  - `aadars_n_cdr_h3 = 251` (정상은 ~50-150 atom-contact pair)
  - 즉 framework atom이 H3 못지않게 antigen이랑 contact → **CDR-driven binding이 아니라 face-laying mis-binding**

**해결**: 5조건 동시 만족하는 모델만 통과.

| 조건 | 기준 | 의미 |
|---|---|---|
| `aadars_frac_framework ≤ 0.30` | framework atom-contact 비율 ≤ 30% | framework가 paratope 점유 안 함 |
| `aadars_frac_cdr_h3 ≥ 0.20` | H3 atom-contact 비율 ≥ 20% | H3가 dominant CDR (정상 paratope의 ~30%) |
| `iface_n_contacts_total ≥ 100` | 총 heavy atom contact pair ≥ 100 | 진짜 interface 존재 |
| `mpnn_iface_n_residues ≥ 20` | MPNN-defined interface residue ≥ 20 | MPNN 기준으로도 interface 인정 |
| `n_chain_breaks == 0` | chain break 없음 | 구조 무결성 |

**H1335 통계**: 19,356 → CDR-clean **923 (4.8%)**

**Module**: `xgbdimer/filter/ab_clean.py` `apply_cdr_clean()`

**시각화** (3D Fab COM scatter):
-  — funnel 단계별 모델 분포

---

## 2. STRICT filter (CDR-clean + 추가 sanity)

CDR-clean 위에 method-level + 구조 검증 추가:

```
STRICT =
  CDR-clean (위 5조건)
+ atom_integrity_qc 통과 (severe bond 이상 없음)
+ method != 'of3' (gO 90-155 Å misalign)
+ gH-aligned gO core RMSD ≤ 6 Å (Ab-Ag에서만, template 검증 가능 시)
+ gH-aligned gO N-term RMSD ≤ 15 Å (hypervariable 영역, 완화)
+ per-residue atom completeness ≥ 98%
```

**H1335 통계**: 19,356 → **STRICT 807 (4.2%)**

**Module**: `xgbdimer/filter/ab_clean.py` `apply_strict_filter()`

**왜 of3 제외**:

| 검증 | of3 결과 | 결론 |
|---|---|---|
| Stage1 ABC global RMSD (gO) | median 26.2 Å | 모든 method 중 최악 |
| gH-aligned gO core RMSD | 90-155 Å | gO 위치 완전 misalign (catastrophic) |
| disulfide topology | gO-gH 잘못 형성 | chain mapping 자체 깨짐 |

---

## 3. Atom integrity (pt2 terminal O bug)

**문제**: Protenix-2 (pt2)가 종종 **C-말단 residue의 backbone O atom을 30-150 Å 떨어진 곳**에 둠. 모든 chain의 마지막 residue 영향 (gO/A.439, gH/B.692, gL/C.248, Fab-L/D.214, Fab-H/E.227 등).

**검출**:
1. **자동 sanitize** — feature extraction 단계 (base.py)에서 `sanitize_cutoff=10.0 Å` 기준으로 atom drop. 분석용 좌표는 깨끗.
2. **원본 cif에는 그대로 남아있음** — 시각화/제출 시 fix 필요.
3. **부산물**: 마지막 직전 residue (예: 691)와 마지막 residue (692) 사이 **peptide C-N bond가 0.9-1.2 Å로 압축** (정상 1.33 Å). terminal O 위치 fix만으로는 해소 안 됨.

**Fix tool**: `xgbdimer/pick/collect.py` 가 옵션으로 terminal O를 geometric 정상 위치 (~1.4 Å from C) 로 재배치 가능. 또는 H1335 `scripts/45_fix_terminal_atoms.py` 직접 사용.

**처리 방침**:
- STRICT 필터에서 **method != 'pt2'** 옵션 권장 (atom fix 부담 회피)
- 또는 pt2 채택 시 `xgbdimer pick collect --fix-terminal-o` 사용

---

## 4. Steric clash (1.7 Å threshold, disulfide-aware)

**무엇**: 같은 모델 내 heavy atom pair distance < 1.7 Å 카운트.

**제외**:
- Same residue (e.g., CB-CG ~1.5 Å 정상)
- Adjacent same-chain residue (peptide bond region)
- **Disulfide bonds** (CYS-SG ↔ CYS-SG up to 2.5 Å — 정상 disulfide)

**Inter-chain clash 별도 카운트**: 서로 다른 chain 사이 clash는 더 심각 (Fab heavy chain backbone과 antigen backbone 충돌 등). `n_inter_clash_17` 컬럼.

**H1335 통계**:

| Method | 평균 clash<1.7Å (total) | 평균 inter-chain |
|---|---|---|
| af3 | 6.5 | 6.1 |
| bt2_jax | 7.7 | **2.8** (최저) |
| pt2 | 36.7 | **27.9** (catastrophic) |
| of3 | 6.5 (단 gO 위치 90Å+) | 6.1 |

**해석**: pt2는 atom-level integrity가 다른 method 대비 5배 안 좋음. bt2_jax는 iptm 높을 때 inter-chain clash가 가장 적음 (Boltz 학습이 backbone integrity 보존하는 경향).

**Module**: `xgbdimer/filter/clash.py` `detect_clashes()` + `run_clash_audit_parallel()`

---

## 5. Template-aligned regional RMSD

**왜 필요**: Stage1의 `sup_rmsd_A` (ABC global Kabsch RMSD)는 misleading. **gH/gL이 약간 misalign되면 gO RMSD가 부풀려짐**.

**H1335 사례**: 
- `af3_selftmpl_mt/bt2_jax s20-2`: ABC global sup_rmsd_A = **19 Å** (탈락 의심) → gH-only Kabsch 후 gO core RMSD = **3.36 Å** (정상)
- `af3_7lbe_mt/af3 s426-1`: 19 Å → **1.63 Å**
- `of3` 모델들: gH-aligned도 **90-155 Å** (진짜 catastrophic)

**올바른 방법**:
1. **Anchor chain (sequence 100% identical)** 만 사용해서 Kabsch align (gH=H1335.B vs 7LBE.A, identity 100%)
2. **Probe chain (검증 대상)** 의 좌표에 같은 transform 적용
3. Probe chain의 region별 RMSD 측정 (HCMV gO는 N-term 1-80 hypervariable + core 200-440 보존)

**Module**: `xgbdimer/filter/template_rmsd.py`
- `kabsch(P, Q)` — 표준 Kabsch SVD
- `regional_rmsd(pred_cif, ref_cif, anchor_chain_pred, anchor_chain_ref, probe_chain_pred, probe_chain_ref, regions)` — anchor align + probe regional RMSD
- residue number offset (±50) 자동 탐색 (signal peptide / numbering 차이 보정)

**사용 예** (H1335):
```python
from xgbdimer.filter.template_rmsd import regional_rmsd
r = regional_rmsd(
    pred_cif='.../model.cif',
    ref_cif='/data/casp17/target_notes/H1335/structures/7LBE.cif',
    anchor_chain_pred='B', anchor_chain_ref='A',          # 둘 다 gH (100% identical)
    probe_chain_pred='A', probe_chain_ref='C',            # 둘 다 gO
    regions={'Nterm': (1, 80), 'core': (200, 440)},
)
# → {'anchor_rmsd': 0.41, 'Nterm': 1.34, 'core': 1.63, 'full': 1.53, ...}
```

---

## 6. PPI-clean (generic 모드)

Ab-Ag와 달리 PPI는 CDR 없음. 더 단순한 sanity:

| 조건 | 기준 |
|---|---|
| `iface_n_contacts_total ≥ 80` | 진짜 interface 존재 |
| `n_chain_breaks == 0` | 구조 무결성 |
| `mpnn_iface_n_residues ≥ 15` | MPNN-confirmed interface |
| `json_iptm ≥ 0.3` | 최소 confidence |

**Module**: `xgbdimer/filter/ppi_clean.py` `apply_ppi_clean()`

---

## 7. Cluster-level 정규화 (size bias 보정)

**문제**: cluster 크기 매우 다른 경우 (H1335: C4=6,458 vs C5=1,007), naive "top xgb per cluster"는 큰 cluster에 편향.

**3가지 정규화 지표**:

| 지표 | 의미 |
|---|---|
| `clean_rate_%` | CDR-clean 통과 비율 (cluster 내 양질 모델 비중) |
| `xgb≥0.30_rate_%` | xgb_v8_pred ≥ 0.30 통과 비율 (high-quality 비중) |
| `xgb_top5_mean` | top 5 모델의 xgb 평균 (top 안정성) |

**H1335 결과**:
| Cluster | EC | n | xgb≥0.30_rate | 해석 |
|---|---|---|---|---|
| C5 | gO N-term | 1,007 | **14.3%** | 작은 cluster but 압도적 high-quality |
| C0 | EC4a | 1,813 | 3.8% | |
| C4 | gO core | 6,458 | 0.7% | 큰 cluster but 평범 |
| C1 | EC2 13H11 | 3,129 | **0%** | trap — 문헌 매칭 잘 되어 보이지만 quality 0 |

**Module**: `xgbdimer/pick/per_cluster.py` `compute_cluster_quality_stats()`

---

## 8. STRICT 재클러스터링 (silhouette gain)

원본 19,356 모델에 KMeans k=6 = silhouette **0.545**.
**STRICT 807 모델만** 재클러스터링 → KMeans k=6 = silhouette **0.757** (gain +0.21).

**왜**: 깨끗한 모델만 모이면 cluster boundary가 sharp해짐. Outlier가 cluster mean을 흐리지 않음.

**Module**: `xgbdimer/cluster/com_kmeans.py` — `cluster_com()` 에 raw 한 번 + STRICT 후 한 번 더 호출.

---

## Filter funnel 시각화

`xgbdimer/viz/_3d.py` `plot_filter_funnel(stats_history, output_png)` 가 각 필터 단계별 모델 수 감소를 막대그래프로 그림. H1335 사례:

```
원본:                19,356  ████████████████████
CDR-clean:              923  █
 + atom integrity:      923  █
 + no of3:              808  █
 + gO core ≤ 6Å:        807  █
 + gO N-term ≤ 15Å:     807  █
 + atom completeness:   807  █
─────────────────────────────────
STRICT 통과율 4.2%
```

---

## 빠른 참조

| 시나리오 | 권장 filter |
|---|---|
| Ab-Ag, conservative submission | CDR-clean + STRICT + method≠pt2 + method≠of3 |
| Ab-Ag, broader candidate pool | CDR-clean only |
| PPI generic | PPI-clean + clash<3 inter |
| Quick QC (atom 깨진 모델만 제외) | clash<5 inter + atom_completeness≥98% |

---

자세한 의사결정 사례 (H1335 5 picks 선정):
- `/data/casp17/playground/gunwoong_analysis/H1335/analysis/REPORT.md §13`
