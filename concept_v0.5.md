# RISE-SPEC — Is AI-ECG "early detection" target-specific within the same patient?

Status: concept / pre-registration draft · v0.5 · 2026-10-30 (change log: §11)
Scope rule: minimum design that answers one question. Anything not listed here goes to §9 and needs an explicit decision before it is added.

## 1. Thesis

AI-ECG papers routinely read "false positives who later develop the target disease" as pre-clinical detection [1–3]. Croon et al. showed that disease-specific AI-ECG models behave as broad cardiovascular risk markers [4], but with one ECG per person: a between-person design cannot separate a stable high-risk *trait* from a developing *target disease*. A within-patient lead–lag design can, because each patient serves as their own control and stable between-person risk drops out. Either result is publishable: a target-specific rise supports the early-detection claim; a non-specific rise extends [4] with within-person evidence. Non-specificity is claimed only through a pre-specified equivalence test; a result too imprecise for either reading is reported as inconclusive (§4).

## 2. Research question

In adults with serial sinus-rhythm ECGs, does an AI-ECG LVSD score rise within the patient before incident LVSD more steeply than before (a) incident AF and (b) hip fracture?

## 3. Gap

- The false-positive → future-disease argument is repeated across models and reviews [1–3].
- Non-specificity has been shown between persons only [4]. No within-patient, event-anchored specificity test was found (Consensus literature check, 30 Oct 2026).
- EchoNext-Mini has been validated in MIMIC-IV, but cross-sectionally: one ECG per patient, no serial or pre-diagnostic analysis [10].

## 4. Pre-specified hypotheses and readout

- **H1 (target-specific):** β_LVSD > β_AF and β_LVSD > β_HIPFX, with β_LVSD > 0. Tested as an intersection–union test: all three conditions must hold at two-sided α = 0.05, so no multiplicity adjustment is needed. β_LVSD > 0 is necessary but never sufficient (§6.6).
- **H0 (non-specific):** pre-event slopes are equivalent across event types (within ±Δ); groups differ in level only.

**Equivalence margin Δ** (logit/year), fixed at G2 before any slope is estimated:

    Δ = 0.10 × D_G2 / 3

D_G2 = mean logit(p) of prevalent-LVSD ECGs minus mean logit(p) of the other ECGs in the G2 sample (§7). Reading: a slope difference smaller than 10% of a full case–control transition spread over the 3-year window is negligible. Coded labels shrink D_G2, which makes `≡` harder to reach, i.e., conservative for H0.

Notation: subscript HIPFX and the letter H stand for the non-cardiac control group. After G1 (§7) this group is all limb-fracture admissions (LIMBFX).

**Contrast classes**, for each of L−A = β_LVSD − β_AF, L−H = β_LVSD − β_HIPFX, A−H = β_AF − β_HIPFX:

| Class | Rule |
|---|---|
| `≡` | 90% CI within (−Δ, +Δ) (TOST, α = 0.05) |
| `>` / `<` | 95% CI excludes 0 (above / below), and not `≡` |
| `?` | none of the above |

**Readout**, rules applied in order; rule 5 makes the table exhaustive:

| # | Pattern | Reading |
|---|---|---|
| 1 | L−A `>`, L−H `>`, β_LVSD 95% CI > 0 | Target-specific pre-clinical signal (H1) |
| 2 | L−A `≡`, L−H `>`, A−H `>` | Cardiac-generic, not LV-specific (qualified by §8, AF comparator) |
| 3 | L−A `≡`, L−H `≡`, A−H `≡` | Generic pre-event drift / acquisition context (H0) |
| 4 | L−A `<` | Score rises more before AF than before LVSD; against H1 |
| 5 | Anything else | Inconclusive or unanticipated: report all contrasts with CIs; no specificity or non-specificity claim |

## 5. Data

- MIMIC-IV-ECG 1.0 (waveforms, machine measurements, machine report text) + MIMIC-IV 3.1 `hosp` (`patients`, `admissions`, `diagnoses_icd`) + MIMIC-IV-ED 2.2 (`edstays`, for acquisition context only). MIMIC-IV-ECHO 1.0 (`structured-measurement`) for the echo-anchored sensitivity analysis S4 only.
- All processing is local (PhysioNet DUA). Only aggregate outputs leave the machine.

## 6. Design

### 6.1 Score

- EchoNext-Mini [9], the publicly released multi-label model (GitHub `PierreElias/IntroECG`, folder `7-EchoNext Minimodel`, `models/echonext_multilabel_minimodel/weights.pt`), output column `lvef_lte_45` (1 of 12), frozen: no training, no fine-tuning. The full EchoNext model [5] has no public weights. Developed on Columbia data (2008–2022) [6], so there is no overlap with MIMIC.
- Waveform input: 10-s 12-lead at 250 Hz (downsample MIMIC 500 Hz), array N×1×2500×12 in the model's lead order, normalized with the released `waveform_normalization_params.json`.
- Tabular input (required, 7 features, standardized with the released `tabular_transformer.joblib`): sex, age, ventricular rate, atrial rate, PR, QRS, QTc. In MIMIC-IV-ECG, intervals are derived from fiducial points, QTc by Bazett, and atrial rate is set to 0 because it is not reported, following [10].
- Freeze the age input at the patient's first eligible ECG to remove mechanical age drift. Rate, PR, QRS and QTc vary with each ECG and are part of the score.
- Analysis scale: logit(p) as released, no recalibration. The released component outputs overestimate risk [10], but prior-shift recalibration adds a constant and Platt scaling rescales every logit by the same factor; Δ is scaled to D_G2 (§4), so neither changes the readout.

### 6.2 Anchors

First admission (`admittime`) carrying a code from one set (MIMIC stores codes without dots):

| Group | ICD-10 | ICD-9 |
|---|---|---|
| LVSD (target) | I50.2x, I50.4x | 428.2x, 428.4x |
| AF (cardiac comparator) | I48.x | 427.31, 427.32 |
| HIPFX (non-cardiac control, pre-specified) | S72.0x–S72.2x | 820.x |
| LIMBFX (non-cardiac control, applied after G1) | S42, S52, S62, S72, S82, S92 | 810–829 |

- Fracture anchors need an ICD-10 initial-encounter 7th character (A/B/C); any code of the set counts as history. A patient whose first fracture-coded admission carries only subsequent-encounter or sequela codes is excluded, because the fracture happened earlier, elsewhere.

- Incident means no earlier admission with any code from the three sets. For LVSD, also no earlier I50.x/428.x or I42.x/425.x.
- If the anchor admission carries codes from more than one set, exclude the patient. Together with the previous rule, this leaves the LVSD group without known AF (§8).
- Exclude AF and HIPFX patients with an LVSD code ≤1 year after anchor.
- Coverage: the anchor admission must have ≥1 ECG (any rhythm) between 30 days before admission and discharge. This shows that MIMIC-IV-ECG reaches the anchor date for that patient; ECGs end around 2019 while admissions run to about 2022 (`checks/00_data_checks.json`).

### 6.3 ECG eligibility

- Window: −3 y to −30 d before anchor (pre-specified); −5 y to −30 d applied after G1 (§7). The last 30 days are dropped because they capture the acute presentation.
- Sinus rhythm and not paced, from the machine statements: ≥1 statement names a sinus rhythm (rhythm, bradycardia, tachycardia, arrhythmia) and none names AF, flutter, pacing, a junctional, ectopic atrial, idioventricular or supraventricular rhythm, an undetermined rhythm or AV dissociation. Exact patterns: `analysis/01_cohort.py`.
- No flat or missing lead: no machine statement of an unsuitable recording or lead, or of lead reversal; in the waveform, 5000×12 samples, no NaN sample, and peak-to-peak ≥0.05 mV in every lead (0.05 mV falls in an empty gap of the distribution in 3,000 random ECGs).
- A patient enters if they have ≥2 eligible ECGs spanning ≥180 days and are aged ≥18 at anchor.

### 6.4 Primary analysis

Patient fixed-effects (within) regression, with t = years relative to anchor:

    logit(p_ij) = α_i + Σ_g β_g · t_ij · 1[g_i = g] + γ · t_ij · age_c,i + δ · t_ij · male_i + ε_ij

α_i absorbs every stable between-patient difference (level/trait). SEs are cluster-robust by patient.

**Primary estimand:** β_LVSD − β_AF (logit/year, 95% CI). This is the headline number; H1 also requires β_LVSD − β_HIPFX (§4).
Readout contrasts: β_LVSD − β_HIPFX and β_AF − β_HIPFX.

β_g averages the rise over the whole window. A rise confined to the last months is diluted, so the last-bin event-study contrast (§6.5) is the key secondary.

### 6.5 Secondary

- **Event study:** the same model with time bins (−36…−24 [ref], −24…−12, −12…−6, −6…−1 months) × group, to show *when* any rise starts.
- **Key secondary:** LVSD − AF difference in the −6…−1 month bin (vs reference), 95% CI.
- **Level:** mean logit(p) in the earliest bin per group. This is the between-person, Croon-type elevation.

### 6.6 Critical controls (one per alternative explanation)

| Alternative explanation | Control |
|---|---|
| EchoNext labels count an ECG as positive up to 1 y before the echo [5,6,8], so some pre-event rise is built in | Contrast against comparator events, never against zero |
| Sicker acquisition context: ED/inpatient ECGs cluster before events, and the model performs worse on inpatient/ICU ECGs [10] | **S1:** only ECGs outside any ED stay or admission |
| New conduction disease (LBBB/IVCD) raises LVSD scores | **S2:** only ECGs with QRS <120 ms |
| Patients with many ECGs (sicker, more contact) dominate the slopes | **S3:** each ECG weighted 1/n_i, so every patient counts equally |
| Rhythm or pacing transitions move scores [7] | Excluded at eligibility (sinus, non-paced only) |
| The model's age input rises with time | Age input frozen (§6.1) |
| Undiagnosed LVSD in the AF group (shared substrate) | AF patients with an LVSD code ≤1 y after anchor excluded (§6.2); a rule-2 reading is qualified (§8) |
| ICD-coded LVSD anchors are late or noisy and are not echo LVEF ≤45% | **S4:** LVSD anchored at the first TTE with LVEF ≤45% that follows a TTE with LVEF >45% (see below); AF and HIPFX unchanged |

S4 definition: MIMIC-IV-ECHO 1.0 `structured-measurement`, `test_type = 'tte'`. The two source systems record LVEF differently, so one value per study is taken in this order: `lvef` (the midpoint with `lvef_upper` when both exist and differ by 0–10 points; a wider or reversed range makes the study indeterminate), else `biplane_lvef`, else `lvef_3d`. PhysioNet does not define `lvef_upper`; locally its median difference from `lvef` is 5 points, i.e., a reported range. Values outside 5–85% are removed. The anchor is the TTE date. The other §6.2 rules apply, and coverage requires ≥1 ECG within ±30 days of that TTE. Because LVSD is then echo-anchored and the comparators are code-anchored, S4 qualifies the readout but cannot replace it.

S1–S4 report the same contrasts. They qualify the primary readout but do not override it.

## 7. Work packages and gates

- **WP1 — Cohort & anchors** (DuckDB over MIMIC tables).
  **G1 (outcome-blind, counts only):** ≥500 patients in each of LVSD, AF and HIPFX. If not met, widen the window to −5 y once (all groups). If HIPFX still fails, switch to all limb-fracture admissions (§10) before any scoring. If LVSD or AF still fails, stop.
  **G1 result (30 Oct 2026, `results/wp1_flow_*.json`):** −3 y: LVSD 466, AF 1,678, HIPFX 183 (fail). −5 y: LVSD 579, AF 2,089, HIPFX 244 (control fails). −5 y with LIMBFX: LVSD 557, AF 2,006, LIMBFX 534 (pass). The study proceeds with the −5 y window and LIMBFX. S4 (same rules): 292 echo-anchored LVSD patients.
- **WP2 — Scoring** all eligible ECGs. First check the implementation against the released example (`EchoNext_Mini_example.ipynb`).
  **G2 (transport check):** using one random sinus, non-paced ECG per adult patient in MIMIC-IV-ECG, the AUROC of the `lvef_lte_45` output for a prevalent LVSD code (I50.2x/I50.4x/428.2x/428.4x on or before the ECG date) must be ≥0.70. If not met, recheck the implementation (lead order, scaling, tabular inputs) once; if still not met, stop. Compute and log D_G2 and Δ (§4) here, before any slope is estimated. Rationale: G2 is meant to catch implementation errors and gross transport failure, which push AUROC towards 0.5. The reference value is 0.835 against echo-measured LVEF ≤45% in MIMIC-IV [10], and coded labels will likely give a lower value, so the threshold is set at the conventional "acceptable discrimination" level rather than near 0.835.
- **WP3 — Analysis:** §6.4 → §6.5 → S1–S4 → §4 readout.

## 8. Limitations accepted up front

- ICD-coded anchors cover BIDMC admissions only, so diagnoses first made as outpatients are invisible. An HFrEF code is not the same as echo LVEF ≤45%.
- Pre-event ECGs are clinically indicated, so patients with ECGs are not a random sample.
- LVSD group composition: the prior-code and multi-code exclusions (§6.2) leave an LVSD group without known AF, although HF and AF often co-occur. Results apply to LVSD without prior or concurrent AF.
- AF as comparator: subclinical LV dysfunction can precede AF, and atrial remodelling may itself move an LVSD score. A rule-2 reading (L−A `≡`) therefore cannot by itself separate "not LV-specific" from "undiagnosed LVSD in the AF group".
- Control group: limb fractures replaced hip fracture at G1. More than half of the fracture patients (6,640 of 11,601) had no ECG at the anchor admission and were removed by the coverage rule, so the control group leans towards patients who get ECGs.
- The study uses one model (EchoNext-Mini, not the full EchoNext model [5]) and one target.

## 9. Out of scope (do not add without a decision)

- Surveillance-bias / IPW analysis of who gets a follow-up echo (separate paper).
- Training or fine-tuning any model; other targets (AS, HCM, amyloid); HEEDB replication.
- Individual-level detectability (MDC of the LVSD score).
- Spline or change-point models of the pre-event trajectory (the event study covers timing).

## 10. Open decisions

- [x] Public weights: only EchoNext-Mini (`weights.pt`, 4.3 MB, on GitHub, not an LFS pointer). It exposes `lvef_lte_45` and requires 7 tabular inputs (§6.1). Checked 30 Oct 2026 on PhysioNet [6], the IntroECG README and the repo contents.
- [x] Equivalence margin: 10% of the full-transition slope (§4). Decided 30 Oct 2026.
- [x] H1 tested as an intersection–union test (§4). Decided 30 Oct 2026.
- [x] Non-cardiac control fallback if HIPFX fails G1: all limb-fracture admissions (ICD-10 S42, S52, S62, S72, S82, S92; ICD-9 810–829), same rules as HIPFX. Decided 30 Oct 2026.
- [x] Echo-confirmed LVSD anchor: sensitivity analysis S4 (§6.6), not the primary anchor. Decided 30 Oct 2026. Local check (`checks/00_data_checks.json`, 30 Oct 2026): MIMIC-IV-ECHO 1.0 `structured-measurement` has a numeric TTE LVEF in 133,861 studies from 70,432 patients; 3,348 patients have a TTE LVEF >45% followed by one ≤45%, before any ECG rule. Values need cleaning (maximum 6570).
- [ ] Event-study bins for the −5 y window (§6.5 was written for −3 y). Proposed: −60…−36 [ref], −36…−24, −24…−12, −12…−6, −6…−1 months, so the reference stays the period furthest from the event. [decide before WP3]
- [ ] Δ denominator: keep Δ = 0.10 × D_G2 / 3 (a full transition over 3 years; the margin describes a negligible slope difference and does not depend on the window), or change it to /5 to match the window, which makes `≡` harder to reach. Proposed: keep /3 and reword §4 as "over 3 years". [decide before WP3]
- [ ] Licence: the IntroECG repo has no licence file (GitHub API: `license: null`); the README only asks to cite [9]. Confirm that research use and publication are acceptable, or ask the authors.
- [x] MIMIC-IV-ECG and `hosp`/ED share the per-patient date shift: 58.5% of ECGs fall inside a same-patient admission or ED stay, against 0.3% after shifting ECG times by +5 years (`checks/00_data_checks.json`).
- [x] Coverage: by approximate real year, ECGs end around 2019 while admissions run to about 2022; anchors after the ECG period would keep only early-window ECGs. Rule adopted: ≥1 ECG at the anchor admission (§6.2), rather than a calendar cut-off, which is only accurate to about ±1 year. Decided 30 Oct 2026.

## 11. Change log

v0.5 (30 Oct 2026) — WP1 done
- §7: G1 result recorded; per the pre-specified rule, the window is −5 y and the control is LIMBFX.
- §4, §6.2: LIMBFX row and notation note; initial-encounter rule for fracture anchors.
- §6.3: rhythm, quality and waveform rules written out as implemented.
- §6.6: S4 LVEF taken from both source systems; `lvef_upper` rule fixed.
- §8: control-group selection by the coverage rule.
- §10: event-study bins and Δ denominator for the −5 y window added.

v0.4 (30 Oct 2026)
- §6.2: coverage rule (≥1 ECG at the anchor admission).
- §5, §6.6, §7: echo-anchored LVSD added as sensitivity analysis S4, with its definition and limits.
- §10: coverage and echo-anchor items closed.

v0.3 (30 Oct 2026)
- §6.1: score is EchoNext-Mini [9] (the full EchoNext model has no public weights); exact weight files, output column, waveform and tabular inputs, MIMIC derivation of the tabular features [10]; no recalibration, with the reason.
- §3: cross-sectional MIMIC-IV validation [10] added; the within-patient gap remains.
- §6.6, §7, §8: label horizon also cited from [6]; worse inpatient/ICU performance [10] added to the S1 rationale; G2 refers to the output column and to the published AUROC, its threshold is lowered from 0.75 to 0.70, and one implementation recheck is allowed before stopping; limitation names the model.
- §7, §10: weights, margin, H1 test and the limb-fracture fallback closed; echo source and licence items added.
- §5, §10: dataset versions added; local data checks (`checks/00_data_checks.py`) close the date-shift item, size the echo anchor and add the coverage rule.
- References [9], [10] added.

v0.2 (30 Oct 2026)
- §1, §4: "≈" replaced by an equivalence test against a pre-specified margin Δ; new inconclusive class; readout made exhaustive (rules 1–5).
- §4: H1 written as an intersection–union test; β_LVSD > 0 added as a necessary condition.
- §6.4, §6.5: last-bin LVSD − AF contrast pre-specified as key secondary, because the linear slope dilutes late rises.
- §6.6: S3 (equal patient weights) and AF shared-substrate row added.
- §7: G1 now covers HIPFX, with a control-switch fallback; G2 sample specified; Δ fixed at G2.
- §8: LVSD-group composition and AF-comparator limitations added.
- §9: spline/change-point models listed as out of scope.
- §10: margin, H1 test and MIMIC coverage items added.

## Key references (peer-reviewed / dataset)

1. Attia ZI, et al. Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram. *Nat Med.* 2019. doi:10.1038/s41591-018-0240-2
2. König S, et al. Artificial intelligence-based identification of left ventricular systolic dysfunction from 12-lead electrocardiograms: external validation and advanced application of an existing model. *Eur Heart J Digit Health.* 2023. doi:10.1093/ehjdh/ztad081
3. Bjerkén L, et al. Artificial intelligence enabled ECG screening for left ventricular systolic dysfunction: a systematic review. *Heart Fail Rev.* 2022. doi:10.1007/s10741-022-10283-1
4. Croon P, et al. Phenotypic selectivity of artificial intelligence-enhanced electrocardiography in cardiovascular diagnosis and risk prediction. *Circulation.* 2025. doi:10.1161/CIRCULATIONAHA.125.076279
5. Poterucha TJ, et al. Detecting structural heart disease from electrocardiograms using AI. *Nature.* 2025. doi:10.1038/s41586-025-09227-0
6. Elias P, Finer J. EchoNext: a dataset for detecting echocardiogram-confirmed structural heart disease from ECGs (v1.1.1). *PhysioNet.* 2026. doi:10.13026/7cfw-d091
7. Yoo A, et al. Serial AI-enabled electrocardiogram trajectories during heart failure treatment. *JACC Case Rep.* 2026. doi:10.1016/j.jaccas.2026.109024
8. EchoNext: an ECG-based deep learning model to detect structural heart disease (Abstract 14647). *Circulation.* 2023;148(Suppl 1). doi:10.1161/circ.148.suppl_1.14647
9. Hughes JW, Jing L, Finer J, et al. EchoNext-Mini: a dataset and baseline AI model for detecting structural heart disease from electrocardiograms. *NEJM AI.* 2026;3(5). doi:10.1056/AIdbp2500516
10. Otabor E, Hassan A, Okunlola A, et al. Transportability of an artificial intelligence electrocardiography model for structural heart disease: external validation and recalibration of EchoNext-Mini in MIMIC-IV. *Eur Heart J Digit Health.* 2026;7(8):ztag147. doi:10.1093/ehjdh/ztag147
