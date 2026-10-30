# RISE-SPEC — Is AI-ECG "early detection" target-specific within the same patient?

Status: concept / pre-registration draft · v0.1 · 2026-10-30
Scope rule: minimum design that answers one question. Anything not listed here goes to §9 and needs an explicit decision before it is added.

## 1. Thesis

AI-ECG papers routinely read "false positives who later develop the target disease" as pre-clinical detection [1–3]. Croon et al. showed that disease-specific AI-ECG models behave as broad cardiovascular risk markers [4], but with one ECG per person: a between-person design cannot separate a stable high-risk *trait* from a developing *target disease*. A within-patient lead–lag design can, because each patient serves as their own control and stable between-person risk drops out. Either result is publishable: a target-specific rise supports the early-detection claim; a non-specific rise extends [4] with within-person evidence.

## 2. Research question

In adults with serial sinus-rhythm ECGs, does an AI-ECG LVSD score rise within the patient before incident LVSD more steeply than before (a) incident AF and (b) hip fracture?

## 3. Gap

- The false-positive → future-disease argument is repeated across models and reviews [1–3].
- Non-specificity has been shown between persons only [4]. No within-patient, event-anchored specificity test was found (Consensus literature check, 30 Oct 2026).

## 4. Pre-specified hypotheses and readout

- **H1 (target-specific):** β_LVSD > β_AF and β_LVSD > β_HIPFX.
- **H0 (non-specific):** pre-event slopes do not differ by event type; groups differ in level only.

| Pattern | Reading |
|---|---|
| β_LVSD > β_AF and β_LVSD > β_HIPFX | Target-specific pre-clinical signal |
| β_LVSD ≈ β_AF > β_HIPFX | Cardiac-generic, not LV-specific |
| β_LVSD ≈ β_AF ≈ β_HIPFX | Generic pre-event drift / acquisition context |

">" = 95% CI of the difference excludes 0; "≈" = CI includes 0 (not an equivalence claim).

## 5. Data

- MIMIC-IV-ECG (waveforms, machine measurements, machine report text) + MIMIC-IV `hosp` (`patients`, `admissions`, `diagnoses_icd`) + MIMIC-IV-ED (`edstays`, for acquisition context only).
- All processing is local (PhysioNet DUA). Only aggregate outputs leave the machine.

## 6. Design

### 6.1 Score

- EchoNext public weights [5,6], LVEF ≤45% output, frozen: no training, no fine-tuning. Developed on NewYork-Presbyterian/Columbia data, so there is no overlap with MIMIC.
- Input per the EchoNext spec: 10-s 12-lead at 250 Hz (downsample MIMIC 500 Hz), plus tabular covariates if the model requires them [verify]. Freeze the age input at the patient's first eligible ECG to remove mechanical age drift.
- Analysis scale: logit(p).

### 6.2 Anchors

First admission (`admittime`) carrying a code from one set (MIMIC stores codes without dots):

| Group | ICD-10 | ICD-9 |
|---|---|---|
| LVSD (target) | I50.2x, I50.4x | 428.2x, 428.4x |
| AF (cardiac comparator) | I48.x | 427.31, 427.32 |
| HIPFX (non-cardiac control) | S72.0x–S72.2x | 820.x |

- Incident means no earlier admission with any code from the three sets. For LVSD, also no earlier I50.x/428.x or I42.x/425.x.
- If the anchor admission carries codes from more than one set, exclude the patient.
- Exclude AF and HIPFX patients with an LVSD code ≤1 year after anchor.

### 6.3 ECG eligibility

- Window: −3 y to −30 d before anchor. The last 30 days are dropped because they capture the acute presentation.
- Sinus rhythm and not paced (from machine report text); no flat or missing lead.
- A patient enters if they have ≥2 eligible ECGs spanning ≥180 days and are aged ≥18 at anchor.

### 6.4 Primary analysis

Patient fixed-effects (within) regression, with t = years relative to anchor:

    logit(p_ij) = α_i + Σ_g β_g · t_ij · 1[g_i = g] + γ · t_ij · age_c,i + δ · t_ij · male_i + ε_ij

α_i absorbs every stable between-patient difference (level/trait). SEs are cluster-robust by patient.

**Primary estimand:** β_LVSD − β_AF (logit/year, 95% CI).
Secondary contrasts: β_LVSD − β_HIPFX and β_AF − β_HIPFX.

### 6.5 Secondary

- **Event study:** the same model with time bins (−36…−24 [ref], −24…−12, −12…−6, −6…−1 months) × group, to show *when* any rise starts.
- **Level:** mean logit(p) in the earliest bin per group. This is the between-person, Croon-type elevation.

### 6.6 Critical controls (one per alternative explanation)

| Alternative explanation | Control |
|---|---|
| EchoNext labels count an ECG as positive up to 1 y before the echo [5,8], so some pre-event rise is built in | Contrast against comparator events, never against zero |
| Sicker acquisition context: ED/inpatient ECGs cluster before events | **S1:** only ECGs outside any ED stay or admission |
| New conduction disease (LBBB/IVCD) raises LVSD scores | **S2:** only ECGs with QRS <120 ms |
| Rhythm or pacing transitions move scores [7] | Excluded at eligibility (sinus, non-paced only) |
| The model's age input rises with time | Age input frozen (§6.1) |

## 7. Work packages and gates

- **WP1 — Cohort & anchors** (DuckDB over MIMIC tables).
  **G1:** ≥500 patients each in LVSD and AF. If not met, widen the window to −5 y once. If still not met, stop.
- **WP2 — Scoring** all eligible ECGs. Verify the output head and input spec first.
  **G2 (transport check):** using one random ECG per patient, the AUROC of the LVEF output for a prevalent LVSD code (I50.2x/I50.4x/428.2x/428.4x on or before the ECG date) must be ≥0.75. If not met, stop.
- **WP3 — Analysis:** §6.4 → §6.5 → S1, S2.

## 8. Limitations accepted up front

- ICD-coded anchors cover BIDMC admissions only, so diagnoses first made as outpatients are invisible. An HFrEF code is not the same as echo LVEF ≤45%.
- Pre-event ECGs are clinically indicated, so patients with ECGs are not a random sample.
- The study uses one model and one target.

## 9. Out of scope (do not add without a decision)

- Surveillance-bias / IPW analysis of who gets a follow-up echo (separate paper).
- Training or fine-tuning any model; other targets (AS, HCM, amyloid); HEEDB replication.
- Individual-level detectability (MDC of the LVSD score).

## 10. Open decisions

- [ ] Which EchoNext weights are public (full model vs Mini-Model), and does the public model expose an LVEF ≤45% output? [verify]
- [ ] Non-cardiac control: hip fracture (default) or another non-cardiac admission.
- [ ] Echo-confirmed LVSD anchor as a sensitivity analysis, if a reliable echo-derived LVEF source exists in the available MIMIC modules. [verify]

## Key references (peer-reviewed / dataset)

1. Attia ZI, et al. Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram. *Nat Med.* 2019. doi:10.1038/s41591-018-0240-2
2. König S, et al. Artificial intelligence-based identification of left ventricular systolic dysfunction from 12-lead electrocardiograms: external validation and advanced application of an existing model. *Eur Heart J Digit Health.* 2023. doi:10.1093/ehjdh/ztad081
3. Bjerkén L, et al. Artificial intelligence enabled ECG screening for left ventricular systolic dysfunction: a systematic review. *Heart Fail Rev.* 2022. doi:10.1007/s10741-022-10283-1
4. Croon P, et al. Phenotypic selectivity of artificial intelligence-enhanced electrocardiography in cardiovascular diagnosis and risk prediction. *Circulation.* 2025. doi:10.1161/CIRCULATIONAHA.125.076279
5. Poterucha TJ, et al. Detecting structural heart disease from electrocardiograms using AI. *Nature.* 2025. doi:10.1038/s41586-025-09227-0
6. Elias P, Finer J. EchoNext: a dataset for detecting echocardiogram-confirmed structural heart disease from ECGs (v1.1.1). *PhysioNet.* 2026. doi:10.13026/7cfw-d091
7. Yoo A, et al. Serial AI-enabled electrocardiogram trajectories during heart failure treatment. *JACC Case Rep.* 2026. doi:10.1016/j.jaccas.2026.109024
8. EchoNext: an ECG-based deep learning model to detect structural heart disease (Abstract 14647). *Circulation.* 2023;148(Suppl 1). doi:10.1161/circ.148.suppl_1.14647
