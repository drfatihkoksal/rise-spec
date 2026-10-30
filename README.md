# RISE-SPEC: within-patient AI-ECG trajectories before left ventricular systolic dysfunction

Analysis code, protocol versions and aggregate results for the study *Target-specific and accelerating:
within-patient artificial intelligence electrocardiogram trajectories before left ventricular systolic
dysfunction* (manuscript submitted).

**Question.** Within the same patients, does the score of an AI-ECG model for left ventricular systolic
dysfunction (LVSD) rise faster before incident LVSD than before atrial fibrillation (AF), before a limb
fracture, and over the same time in patients without any event? Two frozen public models (EchoNext-Mini,
DeepECG-SL) were applied to serial sinus-rhythm ECGs in MIMIC-IV and the Harvard-Emory ECG Database (HEEDB);
slopes before each event were compared with patient fixed-effects regression and pre-specified decision rules.

## Contents

| Path | What it holds |
|---|---|
| `concept.md`, `concept_v*.md` | Protocol (current version and earlier versions; change log in §11) |
| `checks/` | Data and implementation checks run before the analyses, with their aggregate outputs |
| `analysis/` | Cohorts, scoring, models, robustness analyses, figures, tables |
| `results/` | Aggregate results (JSON/CSV) written by the analysis scripts |
| `figures/` | Figures 1–2 with the data behind them |
| `registry.json` | Pointers from every number in the manuscript to the result file and cell it comes from |

No patient-level data are included. Patient-level intermediates are written to `work/`, which is not
part of the repository.

## Data access

The data are available to credentialed users under the respective data use agreements; they cannot be
shared here.

- MIMIC-IV 3.1 (https://doi.org/10.13026/kpb9-mt58), MIMIC-IV-ECG 1.0 (https://doi.org/10.13026/4nqg-sb35),
  MIMIC-IV-ED 2.2 (https://doi.org/10.13026/5ntk-km72) and MIMIC-IV-ECHO 1.0 (https://doi.org/10.13026/nrjh-5r77)
  from PhysioNet.
- Harvard-Emory ECG Database 5.0 (https://doi.org/10.60508/765d-f248) from the Brain Data Science Platform.
- EchoNext 1.1.1 test set (https://doi.org/10.13026/7cfw-d091) from PhysioNet, for the implementation check.

Data locations are set as constants at the top of each script (`V`, `HOSP`, `ECG`, `ED`, `ECHO`, `HEEDB`,
`MIMIC_ECG`, `MIMIC_HOSP`); edit them to your local copies.

## Models

Both models are used without training or fine-tuning. Place them under `external/` (not part of the repository):

| Model | Source | Pinned version | Expected path |
|---|---|---|---|
| EchoNext-Mini | https://github.com/PierreElias/IntroECG | commit 15233e9 | `external/IntroECG/` |
| DeepECG-SL, fine-tuned for LVEF (MHI) | https://huggingface.co/heartwise/deepecg-sl_finetuned_LVEF_MHI | revision 1ee6698 | `external/deepecg/weights/deepecg-sl_finetuned_LVEF_MHI/` |
| DeepECG preprocessing (ported, not imported) | https://github.com/HeartWise-AI/DeepECG_Docker | commit c81d677 | `external/deepecg/DeepECG_Docker/` |

`checks/03_model_provenance.py` records the revisions and SHA-256 hashes of the weight files
(`checks/03_model_provenance.json`). Weights are loaded with `torch.load(..., weights_only=True)`.

## Environment

Python 3.13 with the packages in `requirements.txt`. Scoring uses a CUDA GPU if one is available.

## Reproducing the results

Run from the repository root, in this order. Each script states its protocol section in its docstring.

| Step | Command | Output | Used in |
|---|---|---|---|
| Data checks | `python3 -I checks/00_data_checks.py`, `checks/01_heedb_checks.py` | `checks/0[01]_*.json` | Supplementary Section S1 |
| MIMIC-IV cohort | `python3 -I analysis/01_cohort.py --window-years 5 --control LIMBFX` | `results/wp1_flow_5y_limbfx.json` | Figure 1, Table 1, Supplementary Table S1 |
| EchoNext-Mini check and scoring | `python3 -I analysis/02_score.py check`, then `score` | `results/wp2_check.json`, `results/wp2_g2.json` | Supplementary Table S3 |
| MIMIC-IV analysis | `python3 -I analysis/03_analysis.py --verify` | `results/wp3_results.json` | Table 2, Figure 2, Supplementary Tables S4, S7, S8 |
| HEEDB cohort, scoring, analysis | `analysis/11_heedb_cohort.py`, `12_heedb_score.py`, `13_heedb_analysis.py` | `results/wp_h_*.json` | Table 2, Figure 2, Supplementary Table S4 |
| Bridge check | `python3 -I analysis/05_bridge.py` | `results/wp_h_bridge.json` | Supplementary Section S2 |
| Interval agreement | `python3 -I checks/02_interval_agreement.py` | `checks/02_interval_agreement.json` | Supplementary Table S3 |
| DeepECG-SL load check, scoring, analysis | `analysis/14_deepecg_check.py`, `15_deepecg_score.py`, `16_deepecg_analysis.py` | `results/deepecg_load_check.json`, `results/wp_d_*.json` | Table 2, Figure 2 |
| Post hoc R1–R6 | `analysis/17_robustness.py` … `21_r6_random_date.py` | `results/robustness_r*.json` | Supplementary Tables S5, S6 |
| Figures | `analysis/22_figure_event_years.py`, `23_flow_diagram.py` | `figures/` | Figures 1–2 |
| Manuscript values and tables | `analysis/24_paper_numbers.py`, `25_manuscript_tables.py`, `26_supplement.py` | `results/manuscript_values.json`, tables, supplement | Text, Tables 1–2, Supplementary material |

The gate runs with the pre-specified settings (`01_cohort.py` with the default three-year window and hip
fracture, then `--window-years 5`) produce `results/wp1_flow_3y_hipfx.json` and `results/wp1_flow_5y_hipfx.json`.

The full pipeline was run on 30 October 2026 on one workstation (32 CPU cores, one NVIDIA RTX 5090 GPU).
The scoring steps take the longest; model fitting, figures and tables take minutes.

## Order of specification

Each protocol after the MIMIC-IV analysis (HEEDB replication, second model, post hoc analyses) was committed
before the corresponding scores or analyses were run; the commit history shows the order. The repository was
created after the MIMIC-IV analysis.

## Citation and licence

See `CITATION.cff`. Code is released under the MIT licence (`LICENSE`). The models and datasets keep their
own licences and data use agreements.
