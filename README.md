# VoterPulse

**Predicting Lok Sabha constituency voter turnout with machine learning.**

A complete undergraduate ML assignment: problem framing, preprocessing, EDA, four algorithms, honest evaluation, hyperparameter tuning, a Streamlit UI, and this README.

| | |
| --- | --- |
| Task | Supervised **regression** |
| Rows | 534 parliamentary constituencies · 33 states / UTs |
| Target | `target_turnout_pct` |
| Best model | Tuned Gradient Boosting |
| Test R² | **0.9262** |
| Test MAE | **2.23 percentage points** |
| Naive baseline | R² 0.8704 · MAE 2.97 pp |

The live lab (charts, viva cards, in-browser predictor) accompanies this repo.

---

## 1. Problem

Given a constituency’s electoral roll and its **previous-election turnout**, predict the **next** turnout percentage.

Why it matters:

- Booth staffing and ballot paper estimates
- Campaign effort in low-participation urban seats
- Early warning when a seat is likely to drop several points

Success rule, written before training: **beat the naive baseline** (reuse last turnout) on a held-out 20% of seats, with MAE under 3 pp.

## 2. Dataset

File: [`data/voter_turnout.csv`](data/voter_turnout.csv)

| Column | Role |
| --- | --- |
| `state_name`, `pc_no`, `pc_name` | Identity (SC/ST tag in the name marks reserved seats) |
| `total_electors`, `male_electors`, `female_electors`, `third_gender_electors` | Roll composition |
| `previous_turnout_pct` | Strongest predictor (corr 0.93 with the target) |
| `target_turnout_pct` | Label |

Quality checks:

- 0 missing values, 0 duplicate seats
- Male + female + third-gender = total electors on every row
- 9 of 543 Lok Sabha seats are absent (J&K, Ladakh, DNH&DD, Gujarat PC-24 Surat). They are documented, not imputed.

## 3. Features

**Baseline:** previous turnout, total electors, female share, third-gender share.

**Improved (used by the tree models):**

- `log_electors` — mega-seats (Malkajgiri 3.7M) do not dominate
- `gender_gap` — (male − female) / total
- `state_mean_prev` — training-fold mean previous turnout of that state (no test leakage)
- `is_reserved`, `is_urban` — parsed from the constituency name

Train / test split: **427 / 107**, `random_state=42`.

## 4. Models

All scores are on the **same 107 test seats**.

| Model | MAE | RMSE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Naive (previous = target) | 2.967 | 3.876 | 0.8704 | 4.719 |
| Linear Regression | 2.788 | 3.463 | 0.8965 | 4.302 |
| Ridge Regression | — | — | — | — |
| Decision Tree (depth 6) | — | — | — | — |
| Random Forest | 2.350 | 3.132 | 0.9153 | 3.735 |
| Gradient Boosting | 2.282 | 2.963 | 0.9242 | 3.683 |
| **Tuned Gradient Boosting** | **2.230** | **2.925** | **0.9262** | **3.571** |

Ridge / tree exact rows are in [`reports/metrics.json`](reports/metrics.json) after you run `python src/train.py`.

**Winner hyperparameters** (4-fold grid search, 81 combinations):

```
learning_rate=0.12  max_depth=3  min_samples_leaf=4  n_estimators=80
```

## 5. How to run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/train.py                # retrains and refreshes figures
python src/predict.py --state "Telangana" --name "Hyderabad"
streamlit run app.py
```

## 6. Project layout

```
data/voter_turnout.csv     dataset
src/features.py            shared feature engineering
src/train.py               full pipeline
src/predict.py             CLI predictor
app.py                     Streamlit UI
models/                    saved sklearn models
reports/figures/           EDA + evaluation plots
reports/metrics.json       scoreboard
REPORT.md                  write-up
VIVA_NOTES.md              likely questions
```

## 7. Deploy (Streamlit Cloud)

1. Push this folder to a public GitHub repo (README at the root).
2. Open [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Repository → branch `main` → file `app.py`.
4. Paste the `https://….streamlit.app` URL into your report.

## 8. What to say in viva

See [VIVA_NOTES.md](VIVA_NOTES.md). Short version:

- Regression, not classification.
- Previous turnout is the main signal; other features predict the *change*.
- Split first, encode state means on train only.
- MAE 2.23 pp is the number to remember.
- We do not use a neural net on 427 rows.

## License

Assignment use. Dataset is an extract of public electoral-roll / turnout figures.
