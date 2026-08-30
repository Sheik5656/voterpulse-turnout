# VoterPulse — Project report

## 1. Problem identification (10)

**Statement.** Predict `target_turnout_pct` for an Indian parliamentary constituency from its previous turnout and electoral-roll composition.

**Type.** Supervised regression.

**Motivation.** Turnout drives polling-booth logistics and campaign intensity. Urban seats such as Hyderabad (~49%) and rural high-participation seats such as Dhubri (~92%) cannot be staffed the same way.

**Success criterion.** On a held-out 20% of constituencies, beat the naive baseline “predicted = previous turnout”, and keep MAE under 3 percentage points.

## 2. Dataset and preprocessing (15)

534 rows, 9 columns, 33 states/UTs. Zero nulls, zero duplicate `(state, pc_no)` pairs. Gender headcounts sum to `total_electors` on every row.

Nine of 543 Lok Sabha seats are missing: Jammu & Kashmir (5), Ladakh (1), Dadra & Nagar Haveli and Daman & Diu (2), Gujarat PC-24 Surat (1). They are listed, not filled in.

Derived columns (rates, log electors, reserved/urban flags) are computed after the quality checks and **before** the split. State-mean encoding is fit on the training fold only.

Split: 427 train / 107 test, `random_state=42`.

## 3. EDA (10)

- Mean target turnout 66.9%, mean previous 68.4% (national drift −1.5 pp).
- Correlation(previous, target) = **0.93**. The modelling problem is the residual.
- Urban/metro-tagged seats average **58.0%**; the rest **67.7%**.
- Reserved (SC/ST) seats average **68.8%** vs **66.3%** general.
- Lowest seats are large urban constituencies (Hyderabad, Bangalore, Mumbai, Delhi, Patna Sahib). Highest seats cluster in West Bengal, coastal Andhra, and parts of the North-East.
- Kerala, Haryana, Rajasthan trend down vs previous; Andhra Pradesh, Telangana, Karnataka trend up.

Figures: `reports/figures/01_turnout_distribution.png` through `08_feature_importance.png`.

## 4. Algorithms (20)

| Algorithm | Why it is here |
| --- | --- |
| Naive previous | Honesty. If ML cannot beat this, stop. |
| Linear Regression | Explainable baseline; coefficients after scaling. |
| Ridge | Same story on the engineered feature set, L2 penalty. |
| Decision Tree | Non-linear rules a viva panel can draw. |
| Random Forest | Bagging to cut tree variance. |
| Gradient Boosting | Sequential residual correction — usually strongest on small tabular data. |

All models share the same split. Trees use the improved 8-feature matrix; linear regression uses the 4-feature baseline so the “improvement” page has a clean before/after.

## 5. Evaluation (10)

Held-out test (107 seats):

| Model | MAE | RMSE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Naive | 2.967 | 3.876 | 0.8704 | 4.719 |
| Linear Regression | 2.788 | 3.463 | 0.8965 | 4.302 |
| Random Forest | 2.350 | 3.132 | 0.9153 | 3.735 |
| Gradient Boosting | 2.282 | 2.963 | 0.9242 | 3.683 |
| **Tuned GB** | **2.230** | **2.925** | **0.9262** | **3.571** |

MAE is the headline: the tuned model is wrong by **2.23 points** on average. RMSE is only a little higher, so there are not many catastrophic misses. Residuals centre on zero. A short list of 6–8 point errors is reported rather than hidden — local politics is not in the file.

## 6. Improvement (10)

1. Replace raw electors with `log_electors`.
2. Add `gender_gap`, `state_mean_prev` (train-only), `is_reserved`, `is_urban`.
3. Grid-search Gradient Boosting: 3×3×3×3 = 81 cells, 4-fold CV, score = R².
4. Best params: `n_estimators=80`, `max_depth=3`, `learning_rate=0.12`, `min_samples_leaf=4`.

Lift vs naive: **+0.056 R²**, **−0.74 pp MAE**. Lift vs untuned linear: **+0.030 R²**, **−0.56 pp MAE**.

We did not add a neural network. 427 rows is the wrong regime, and feature importance would disappear.

## 7. Application (10)

- `app.py` — Streamlit: problem, EDA, scoreboard, live predictor with sliders.
- `src/predict.py` — CLI for a single seat.
- Companion web lab with the same model running in the browser (exported trees).

## 8. Limitations

- 9 seats missing; no J&K signal.
- Urban keyword list is a proxy, not a census classification.
- Previous turnout dominates; the model will not foresee a one-off local shock.
- State encoding cannot help a state that fell entirely into the test fold (fallback = national train mean).

## 9. Conclusion

Constituency turnout is highly persistent. A carefully split Gradient Boosting model, with a handful of EDA-driven features and a small grid search, predicts the next number to about two percentage points. That is enough to be useful for planning, and simple enough to defend in a viva.
