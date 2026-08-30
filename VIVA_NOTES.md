# Viva notes — VoterPulse

Memorise the **bold** numbers. Speak in short sentences.

## One-minute pitch

This is a regression problem. I predict next Lok Sabha turnout for a constituency from last turnout and the electoral roll. 534 seats, 80/20 split, seed 42. The naive baseline already has R² **0.87**. Tuned Gradient Boosting reaches R² **0.926** and MAE **2.23 points**. I beat the baseline without leaking the test set.

## Likely questions

**What is the target?**  
`target_turnout_pct`. A percentage, so regression, not classification.

**Why is previous turnout so strong?**  
People who voted last time mostly vote again. Correlation is **0.93**. The model’s real job is the leftover change.

**How did you split?**  
`train_test_split(test_size=0.2, random_state=42)` → 427 / 107. Split happens before fitting or encoding.

**What is leakage? Did you leak?**  
Leakage is using test information while training. State means are computed on train only. Grid search uses CV on train only. Test is touched once, at the end.

**Explain MAE vs RMSE vs R².**  
MAE = average absolute miss, in points (**2.23**). RMSE = same units but squares big errors (**2.93**). R² = fraction of variance explained vs predicting the mean (**0.926**).

**Why Gradient Boosting?**  
Small tabular data. Each tree fits the previous residual. Shallow trees (depth 3) plus a learning rate of 0.12. I can export the trees and run them in a browser.

**Why not a neural network / SVM / XGBoost?**  
NN: too few rows, no importance, harder viva. SVM: extra kernel story, similar accuracy. XGBoost: fair, but sklearn Gradient Boosting is enough and stays in one library.

**What features did you add and why?**  
`log_electors` (size), `gender_gap`, `state_mean_prev` (Kerala vs Bihar level), `is_reserved`, `is_urban` (urban seats average 58% vs 68%).

**How did you tune?**  
GridSearchCV, 81 combinations, 4-fold CV, scoring R². Winner: 80 trees, depth 3, learning rate 0.12, min_samples_leaf 4.

**Where does it fail?**  
A few test seats miss by 6–8 points. Local campaigns, weather, by-elections are not in the CSV. I show those seats instead of deleting them.

**Why 534 not 543?**  
J&K (5), Ladakh (1), DNH&DD (2), Surat (1). I do not impute geography I do not have.

**Is this classification if I bin turnout into high/medium/low?**  
I could, but I would throw away the 2-point precision the Election Commission actually needs. Regression is the right type.

**How do you know you did not overfit?**  
Test MAE and CV R² are close. A depth-3 boosted tree on 427 rows with min_samples_leaf 4 is a small model.

## Numbers cheat-sheet

| Thing | Value |
| --- | --- |
| Rows / states | 534 / 33 |
| Train / test | 427 / 107 |
| Corr(prev, target) | 0.93 |
| Mean target | 66.9% |
| Naive R² / MAE | 0.8704 / 2.97 |
| Linear R² / MAE | 0.8965 / 2.79 |
| Tuned R² / MAE | 0.9262 / 2.23 |
| Urban vs other | 58.0% vs 67.7% |
| Grid size | 81 |
