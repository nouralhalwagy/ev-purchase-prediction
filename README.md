# Predicting Electric Vehicle Purchases

My solution to the Kaggle competition [Playground Series – Season 6, Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9). The task is to predict whether a person will buy an electric vehicle (`Will_Buy_EV`) from demographic, commuting and charging-infrastructure features. Submissions are scored by **ROC-AUC**.

📓 **Notebook:** [`notebooks/ev-purchase-prediction.ipynb`](notebooks/ev-purchase-prediction.ipynb) · [View on Kaggle](https://www.kaggle.com/code/nouralhalwagy/electricvehcilepurch-competition-t1)

## Results

| Submission | Private LB | Public LB |
|---|---|---|
| **XGBoost + LightGBM + CatBoost blend** (0.36 / 0.54 / 0.10) | **0.94086** | **0.94149** |
| XGBoost + LightGBM blend (0.40 / 0.60) | 0.94084 | 0.94148 |

## Approach

1. **Exploratory analysis.** Looked at feature distributions, percentiles and how each numeric feature correlates with the target.
2. **Feature engineering.** Added interaction features:
   - `income_X_subsidy`: annual income × subsidy availability
   - `Concern_X_subsidy`: environmental concern level × subsidy availability
   - Also tried a charging "worry score", total nearby chargers and commute per charging station. These were dropped after experiments.
3. **Preprocessing.** A scikit-learn `ColumnTransformer`:
   - numeric features passed through
   - one-hot encoding for nominal categoricals
   - ordinal encoding for `Range_Anxiety_Level` (Low < Medium < High)
4. **Models.** Each model was evaluated with 10-fold stratified cross-validation, using out-of-fold (OOF) predictions:
   - Logistic Regression (baseline)
   - XGBoost
   - LightGBM
   - CatBoost (native categorical handling)
5. **Ensembling.** Searched blend weights on the OOF predictions to maximise AUC. Then refit each model on the full training set and blended their test-set probabilities.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Matplotlib · Seaborn

## Running it

```bash
pip install -r requirements.txt
```

Download the competition data from the [Kaggle data page](https://www.kaggle.com/competitions/playground-series-s6e9/data). In the notebook, change the `/kaggle/input/...` paths to point to your local `train.csv` and `test.csv`. You can also run the notebook directly on Kaggle.

## Repository structure

```
├── notebooks/
│   └── ev-purchase-prediction.ipynb
├── requirements.txt
└── README.md
```
