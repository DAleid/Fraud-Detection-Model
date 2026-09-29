# Credit Card Fraud Detection

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DAleid/Fraud-Detection-Model/blob/main/fraud_detection.ipynb)

Machine learning models that detect fraudulent credit card transactions in a highly imbalanced dataset (about 0.17% fraud).

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 transactions, of which 492 are fraud. Features `V1`-`V28` are anonymised PCA components; `Time` and `Amount` are the only original features. The dataset is not included in this repository. The notebook downloads it automatically with `kagglehub`.

## Approach

1. **EDA:** class imbalance, missing values, duplicates, transaction amounts by class.
2. **Preprocessing:** stratified train/validation/test split (60/20/20); `Time` and `Amount` scaled with a scaler fit on the training set only, to avoid data leakage.
3. **Imbalance handling:** class weights.
4. **Models:** Logistic Regression, Random Forest, Gradient Boosting.
5. **Evaluation:** PR-AUC, ROC-AUC, precision, recall and F1. Accuracy is not used, because always predicting "legit" would score about 99.8%.
6. **Threshold tuning:** the decision threshold is chosen on the validation set (maximising F1) and applied unchanged to the test set.

## Results (held-out test set)

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7209 | 0.9726 | 0.8000 | 0.8163 | 0.8081 |
| **Random Forest** | **0.8438** | 0.9572 | **0.8817** | **0.8367** | **0.8586** |
| Gradient Boosting | 0.7403 | 0.9552 | 0.8081 | 0.8163 | 0.8122 |

Random Forest performed best in this experiment. Logistic Regression has the highest ROC-AUC but the lowest PR-AUC, which shows why ROC-AUC alone is misleading on imbalanced data.

## Limitations

- The test set contains only about 98 fraud cases and a single random split was used, so small differences between models may not be significant.
- Features are anonymised, so predictions cannot be explained in business terms.
- A time-based split would be more realistic than a random one.

## Run it

Click the Colab badge above, or locally:

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

Downloading the dataset with `kagglehub` may require a Kaggle account and API token.

## Possible next steps

- Compare class weighting with SMOTE.
- Add SHAP explanations.
- Use an LLM to turn flagged transactions into short alert summaries for analysts.
