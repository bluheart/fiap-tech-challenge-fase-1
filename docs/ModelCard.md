# Model Card for FlexibleMLP-Churn-Predictor

## Model Details
- **Developer:** Caio Ribeiro
- **Version:** 1.0
- **Model Type:** Neural Network (Multi-Layer Perceptron)
- **License:** MIT

## Intended Use
- **Primary Use:** Binary Classification - Customer Churn Prediction

## Training Data
- **Name:** Telco Customer Churn Dataset
- **Total Samples:** 7,043 customers
- **Features:** 21 original features (including target)
- **Churn Rate:** ~26.5% (1,869 churners / 5,174 non-churners)
- **Class Distribution:** Imbalanced

## Quantitative Analysis
- Comparison with baseline models

| tags.mlflow.runName                        | metrics.recall_score | metrics.precision_score | metrics.accuracy_score | metrics.f1_score | metrics.roc_auc | metrics.log_loss |
|--------------------------------------------|----------------------|-------------------------|------------------------|------------------|-----------------|------------------|
| mlp model run threshold 0.20               | 0.863636             | 0.483533                | 0.718950               | 0.619962         | 0.855917        | 0.406376         |
| mlp model run threshold 0.25               | 0.775401             | 0.529197                | 0.757275               | 0.629067         | 0.845984        | 0.419525         |
| random forest 1k est                       | 0.496000             | 0.671480                | 0.801278               | 0.570552         | 0.838959        | 0.446341         |
| random forest max_depth=4                  | 0.330667             | 0.692737                | 0.782825               | 0.447653         | 0.848192        | 0.433798         |
| random forest                              | 0.493333             | 0.672727                | 0.801278               | 0.569231         | 0.838690        | 0.446499         |
| logistic balanced                          | 0.818667             | 0.522998                | 0.753016               | 0.638254         | 0.857191        | 0.476833         |
| dummy classifier no scaling                | 0.000000             | 0.000000                | 0.733854               | 0.000000         | 0.500000        | 0.579390         |
| logistic balanced no scaling               | 0.797333             | 0.528269                | 0.756565               | 0.635494         | 0.856389        | 0.469435         |
| logistic unbalanced no scaling             | 0.525333             | 0.661074                | 0.801987               | 0.585438         | 0.853764        | 0.407887         |

## Ethical Considerations
- No signicant biases were found during tests

## Limitations & Recommendations
- Low precision, can be raised with higher thresholds at the cost of recall score