# Techniques-Of-Artificial-Intelligence-
Techniques of AI final project. Predicts income (&lt;=/> $50k) on the Adult/'salary' dataset using KNN and SVM. Includes preprocessing, label encoding, train/valid/test split, feature scaling, k-fold CV and simple tuning. Best: KNN (k=21) ~0.82 acc; SVM ~0.81 with scaling; lower w/o scaling.


# Techniques of AI — Income Classification (KNN & SVM)

Predict whether annual income is <=50K or >50K on the Adult/"salary" dataset. Two models are implemented—K-Nearest Neighbors and Support Vector Machine—along with preprocessing, feature scaling, and validation.

## Dataset
- Source: Kaggle “salary” (Adult Income) dataset, 32,561 rows, 15 columns.
- Target: `salary` (<=50K vs >50K)
- Selected features: dropped very high-scale or low-informative columns; categorical features label-encoded.

## Methodology
- Split: 80% train, 10% validation, 10% test (via 80/20 then 50/50 split).
- Models: KNN (Euclidean), SVM (SVC).
- Scaling: `StandardScaler` applied to features prior to final training.
- Validation:
  - K-fold CV to choose K for KNN.
  - GridSearchCV attempted for SVM; due to hardware limits, simple CV over C values was run.
- Metrics: accuracy, confusion matrix, precision/recall/F1.

## Results 
| Model | Scaling | Key setting | Accuracy |
|------:|:------:|:-----------:|---------:|
| KNN   | Yes    | k=21        | ~0.819 |
| SVM   | Yes    | default     | ~0.810 |
| SVM   | No     | default     | ~0.763 |

> Observation: Feature scaling improved both models; KNN slightly outperformed SVM on this dataset.

## Reproduce
```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install deps
pip install -U pandas numpy scikit-learn matplotlib seaborn jupyter

# run the notebook or script
jupyter lab   # or: jupyter notebook
# open notebooks/techniques-of-ai-final.ipynb (or your .ipynb) and run all cells

