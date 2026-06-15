# 🏥 Individual Health Insurance Costs Prediction

A data-driven regression solution designed to **predict individual medical insurance charges** based on a patient's demographic and lifestyle attributes. Leveraging advanced ensemble learning techniques, this project helps estimate healthcare premiums and analyze the primary risk factors that drive medical inflation costs.

---

## 🎯 Key Features
- **Exploratory Data Analysis (EDA):** Comprehensive statistical distributions modeling the correlation between personal habits (e.g., smoking) and insurance premium costs.
- **Categorical Feature Encoding:** Seamless mapping of qualitative categorical attributes (`sex`, `smoker`, `region`) into numerical formats optimized for regression models.
- **Advanced Ensemble Regressor:** Implements a tuned `GradientBoostingRegressor` to capture complex, non-linear relationships across user profiles.
- **High Predictive Power:** Reaches a solid regression fit with an $R^2$ Score of **87.45%** on the evaluation testing partition.

---

## 📊 Dataset Structure
The system processes a structured insurance profile dataset (`insurance.csv`) containing individual medical records with the following core attributes:
- `age`: Age of primary beneficiary (continuous).
- `sex`: Insurance contractor gender (`female`, `male`).
- `bmi`: Body Mass Index, providing an objective index of body weight relative to height ($kg/m^2$).
- `children`: Number of dependents / children covered by health insurance.
- `smoker`: Smoking status of the individual (`yes`, `no`).
- `region`: The beneficiary's residential area in the US (`southwest`, `southeast`, `northwest`, `northeast`).
- `charges`: **Target Variable** representing individual medical costs billed by health insurance.

---

## 🛠️ Data Pipeline & Preprocessing

### 1. Feature Encoding & Transformation
- String columns describing personal habits and geographical locations are structured using robust data-science encoders.
- Multi-collinearity checks and missing-value validations are performed during initial runtime to ensure structural data integrity.

### 2. Validation Framework
The dataset records are strictly divided into dedicated training and testing matrices via `train_test_split` with a fixed seed (`random_state=42`) to maintain reproducible, unbiased benchmarking across experiments.

---

## 🤖 Modeling & Performance Evaluation

The core regression engine runs a powerful **Gradient Boosting Regressor** (`n_estimators=200`) designed to minimize continuous error residuals. 

### 📈 Final Performance Metrics
The production model delivered excellent scoring metrics when evaluated against unseen profiles:

| Evaluation Metric | Test Partition Score |
| :--- | :--- |
| **$R^2$ Score (Coefficient of Determination)** | **0.8745 (87.45%)** |
| **MSE (Mean Squared Error)** | *Minimized to baseline standard deviations* |

*The statistical results highlight that demographic factors combined with critical lifestyle attributes (such as smoking) hold the highest predictive weight in evaluating health risk pricing.*

---

## 🚀 Installation & Usage

### Prerequisites
Make sure you have Python installed on your system. Install the baseline data-science dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
