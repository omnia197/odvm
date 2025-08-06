> ⚠️ UNDER CONSTRUCTION

<h1 align="center">ODVM: Open, Dynamic, and Versatile Modeling</h1>

<p align="center">
  <img src="https://img.shields.io/badge/AutoML-Fast%20%26%20Flexible-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Built%20with-Python%203.8+-yellow?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Alpha-orange?style=flat-square" />
</p>

---

<p align="center">
  <b>From raw data to insights and models — clean, visual, and automated.</b><br>
  <i>Because clarity in data is power in decision.</i>
</p>

---

## What is ODVM?

**ODVM** is a modular and extendable AutoML pipeline designed to take your raw dataset from cleaning to model training and evaluation — with minimal configuration and maximum flexibility.

### Key Features

- Automated EDA (analysis + visualization)
-  Smart Preprocessing (missing values, encoding, scaling, outlier handling)
-  Model Selection (Scikit-Learn + XGBoost + LightGBM + CatBoost)
-  Hyperparameter Tuning (Grid / Random)
-  Reporting (HTML, Excel)
-  Task Detection (classification, regression, clustering, etc.)
-  Multi-backend support (Pandas / Dask)

All powered by Python & configurable via a simple `config.json`.

---

## Quick Start

```python
from core.runner import ODVM

odvm = ODVM(
    data="data/housing.csv",
    target="median_house_value",
    config="config.json"
)

odvm.run(eda=True, preprocess=True, model=True)

```
## Installation

```
git clone https://github.com/omnia197/odvm.git
cd odvm
pip install -r requirements_dev.txt

```

All behavior is driven through a simple, clean JSON config file:

```
{
  "eda": {
    "visualize": true,
    "save_summary": true,
    "save_dir": "outputs/eda_figures"
  },
  "preprocessing": {
    "missing_strategy": "mean",
    "outliers": { "strategy": "cap" },
    "encoding": "label",
    "scaling": "standard"
  },
  "split": {
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42
  },
  "modeling": {
    "allowed_models": ["LinearRegression", "DecisionTreeRegressor"],
    "tuning": true,
    "tuning_strategy": "grid",
    "param_grids": {
      "DecisionTreeRegressor": {
        "max_depth": [3, 5, 10]
      }
    },
    "cv": 3,
    "scoring": "r2"
  },
  "data": {
    "file_path": "data1.csv"
  }
}
```


## Coming Soon

We’re actively expanding ODVM — here’s what’s coming:

- PDF Reporting: Full modeling and EDA reports including charts and performance metrics

- Explainability (XAI): Built-in support for SHAP and LIME to interpret model decisions

- Auto Feature Engineering: Automatic generation of polynomial, interaction, and time-based features

- Auto Task Detection: Automatically detect whether the problem is classification, regression, clustering, etc.

- Streamlit Dashboard: Interactive dashboard to explore your dataset, models, and predictions

- FastAPI Deployment: Instantly serve trained models as RESTful API endpoints

- Plugin System: Add custom models, scalers, encoders, or even full pipeline blocks easily

- CLI Interface: Run ODVM workflows directly from the terminal using simple commands


---

## Contributing

We welcome all contributions, big or small!

To get started:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---

### Contact

[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:omnia18ayman@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/omnia-ayman-1b8340269/)
