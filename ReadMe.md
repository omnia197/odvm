> UNDER CONSTRUCTION


<h1 align="center"> ODVM: Open, Dynamic, and Versatile Modeling
 </h1>

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

ODVM is a **modular AutoML system** that transforms raw data into actionable models through:

- Smart preprocessing (cleaning, scaling, encoding)
- Model selection & tuning (Sklearn + XGB + LGBM + CatBoost)
- Performance evaluation & explainability
- Upcoming: Dashboards, API deployment, full PDF reports

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

## Coming Soon

ODVM is just getting started — here’s what’s next:

- *Auto EDA module*  

  Automatically detects feature types, distributions, correlations, and visualizes them with minimal config

- *PDF & HTML reporting*  

  Export full model evaluation reports, hyperparameter summaries, and EDA visuals in one click

- *Explainability (XAI)*  

  SHAP-based explainers for tree models & LIME for black-box interpretation.

- *Auto Feature Engineering*  

  Generate polynomial, time-based, or domain-specific features automatically.

- *CLI support*

  Run ODVM via terminal with simple arguments:  
  `odvm run --data my.csv --target y --config config.json`

- *Streamlit Dashboard* 

  Explore your data and models interactively through a beautiful, live dashboard.

- *FastAPI Deployment* 
  One-line deployment of trained model as an API endpoint.

- *Plugin System*

  Custom encoders, scalers, models, or even full blocks you can drop into ODVM.


---

## Contributing

We welcome all contributions, big or small!

To get started:

1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a Pull Request

---
