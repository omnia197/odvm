# Current Limitations in ODVM

ODVM is still in early development (v0.x). While it's already functional for many machine learning workflows, several components are still being enhanced.

Below is a list of current known limitations and future work:

| Area                      | Status     | Details                                                                 |
|---------------------------|------------|-------------------------------------------------------------------------|
| Time-Series Support       | Missing | No specific processing for temporal data (e.g., lags, trends, resample) |
| Data Type Inference       | Partial | Some mixed-type or ambiguous columns may need manual adjustments         |
| Auto Feature Engineering  | Missing | No automatic generation of new features (yet)                            |
| Dashboard Interactivity   | Prototype | Basic Streamlit UI exists, but lacks filtering and deeper insights      |
| Model Drift Monitoring    | Missing | Concept & data drift detection not yet implemented                       |
| Out-of-Core Dask Training | Missing | Dask support limited to loading and EDA; model training uses sklearn     |
| Testing Suite             | Missing | Unit, integration & regression tests are not yet added                   |
| CLI Tool                  | Planned | CLI execution planned but not released                                   |
| Deployment Export (API)   | In Progress | FastAPI module exists but not fully integrated                           |

---

## Planned Features

The following are already in development or planned:

-  Grid & Random Hyperparameter Tuning
- SHAP + LIME Explainability
- EDA Reporting to HTML & PDF
- Plugin System for custom models and transformers
- One-click deployment via FastAPI

---

## Contribute

If you'd like to help solve any of the above or have ideas to add —  
please check our [issues](https://github.com/omnia197/odvm/issues) or open a [new one](https://github.com/omnia197/odvm/issues/new)!

