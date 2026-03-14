# BioMedAI 👨🏼‍🔬👨🏼‍🔬
## Where Intelligence solves medical engineering problems🧑🏼‍⚕️

### Overview👩🏼‍💻

BiomedAI is a  Predictive Maintenance Intelligence Suite, an end-to-end machine learning system built to predict equipment failures, automate fault triage, and optimize spare-parts and technician assignments.
The framework integrates heterogeneous data sources — Scania (sensor-based) and MaintNet (vehicle, facility, aviation work orders) — into a unified analytics and decision-support pipeline.

By combining tabular predictive modeling, natural language processing, and knowledge-based recommendation, this project provides a robust, interpretable, and domain-agnostic maintenance intelligence platform.

### 📁 Repository Structure
``` php
📦 Biomed_AI
│
├── 📁 data/
│ ├── 📁 scania_dataset/
│ │ ├── data/
│ │ │ ├── train.csv
│ │ │ ├── test.csv
│ │ │ └── testlabels.csv
│ │ └── document/
│ │ └── reference.pdf
│ │
│ └── 📁 maintnet_dataset/
│ ├── vehicle/
│ │ ├── tickets.csv
│ │ ├── work_orders.csv
│ │ └── parts.csv
│ ├── facility/
│ │ ├── tickets.csv
│ │ └── work_orders.csv
│ └── aviation/
│ ├── tickets.csv
│ └── work_orders.csv
│
├── 📁 notebooks/
│ ├── 01_scaniadata_eda_preprocessing.ipynb
│ ├── 02_failure_risk_modeling.ipynb
│ ├── 03_ticket_triage_nlp.ipynb
│ ├── 04_parts_technician_recommendation.ipynb
│ └── 05_evaluation_and_reporting.ipynb
│
├── 📁 src/
│ ├── 📁 data_pipeline/
│ │ ├── preprocess_scaniadata.py
│ │ ├── preprocess_maintnet.py
│ │ └── feature_store_builder.py
│ │
│ ├── 📁 models/
│ │ ├── failure_risk_lightgbm.py
│ │ ├── distilbert_triage.py
│ │ └── apriori_recommendation.py
│ │
│ └── 📁 utils/
│ ├── shap_explainability.py
│ └── metrics.py
│
├── 📁 outputs/
│ ├── cleaned_data/
│ ├── model_checkpoints/
│ ├── shap_plots/
│ └── reports/
│
├── requirements.txt
├── LICENSE
└── README.md
``` 
### Project Objectives🎗
#### 1. Failure Risk Prediction (Tabular)

Predict probability of component or system failure using Scania sensor and operational data.

Provide interpretable SHAP-based feature attributions per device.

Enable maintenance teams to prioritize at-risk equipment.

#### 2. Ticket Triage & Fault Categorization (NLP)

Automate classification of maintenance tickets into urgency and fault category.

Use DistilBERT fine-tuning with optional structured feature fusion.

Support multi-task learning with shared encoders.

#### 3. Parts & Technician Recommendation

Generate rule-based recommendations of parts and technician groups for each fault category.

Use Apriori association mining for confidence-based recommendations.

Evaluate via top-k recall and precision@k metrics.

### Datasets
#### 1. Scania Dataset

Source: Industrial sensor readings, operational logs, and failure labels.

Structure:

data/*.csv → numeric sensor metrics.

testlabels.csv → failure labels.

Use: Predictive modeling (Failure Risk).
#### 2. MaintNet Dataset

Source: Multi-industry maintenance management system (Vehicle, Facility, Aviation).

Structure:

tickets.csv → text descriptions of faults.

work_orders.csv → metadata and repair actions.

parts.csv, technicians.csv → recommendation mapping.

Use: NLP triage, recommendation system, cross-domain learning.

### Modeling Framework
1️⃣ Failure Risk Prediction
| Stage              | Description                                                                      |
| ------------------ | -------------------------------------------------------------------------------- |
| **Baselines**      | Frequency/rule-based system + Logistic Regression.                               |
| **Main Models**    | LightGBM / CatBoost with class-imbalance handling (focal loss or class weights). |
| **Calibration**    | Platt and Isotonic scaling for probability reliability.                          |
| **Explainability** | SHAP summary plots and per-device force plots.                                   |

2️⃣ Ticket Triage & Fault Category (NLP)
| Stage                 | Description                                                                           |
| --------------------- | ------------------------------------------------------------------------------------- |
| **Baselines**         | TF-IDF + Linear SVM / Logistic Regression.                                            |
| **Main Model**        | DistilBERT fine-tuned on ticket text with optional structured features (late fusion). |
| **Multi-Task Setup**  | Shared encoder with dual output heads for Urgency & Fault Category.                   |
| **Data Augmentation** | Synonym replacement, paraphrasing, and typo noise injection.                          |

3️⃣ Parts & Technician Recommendation
| Stage               | Description                                                                        |
| ------------------- | ---------------------------------------------------------------------------------- |
| **Approach**        | Association rules on `(fault category, device type)` → `(part, technician group)`. |
| **Algorithm**       | Apriori rule mining.                                                               |
| **Ranking Metrics** | Support × Confidence, evaluated with top-k recall and precision@k.                 |


### End-to-End Architecture
           ┌─────────────────────────────┐
           │     Data Acquisition        │
           │ Scania + MaintNet Datasets  │
           └──────────────┬──────────────┘
                          │
           ┌──────────────┴──────────────┐
           │  Data Preprocessing Layer   │
           │  Cleaning, Imputation, EDA  │
           └──────────────┬──────────────┘
                          │
           ┌──────────────┴──────────────┐
           │     Feature Engineering     │
           │  (numeric + textual fusion) │
           └──────────────┬──────────────┘
                          │
           ┌──────────────┴──────────────┐
           │      Model Training         │
           │  Risk, NLP, Recommendation  │
           └──────────────┬──────────────┘
                          │
           ┌──────────────┴──────────────┐
           │   Explainability & Serving  │
           │   SHAP, Reports, API layer  │
           └─────────────────────────────┘

### How to Use

1. Clone the Repository:
   ```bash
   git clone https://github.com/<your-username>/predictive-maintenance-intelligence-suite.git
   cd predictive-maintenance-intelligence-suite

2. Install Dependencies:
   ```bash
   pip install -r requirements.txt

3. Run Data Preprocessing
   ```bash
   jupyter notebook notebooks/01_scaniadata_eda_preprocessing.ipynb

4. Train the Models
    ```bash
   jupyter notebook notebooks/02_failure_risk_modeling.ipynb
   jupyter notebook notebooks/03_ticket_triage_nlp.ipynb
   jupyter notebook notebooks/04_parts_technician_recommendation.ipynb

5. View Reports and Explainability Outputs

### Key Deliverables

✅ Cleaned tabular and textual datasets.

✅ Failure risk prediction model (LightGBM + SHAP explainability).

✅ DistilBERT ticket triage classifier with multi-task outputs.

✅ Apriori-based recommendation engine for parts and technician matching.

✅ Unified evaluation dashboards and calibration metrics.
 
### Technical Stack
| Component         | Technology                                                                   |
| ----------------- | ---------------------------------------------------------------------------- |
| **Language**      | Python 3.10+                                                                 |
| **Libraries**     | Pandas, NumPy, Scikit-learn, LightGBM, CatBoost, Transformers, SHAP, Mlxtend |
| **Visualization** | Matplotlib, Seaborn, Plotly                                                  |
| **Environment**   | Jupyter / Google Colab                                                       |
| **Serving**       | FastAPI (for deployment phase)                                               |


### License

This project is licensed under the MIT License — see the LICENSE file for details.

### Contact & Collaboration


    
    👩‍💼 Zaynab Atwi

Biomedical Engineer | BCI Researcher | Founder & CEO – VivoSalus Ventures

🔗 [LinkedIn](https://www.linkedin.com/in/zaynabatwi/)

For partnership inquiries or research collaboration, please contact:

📧 [zaynabatwi.143@gmail.com](zaynabatwi.143@gmail.com)



