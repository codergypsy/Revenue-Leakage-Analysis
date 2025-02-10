# Revenue Leakage Analysis in Insurance Claims

![Power BI Dashboard Preview](![image](https://github.com/user-attachments/assets/3c124f20-9a6e-4f47-b3e4-63aa454397e8)) 

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation Guide](#installation-guide)
3. [Dataset Generation](#dataset-generation)
4. [Data Analysis Pipeline](#data-analysis-pipeline)
5. [Model Training](#model-training)
6. [Power BI Dashboard](#power-bi-dashboard)
7. [Troubleshooting](#troubleshooting)
8. [Contributing](#contributing)
9. [License](#license)

## 📌 Project Overview <a name="project-overview"></a>
This project identifies potential revenue leakage points (>$500K annually) in insurance operations using machine learning (Random Forest) and data analytics. The solution analyzes policy data, claims history, and customer information to detect patterns leading to financial losses.

**Key Features**:
- Synthetic data generation for insurance domain
- Random Forest classification with feature importance analysis
- Power BI dashboard for business intelligence
- Automated data preprocessing pipeline
- Financial impact estimation module

## 🛠️ Installation Guide <a name="installation-guide"></a>

### Requirements
- Python 3.8+
- Power BI Desktop (for visualization)
- RAM: 8GB+ (16GB recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/codergypsy/revenue-leakage-analysis.git
cd revenue-leakage-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Generation <a name="dataset-generation"></a>
```bash
import pandas as pd
import numpy as np
from faker import Faker

def generate_insurance_data(num_records=10000):
    fake = Faker()
    np.random.seed(42)
    
    data = {
        'policy_id': [fake.uuid4() for _ in range(num_records)],
        'premium_amount': np.random.lognormal(8, 0.5, num_records),
        'coverage_type': np.random.choice(['Comprehensive', 'Liability', 'Collision'], num_records),
        'claim_count': np.random.poisson(1.2, num_records),
        'avg_claim_amount': np.abs(np.random.normal(15000, 6000, num_records)),
        # ... (full generator from previous code)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('insurance_data.csv', index=False)

if __name__ == "__main__":
    generate_insurance_data()
```

## Data Analysis Pipeline <a name="data-analysis-pipeline"></a>
#Feature Engineering
# Age group binning with inclusive ranges
```bash
df['age_group'] = pd.cut(
    df['customer_age'],
    bins=[17, 30, 45, 60, 80],
    labels=['18-30', '31-45', '46-60', '61-80'],
    right=False
)

# Financial ratios
df['premium_claim_ratio'] = np.divide(
    df['premium_amount'],
    df['avg_claim_amount'],
    where=df['avg_claim_amount'] > 0,
    out=np.zeros(len(df))
)

# Temporal features
df['policy_age'] = (pd.to_datetime('today') - df['policy_start_date']).dt.days // 365
```

## Model Training <a name="model-training"></a>
# Random Forest Classifier
```bash
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        class_weight='balanced',
        random_state=42
    ))
])

# Hyperparameter tuning
param_grid = {
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring='recall')
grid_search.fit(X_train, y_train)
```

## Power BI Dashboard <a name="power-bi-dashboard"></a>
#Key Components
**Metrics Card:**
```bash

1. Avg Leakage Probability = AVERAGE('leakage_analysis_results'[leakage_probability])
2. High Risk Policies = CALCULATE(
    COUNTROWS('leakage_analysis_results'),
    'leakage_analysis_results'[predicted_leakage] = 1
)
3. Leakage % of Total = 
DIVIDE(
    [Leakage Cases],
    COUNTROWS('leakage_analysis_results')
)
4. Recovery Potential = [Total Leakage] * 0.7
5. Total Leakage = SUM('leakage_analysis_results'[estimated_leakage_amount])



##Data Flow
```bash
graph LR
    A[Python Model] --> B{{Feature Importance}}
    A --> C{{Processed Data}}
    B --> D[Power BI]
    C --> D
    D --> E[Dashboard Visuals]
```

##Troubleshooting <a name="troubleshooting"></a>
# Handle infinite values
```bash
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.median(), inplace=True)
```

# Date Conversion
```bash
df['policy_start_date'] = pd.to_datetime(
    df['policy_start_date'],
    errors='coerce'
).dt.date
```
## Contributing <a name="contributing"></a>
Fork the repository

1. Create feature branch (git checkout -b feature/new-feature)
2. Commit changes (git commit -m 'Add new feature')
3. Push to branch (git push origin feature/new-feature)
4. Open Pull Request


