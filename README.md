# Mod24_AIML_FINAL_CAPSTONE-Sales-Analysis
Mod24_AIML_FINAL_CAPSTONE: Sales Analysis - Late deliveries and Fraudulent orders

# 🧠 Mod24_AIML_FINAL_CAPSTONE Project  
## **Sales Analysis — Late Delivery & Fraudulent orders**

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![ML](https://img.shields.io/badge/Machine%20Learning-ScikitLearn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)

---

### 📘 **Project Overview**

This project focuses on analyzing and predicting late deliveries and fraudulent transactions using machine learning classification techniques. The objective is to identify high-risk transactions and improve operational efficiency across the supply chain.
The dataset used is DataCo Supply Chain Dataset, which contains over 180,000 records of order, customer, and product details. Two target variables were created for analysis:
This project applies **machine learning classification** techniques to analyze and predict:
- **Late Delivery** — which orders are likely to arrive late.
- **Fraudulent orders** — which transactions show suspicious or fraudulent behavior.
- **Model Comparison** — Evaluate and compare the performance of various machine learning classification algorithms.

The dataset used is the **DataCo Supply Chain Dataset**, containing rich customer, product, and logistics information across global transactions.
The dataset can be downloaded from:

[DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS](https://data.mendeley.com/datasets/8gx2fvg2k6/5)
---

### 🎯 **Objectives**
1. Identify key factors influencing **late deliveries** and **fraudulent orders**.  
2. Build and compare multiple ML models for classification performance.  
3. Improve business efficiency through data-driven insights.

---

### ⚙️ **Workflow**

#### **1. Data Preprocessing**
- Handled missing values and duplicates.  
- Removed irrelevant or high-cardinality columns (e.g., `Customer Name`, `Product Description`).  
- Corrected data types (numeric, categorical, date).  

#### **2. Feature Engineering**
- Created new features   
- Dropped date columns before model training to prevent leakage.
- Derived new variables such as profit ratios and simplified categorical levels.
- Created binary flags for fraud (suspected_fraud) and late delivery (late_delivery_risk).
- Removed high-cardinality text columns (e.g., customer names, product descriptions).

#### **3. Exploratory Data Analysis (EDA)**
- **Univariate:** histograms, count plots for feature distribution.  
- **Bivariate:** relationships between target and key predictors.  
- **Multivariate:** correlation heatmaps for numeric features.

#### **4. Preprocessing & Transformation**
- **One-Hot Encoding (OHE)** for categorical features.  
- **StandardScaler / MinMaxScaler** for numeric normalization.  
- **Train-Test Split (80/20)** performed before scaling (to avoid leakage).  

#### **5. Model Building**
Trained and evaluated the following classification models:
- Logistic Regression  
- Gaussian Naive Bayes  
- Support Vector Machines (SVM)  
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Extra Trees Classifier  
- Extreme Gradient Boosting (XGBoost)

#### **6. Evaluation Metrics**
Models were compared using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Cross-Validation Accuracy**

---

### 📊 **Model Performance Summary**

| Model | Fraud F1 (%) | Late Delivery F1 (%) | Key Highlight |
|--------|---------------|----------------------|----------------|
| Logistic Regression | 30.78 | 98.96 | Simple baseline |
| Gaussian NB | 27.93 | 71.96 | Weak on correlated data |
| SVM | 29.30 | 98.96 | Stable but computationally heavy |
| KNN | 37.37 | 82.45 | Sensitive to scaling |
| Random Forest | 66.35 | 98.32 | Strong ensemble |
| Extra Trees | 59.57 | 99.27 | Fast, robust |
| XGBoost | 76.53 | 99.31 | Excellent balance |
| **Decision Tree** | **80.08** | **99.49** | ✅ Best overall performer |

---

### 🔍 **Feature Importance**
#### Fraud Detection:
- `Customer Full Name`, `Payment Type`, and `Shipping Mode` showed the highest predictive power.  
- Interestingly, `Days for Shipping (Real)` had moderate importance, suggesting potential operational anomalies.

#### Late Delivery Prediction:
- Top predictors: `Days for shipping (real)`, `Days for shipment (scheduled)`, and `Shipping Mode`.  
- Most delays occurred under **Standard Class shipping** and in **LATAM** regions.

---

### 🚀 **Key Findings**
- **Decision Tree Classifier** achieved the best balance between interpretability and accuracy.  
🔸 Fraud Detection
- Best Model: Decision Tree (F1 Score: 80.08%)
- Features like Customer Full Name, Shipping Mode, and Payment Type were key contributors.
- Fraud detection accuracy was ~99%, but recall was lower, indicating potential false negatives.

🔸 Late Delivery Risk
- Best Model: Decision Tree (F1 Score: 99.49%)
- Most late deliveries were from Standard Class Shipping and LATAM region.

Days for shipping and scheduled shipment time were dominant predictors.
---

### 💼 **Business Impact**
| Area | Impact |
|-------|--------|
| Fraud Detection | Early detection reduces revenue loss and fraud exposure. |
| Logistics Optimization | Identifies shipping modes or regions with recurring delays. |
| Customer Experience | Improved delivery reliability enhances satisfaction and retention. |

---

### 🧠 **Next Steps**
1. Perform **hyperparameter tuning** (GridSearchCV / RandomizedSearchCV) for top models.  
2. Use **SHAP / LIME** for interpretability and feature explainability.  
3. Deploy models as APIs or integrate into BI dashboards (Power BI / Streamlit).  
4. Schedule retraining and monitoring for **model drift** detection.

---

### 🧰 **Tech Stack**
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming | Python 3.9 |
| Libraries | pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imblearn |
| Environment | Jupyter Notebook / Google Colab |
| Visualization | matplotlib, seaborn |
| Version Control | GitHub |

---

### 📈 **Results**
- **F1 Score (Fraudulent orders):** 80.08%  
- **F1 Score (Late Delivery):** 99.49%  
- **Best Model:** Decision Tree Classifier  

---

### 🧾 **How to Run**
```bash
# Clone the repository
git clone https://github.com/<yourusername>/LateDelivery_Fraud_Detection.git
cd LateDelivery_Fraud_Detection

# Install dependencies
pip install -r requirements.txt

# Open the Jupyter Notebook
jupyter notebook- Mod24_AIML_CAPSTONE_Sales_Analysis_Late_Delivery_Risk_and_Fraud_Analysis - FINAL.ipynb
