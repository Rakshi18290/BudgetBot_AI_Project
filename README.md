# 💰 BudgetBot AI — Smart Personal Finance & Budget Planning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Powered-green.svg)](https://xgboost.readthedocs.io/)
[![Google Gemini AI](https://img.shields.io/badge/Google%20Gemini-AI%20Advisor-8E44AD.svg)](https://aistudio.google.com/)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**BudgetBot AI** is an intelligent, machine-learning-driven personal finance application designed to predict category-wise monthly budgets, evaluate financial health, compute loan EMIs, and deliver personalized AI financial guidance powered by **Google Gemini AI**.

---

## 🌟 Key Features

- **📊 Multi-Output Machine Learning Predictions**:
  - Leverages trained ML algorithms (**XGBoost**, **Random Forest**, **Decision Tree**, **Linear Regression**) to forecast spending across 10 key categories:
    - 🛒 Groceries
    - ⚡ Utilities
    - 🏠 Rent
    - 💳 Loan Repayment & EMIs
    - 🎬 Entertainment
    - 🏥 Healthcare
    - 🛡️ Insurance
    - 🚗 Transportation
    - 🍽️ Eating Out
    - 💰 Recommended Savings
- **🤖 Built-in AI Financial Advisor**:
  - Integrated with **Google Gemini 2.5 Flash** for India-specific personal finance guidance, tax planning strategies, debt reduction tactics, and budgeting insights.
- **🧮 Smart EMI & Debt Calculator**:
  - Calculates loan monthly installments automatically and integrates debt ratios directly into predicted budget profiles.
- **📈 Model Comparison & Metrics Dashboard**:
  - Transparent model performance indicators ($R^2$ score: **~0.92**, Accuracy: **91.97%**) with metrics visualization.
- **🎨 Glassmorphic Streamlit Dashboard**:
  - Highly dynamic, modern, responsive user interface with custom metric cards and intuitive layout.

---

## 🏗️ Project Architecture

```
BudgetBot/
├── BudgetBot_AI_Project1/
│   ├── frontend_app/
│   │   ├── main.py             # Streamlit Application Entrypoint
│   │   ├── utils.py            # ML Model Loader, Preprocessor, & Gemini AI Integration
│   │   └── requirements.txt    # Application Dependencies
│   ├── models/                 # Saved Model Artifacts & Preprocessors
│   │   ├── xgb_tuned.pkl       # Optimized XGBoost MultiOutput Regressor
│   │   ├── xgb_models_bundle.pkl
│   │   ├── dt_multioutput.pkl  # Decision Tree Model
│   │   ├── preprocessor.pkl    # Sklearn Feature Pipeline
│   │   └── model_metrics.csv   # Performance Summary
│   ├── data/                   # Train/Test Parquet & CSV Datasets
│   └── results/                # Metrics Summary JSON Output
├── .gitignore                  # Git Exclusion Configuration
└── README.md                   # Project Documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python **3.10+** (Tested on Python 3.10 – 3.14)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Rakshi18290/BudgetBot_AI_Project.git
cd BudgetBot_AI_Project/BudgetBot_AI_Project1/frontend_app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run main.py
```
Open your browser at `http://localhost:8501`.

---

## 🤖 Configuring Google Gemini AI Assistant (Optional)

To enable the interactive **BudgetBot AI Financial Advisor**:
1. Get a free API Key from [Google AI Studio](https://aistudio.google.com/).
2. Enter your API Key in the application sidebar or set it as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

---

## 📊 Machine Learning Model Benchmarks

| Model | $R^2$ Score | MAE | RMSE | Status |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost (Tuned)** | **0.9197** | **Low** | **Optimal** | 🏆 Best Model |
| Random Forest | 0.9150 | Medium | Low | Strong |
| Decision Tree | 0.8842 | Medium | Medium | Baseline |
| Linear Regression | 0.8120 | High | High | Benchmark |

---

## 🛠️ Built With

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Generative AI**: [Google Gemini AI (`google-genai`)](https://pypi.org/project/google-genai/)
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
