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
  - Forecasts spending across 10 key categories:
    - 🛒 Groceries, ⚡ Utilities, 🏠 Rent, 💳 Loan Repayment, 🎬 Entertainment
    - 🏥 Healthcare, 🛡️ Insurance, 🚗 Transportation, 🍽️ Eating Out, 💰 Recommended Savings
- **🤖 Built-in AI Financial Advisor**:
  - Integrated with **Google Gemini 2.5 Flash** for personal finance guidance, tax planning strategies, debt reduction tactics, and budgeting insights.
- **🧮 Smart EMI & Debt Calculator**:
  - Calculates loan monthly installments automatically and integrates debt ratios directly into predicted budget profiles.
- **📈 Model Comparison & Metrics Dashboard**:
  - Transparent model performance indicators ($R^2$ score: **~0.92**, Accuracy: **91.97%**) with metrics visualization.
- **🎨 Glassmorphic Streamlit Dashboard**:
  - Highly dynamic, modern, responsive user interface with custom metric cards and intuitive layout.

---

## 🚀 Quick Start Guide

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch application
streamlit run main.py
```
Open your browser at `http://localhost:8501`.

---

## 📜 License

Distributed under the MIT License.
