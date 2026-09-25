# utils.py
import os
import joblib
import json
import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Load frequently-used artifacts (lazy load helpers)
def load_preprocessor():
    return joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))

def load_model_by_name(name):
    """
    name in ['Linear','DecisionTree','RandomForest','XGBoost_Bundle','XGBoost_Tuned']
    """
    if name == "Linear":
        return joblib.load(os.path.join(MODELS_DIR, "linear_multioutput.pkl"))
    if name == "DecisionTree":
        return joblib.load(os.path.join(MODELS_DIR, "dt_multioutput.pkl"))
    if name == "RandomForest":
        return joblib.load(os.path.join(MODELS_DIR, "rf_multioutput.pkl"))
    if name == "XGBoost_Bundle":
        return joblib.load(os.path.join(MODELS_DIR, "xgb_models_bundle.pkl"))
    if name == "XGBoost_Tuned":
        return joblib.load(os.path.join(MODELS_DIR, "xgb_tuned.pkl"))
    raise ValueError("Unknown model name")

def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

import requests

from google import genai
from google.genai import types

def get_gemini_response(user_prompt, api_key, context=""):
    if not api_key:
        return "AI is not enabled. Please provide a valid Gemini API key (preferably from Google AI Studio)."

    try:
        # Reverting to default version handling as v1 may be too restrictive
        client = genai.Client(api_key=api_key)
        
        system_prompt = (
            "You are BudgetBot AI, a professional Indian personal finance advisor.\n"
            "Rules:\n"
            "- Answer ONLY finance, budgeting, savings, debt, EMI, investment, tax, insurance questions.\n"
            "- Give practical, India-specific advice.\n"
            "- Be concise and professional.\n"
            "- Do NOT answer non-financial questions.\n"
            "- Do NOT use markdown symbols.\n"
            "- Politely refuse unrelated questions.\n"
        )

        final_prompt = (
            f"{system_prompt}\n\n"
            f"User Context:\n{context}\n\n"
            f"User Question:\n{user_prompt}"
        )

        # Updated fallback list with models confirmed to be available for this key
        model_names = [
            "gemini-2.5-flash",
            "gemini-2.0-flash", 
            "gemini-2.0-flash-lite-001",
            "gemini-2.0-flash-001",
            "gemini-1.5-flash"
        ]
        
        last_err = None
        for model_name in model_names:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=final_prompt
                )
                if response and response.text:
                    reply = response.text.strip()
                    return reply.replace("**", "").replace("*", "").replace("##", "").replace("#", "")
            except Exception as inner_e:
                last_err = inner_e
                continue
        
        # If we reach here, all fallbacks failed
        available_models = []
        try:
            for m in client.models.list():
                available_models.append(m.name)
        except:
            available_models = ["Could not list models"]

        return (
            f"AI Error: No compatible models found.\n\n"
            f"Last Error: {str(last_err)}\n\n"
            f"Available models for your key: {', '.join(available_models[:5])}..."
        )

    except Exception as e:
        error_str = str(e).lower()
        if "401" in error_str or "unauthorized" in error_str or "credentials" in error_str:
            return (
                "AI Auth Error: Your API key was rejected (401). \n\n"
                "Possible reasons:\n"
                "1. You are using a Google Cloud Console key instead of an AI Studio key.\n"
                "2. The API key is restricted or has expired.\n\n"
                "Recommended Fix: Please get a new key from https://aistudio.google.com/app/apikey"
            )
        return f"AI Error: {str(e)}"


# EMI helper
def compute_emi(principal, annual_rate=0.12, years=3):
    try:
        p = float(principal)
    except:
        return 0.0
    if p <= 0:
        return 0.0
    r = annual_rate / 12.0
    n = years * 12
    emi = (p * r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return round(emi, 2)

# Predict wrapper that matches earlier logic
def predict_budget(user_dict, preprocessor, model, model_name, TARGETS, num_cols, cat_cols):
    df_user = pd.DataFrame([user_dict])
    # ensure numeric columns exist
    for col in ['Age','Income','Dependents','debt']:
        if col not in df_user.columns:
            df_user[col] = 0.0

    # derived features
    df_user['debt_to_income_ratio'] = (df_user['debt'] / df_user['Income'].replace(0, np.nan)).replace([np.inf,-np.inf],0).fillna(0)
    df_user['expense_ratio'] = df_user.get('expense_ratio', 0.0)

    # Loan repayment if missing
    if 'Loan_Repayment' not in df_user.columns or df_user['Loan_Repayment'].iloc[0] == 0:
        df_user['Loan_Repayment'] = df_user['debt'].apply(lambda d: compute_emi(d))

    # ensure categorical exist
    for c in cat_cols:
        if c not in df_user.columns:
            df_user[c] = "Unknown"

    # align with preprocessor expected features
    feat_cols = list(preprocessor.feature_names_in_)
    for col in feat_cols:
        if col not in df_user.columns:
            df_user[col] = 0.0

    Xu = preprocessor.transform(df_user[feat_cols])

    # predict
    if model_name == "XGBoost_Bundle":
        preds = []
        for t in TARGETS:
            if t in model:
                m = model[t]
                try:
                    y = m.predict(Xu)
                except Exception:
                    # if booster
                    y = m.predict(xgb.DMatrix(Xu))
                preds.append(y[0])
            else:
                preds.append(0.0)
        ypred = np.array(preds)
    elif model_name == "XGBoost_Tuned":
        # model is a MultiOutputRegressor(GridSearchCV(...)) so behaves like sklearn estimator
        ypred = model.predict(Xu)[0]
    else:
        ypred = model.predict(Xu)[0]

    # map to dict
    pred = dict(zip(TARGETS, [float(round(float(v), 2)) for v in ypred]))
    # enforce non-negative
    for k in list(pred.keys()):
        pred[k] = max(0, pred[k])

    # compute savings if present in targets else add
    if 'Savings' in pred:
        pass
    else:
        others = sum(v for k,v in pred.items() if k.lower()!='savings')
        pred['Savings'] = max(0, round(float(df_user['Income'].iloc[0]) - others, 2))

    return pred
