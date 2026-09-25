# main.py - Beautiful BudgetBot AI Interface
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils import load_preprocessor, load_model_by_name, load_metrics, predict_budget, compute_emi
from utils import get_gemini_response

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="BudgetBot AI — Smart Budget Planning",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "BudgetBot AI - Intelligent Budget Planning powered by Machine Learning"
    }
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    * {
        margin: 0;
        padding: 0;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #1f77b4;
    }
    
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border-left: 5px solid #1f77b4;
    }
    
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px;
        border-radius: 16px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    .st-tabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
    }
    
    hr {
        margin: 20px 0;
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURE VISUALIZATION STYLE
# ============================================================================
sns.set(style="whitegrid")
sns.set_palette("husl")

# ============================================================================
# LOAD PROJECT ARTIFACTS
# ============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

preprocessor = load_preprocessor()
metrics = load_metrics()

DEFAULT_TARGETS = ['Rent','Groceries','Transport','Utilities','Healthcare','Education','Eating_Out','Entertainment','Loan_Repayment','Insurance','Miscellaneous','Savings']
TARGETS = DEFAULT_TARGETS.copy()

try:
    feat_names = list(preprocessor.feature_names_in_)
except Exception:
    feat_names = []

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown("""
    <div class="header-card">
        <h1>💰 BudgetBot AI</h1>
        <p style="font-size: 18px; margin-top: 10px; opacity: 0.9;">Intelligent Budget Planning Powered by Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("---")
    
    model_choice = st.selectbox(
        "Select AI Model",
        ["RandomForest", "Linear", "DecisionTree", "XGBoost_Bundle", "XGBoost_Tuned"],
        help="Different models may yield different predictions. XGBoost models typically perform best."
    )
    
    st.markdown("---")
    # Show basic model performance metrics (averaged across targets)
    try:
        metrics_for_models = metrics or {}
        key_map = {
            "XGBoost_Bundle": "XGBoost",
            "XGBoost_Tuned": "XGBoost (Tuned)"
        }
        metrics_key = key_map.get(model_choice, model_choice)
        model_metrics = metrics_for_models.get(metrics_key, {})
        if model_metrics:
            rmses = [v.get("RMSE", 0) for v in model_metrics.values()]
            r2s = [v.get("R\u00b2", v.get("R2", None)) for v in model_metrics.values()]
            maes = [v.get("MAE", 0) for v in model_metrics.values()]
            avg_rmse = float(np.mean(rmses)) if len(rmses) else None
            avg_r2 = float(np.mean([r for r in r2s if r is not None])) if any(r is not None for r in r2s) else None
            avg_mae = float(np.mean(maes)) if len(maes) else None

            st.metric("Accuracy", f"{(avg_r2*100):.2f}%", help="Average R² across all targets")
            st.metric("R²", f"{avg_r2:.4f}" if avg_r2 is not None else "N/A", help="Coefficient of determination")
    except Exception:
        # Fail silently if metrics are unavailable or malformed
        pass
    st.markdown("---")
# ============================================================================
# API CONFIGURATION – GEMINI
# ============================================================================
st.sidebar.markdown("### 🤖 AI Status")
GEMINI_API_KEY = your_api_key 
# use your api key in this place
# Load Gemini API Key securely from Streamlit secrets or environment variables
GEMINI_API_KEY = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    st.sidebar.success("✅ AI Assistant: Enabled")
    gemini_api_key = GEMINI_API_KEY
else:
    st.sidebar.error("❌ AI Assistant: Disabled")
    gemini_api_key = None

st.caption(
    "BudgetBot AI uses Machine Learning + Gemini AI for personalized financial guidance."
)

# ============================================================================
# USER INPUT SECTION
# ============================================================================
st.markdown("### 👤 Tell Us About Yourself")

# Create tabs for organized input
input_tab1, input_tab2 = st.tabs(["Personal Information", "Financial Details"])

with input_tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        income = st.number_input(
            "💵 Monthly Income (₹)",
            min_value=0.0,
            value=60000.0,
            step=1000.0,
            format="%.2f",
            help="Your monthly take-home income"
        )
    
    with col2:
        age = st.number_input(
            "🎂 Age",
            min_value=16,
            max_value=100,
            value=28,
            help="Your age affects expense patterns"
        )
    
    with col3:
        dependents = st.number_input(
            "👨‍👩‍👧‍👦 Dependents",
            min_value=0,
            value=2,
            help="Number of family members dependent on you"
        )

with input_tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        occupation = st.selectbox(
            "💼 Occupation",
            ["IT Professional","Professional","Self_Employed","Student","Retired","Other"],
            help="Your occupation type influences spending patterns"
        )
    
    with col2:
        city_tier = st.selectbox(
            "🏙️ City Tier",
            ["Tier_1","Tier_2","Tier_3","Unknown"],
            help="Urban classification affects cost of living"
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        debt = st.number_input(
            "💳 Outstanding Debt (₹)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help="Total amount you owe (loans, credit cards, etc.)"
        )
    
    with col2:
        emi_months = st.slider(
            "📅 EMI Repayment Term (years)",
            1, 10, 3,
            help="Expected duration to repay the debt"
        )

# Build user profile dictionary
user = {
    "Income": float(income),
    "Age": int(age),
    "Dependents": int(dependents),
    "Occupation": occupation,
    "City_Tier": city_tier,
    "debt": float(debt)
}

# Initialize chat history early
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display EMI estimation if debt exists
if debt > 0:
    emi_est = compute_emi(debt, annual_rate=0.12, years=emi_months)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "📊 Monthly EMI",
            f"₹{int(emi_est):,}",
            f"Principal: ₹{int(debt):,}"
        )
    with col2:
        st.metric(
            "📈 EMI Duration",
            f"{emi_months} years",
            f"{emi_months * 12} months"
        )
    with col3:
        total_repayment = int(emi_est * emi_months * 12)
        st.metric(
            "💰 Total Repayment",
            f"₹{total_repayment:,}",
            f"Includes interest"
        )

st.markdown("---")

# ============================================================================
# PREDICTION BUTTON
# ============================================================================
col_button, col_empty = st.columns([2, 1])

with col_button:
    predict_clicked = st.button(
        "🚀 Generate Smart Budget Prediction",
        use_container_width=True,
        type="primary"
    )

if "budget_generated" not in st.session_state:
    st.session_state.budget_generated = False

if predict_clicked:
    st.session_state.budget_generated = True

# ============================================================================
# RESULTS SECTION
# ============================================================================
if st.session_state.budget_generated:
    with st.spinner("🔄 Analyzing your financial profile with AI..."):
        try:
            # Load model and generate prediction
            model = load_model_by_name(model_choice)
            pred = predict_budget(
                user, preprocessor, model, model_choice, TARGETS,
                num_cols=[c for c in feat_names if c in ['Age','Income','Dependents','debt','debt_to_income_ratio','expense_ratio','Loan_Repayment']],
                cat_cols=['Occupation','City_Tier']
            )

            # Convert predictions to DataFrame
            df_pred = pd.DataFrame(list(pred.items()), columns=["Category","Amount"])
            df_pred['Amount'] = df_pred['Amount'].astype(float)
            df_pred = df_pred.sort_values(by="Amount", ascending=False)
            
            # Calculate total budget (excluding 'Savings' category)
            total_budget = df_pred[df_pred['Category'] != 'Savings']['Amount'].sum()
            savings = income - total_budget

            # Success notification
            st.markdown("""
                <div class="success-card">
                    ✅ <b>Budget Generated Successfully!</b> Here's your personalized budget breakdown.
                </div>
            """, unsafe_allow_html=True)

            # ================================================================
            # RESULTS TABS
            # ================================================================
            if not st.session_state.get('show_comparison', False):
                # Show normal tabs only when not showing comparison
                tab_breakdown, tab_visual, tab_insights, tab_export = st.tabs(
                    ["📊 Budget Breakdown", "📈 Visualizations", "💡 Insights", "💾 Export"]
                )
            else:
                # Show tabs without comparison tab when comparison is active
                tab_breakdown, tab_visual, tab_insights, tab_export = st.tabs(
                    ["📊 Budget Breakdown", "📈 Visualizations", "💡 Insights", "💾 Export"]
                )

            # ================================================================
            # TAB 1: BUDGET BREAKDOWN
            # ================================================================
            with tab_breakdown:
                st.subheader("Monthly Budget Allocation")
                
                # Summary KPIs
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        "💰 Monthly Income",
                        f"₹{int(income):,}"
                    )
                
                with metric_col2:
                    st.metric(
                        "📊 Total Budget",
                        f"₹{int(total_budget):,}",
                        f"{(total_budget/income*100):.1f}% of income"
                    )
                
                with metric_col3:
                    st.metric(
                        "🎯 Savings Potential",
                        f"₹{int(max(0, savings)):,}",
                        f"{(max(0, savings)/income*100):.1f}% of income" if income > 0 else "N/A"
                    )
                
                with metric_col4:
                    st.metric(
                        "📈 Top Category",
                        df_pred.iloc[0]['Category'],
                        f"₹{int(df_pred.iloc[0]['Amount']):,}"
                    )

                st.markdown("---")

                # Detailed budget table
                st.write("**Detailed Budget Categories:**")
                
                df_display = df_pred.copy()
                df_display['% of Budget'] = (df_display['Amount'] / total_budget * 100).round(1)
                
                # Format for display
                def format_currency(val):
                    return f'₹{int(val):,}'
                
                styled_df = df_display.style.format({
                    'Amount': format_currency,
                    '% of Budget': '{:.1f}%'
                }).background_gradient(subset=['Amount'], cmap='RdYlGn', vmin=df_display['Amount'].min(), vmax=df_display['Amount'].max())
                
                st.dataframe(styled_df, use_container_width=True, height=400)

            # ================================================================
            # TAB 2: VISUALIZATIONS
            # ================================================================
            with tab_visual:
                st.subheader("Budget Visualizations")
                
                # Filter out Savings for visualization
                df_viz = df_pred[df_pred['Category'] != 'Savings'].copy()
                viz_total = df_viz['Amount'].sum()
                
                vis_col1, vis_col2 = st.columns(2)
                
                # Bar chart
                with vis_col1:
                    st.write("**Expense Distribution by Category**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    colors = plt.cm.Set3(np.linspace(0, 1, len(df_viz)))
                    bars = ax.barh(df_viz["Category"], df_viz["Amount"], color=colors, edgecolor='black', linewidth=0.5)
                    ax.set_xlabel("Amount (₹)", fontsize=12, fontweight='bold')
                    ax.set_title("Monthly Budget by Category", fontsize=14, fontweight='bold', pad=20)
                    
                    for bar, amount in zip(bars, df_viz["Amount"]):
                        ax.text(amount + viz_total*0.01, bar.get_y() + bar.get_height()/2,
                               f'₹{int(amount):,}', va='center', fontweight='bold', fontsize=9)
                    
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                # Pie chart
                with vis_col2:
                    st.write("**Budget Composition (Pie Chart)**")
                    fig, ax = plt.subplots(figsize=(8, 8))
                    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(df_viz)))
                    wedges, texts, autotexts = ax.pie(
                        df_viz["Amount"],
                        labels=df_viz["Category"],
                        autopct=lambda pct: f'{pct:.1f}%' if pct > 3 else '',
                        startangle=90,
                        colors=colors_pie,
                        textprops={'fontsize': 9, 'weight': 'bold'},
                        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
                    )
                    ax.set_title("Budget Share Distribution", fontsize=14, fontweight='bold', pad=20)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                
                # Stacked bar - Income allocation flow
                st.write("**How Your Income is Allocated**")
                fig, ax = plt.subplots(figsize=(14, 3))
                
                cumulative = 0
                colors_flow = plt.cm.Set3(np.linspace(0, 1, len(df_viz)))
                
                for idx, (cat, amount) in enumerate(zip(df_viz["Category"], df_viz["Amount"])):
                    ax.barh(0, amount, left=cumulative, height=0.8, label=cat, color=colors_flow[idx], edgecolor='white', linewidth=2)
                    percentage = (amount / viz_total) * 100
                    if percentage > 5:  # Only show label if segment is large enough
                        ax.text(cumulative + amount/2, 0, f'{percentage:.0f}%',
                               ha='center', va='center', fontweight='bold', fontsize=9, color='white')
                    cumulative += amount
                
                # Add savings segment at the end
                ax.barh(0, savings, left=cumulative, height=0.8, label='Savings', color='#FFD700', edgecolor='white', linewidth=2)
                savings_pct = (savings / income) * 100
                if savings_pct > 5:
                    ax.text(cumulative + savings/2, 0, f'{savings_pct:.0f}%',
                           ha='center', va='center', fontweight='bold', fontsize=9, color='black')
                
                ax.set_xlim(0, income)
                ax.set_ylim(-0.5, 0.5)
                ax.set_xlabel("Monthly Income (₹)", fontsize=12, fontweight='bold')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize=8)
                ax.set_yticks([])
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            # ================================================================
            # TAB 3: INSIGHTS
            # ================================================================
            with tab_insights:
                st.subheader("💡 Smart Insights & Recommendations")
                
                # User profile summary
                st.write("**Your Financial Profile:**")
                profile_col1, profile_col2, profile_col3 = st.columns(3)
                
                with profile_col1:
                    st.info(f"👤 **Age:** {age} years\n\n👨‍👩‍👧‍👦 **Dependents:** {dependents}")
                
                with profile_col2:
                    st.info(f"💼 **Occupation:** {occupation}\n\n🏙️ **City Tier:** {city_tier}")
                
                with profile_col3:
                    if debt > 0:
                        debt_ratio = (debt / income * 100) if income > 0 else 0
                        st.warning(f"💳 **Debt-to-Income:** {debt_ratio:.1f}%\n\n📊 **Outstanding Debt:** ₹{int(debt):,}")
                    else:
                        st.success(f"✨ **Debt Status:** Debt Free!\n\n📊 **Outstanding Debt:** ₹0")
                
                st.markdown("---")
                
                # Generate insights
                st.write("**Key Financial Insights:**")
                
                insights = []
                
                # Insight 1: Top expenses
                top_3 = df_pred.head(3)
                top_expense_pct = (top_3['Amount'].sum() / total_budget) * 100
                insights.append({
                    'emoji': '📌',
                    'title': 'Top 3 Expense Categories',
                    'description': f"Your top expenses are {', '.join(top_3['Category'].tolist())}, which account for **{top_expense_pct:.1f}%** of your budget. Monitor these closely as they have the most impact on your finances."
                })
                
                # Insight 2: Savings potential
                if savings > 0:
                    savings_pct = (savings / income) * 100
                    insights.append({
                        'emoji': '🎯',
                        'title': 'Savings Potential',
                        'description': f"You have **₹{int(savings):,}/month** ({savings_pct:.1f}% of income) available to save or invest. Consider building an emergency fund first, then investing for long-term wealth."
                    })
                else:
                    insights.append({
                        'emoji': '⚠️',
                        'title': 'Budget Alert',
                        'description': f"Your projected expenses exceed income by **₹{int(abs(savings)):,}**. Review discretionary spending and consider additional income sources."
                    })
                
                # Insight 3: Debt management
                if debt > 0:
                    debt_ratio = (debt / income) * 100
                    if debt_ratio > 100:
                        insights.append({
                            'emoji': '💡',
                            'title': 'Debt Management Priority',
                            'description': f"Your debt-to-income ratio is **{debt_ratio:.1f}%** (high). Prioritize debt repayment to reduce financial stress and improve credit score."
                        })
                    elif debt_ratio > 50:
                        insights.append({
                            'emoji': '💡',
                            'title': 'Manageable Debt Levels',
                            'description': f"Debt-to-income ratio is **{debt_ratio:.1f}%** (moderate). Maintain current repayment while building emergency fund and savings."
                        })
                    else:
                        insights.append({
                            'emoji': '✅',
                            'title': 'Strong Debt Position',
                            'description': f"Debt-to-income ratio is **{debt_ratio:.1f}%** (healthy). You have good control over debt. Focus on strategic investing."
                        })
                
                # Insight 4: Age-based recommendations
                if age < 30:
                    insights.append({
                        'emoji': '🌱',
                        'title': 'Youth Advantage',
                        'description': f"At {age}, you have time on your side. Build an emergency fund (3-6 months expenses), start investing early (index funds, SIPs), and watch compound growth work for you."
                    })
                elif age < 50:
                    insights.append({
                        'emoji': '🎯',
                        'title': 'Peak Earning Years',
                        'description': f"Make the most of these prime years. Maximize retirement contributions, invest in mutual funds, plan for children's education, and consider life insurance."
                    })
                else:
                    insights.append({
                        'emoji': '🏡',
                        'title': 'Wealth Protection',
                        'description': f"Focus on protecting assets and ensuring sustainable retirement income. Review insurance coverage, diversify investments, and plan estate succession."
                    })
                
                # Display insights
                for insight in insights:
                    st.write(f"{insight['emoji']} **{insight['title']}**")
                    st.write(insight['description'])
                    st.write("")  # Spacing

                # Display insights
                for insight in insights:
                    st.write(f"{insight['emoji']} **{insight['title']}**")
                    st.write(insight['description'])
                    st.write("")  # Spacing

            # ================================================================
            # TAB 5: EXPORT
            # ================================================================
            with tab_export:
                st.subheader("📥 Export Your Budget Report")
                
                # Create comprehensive export data
                export_summary = {
                    "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Model": model_choice,
                    "User Profile": {
                        "Age": age,
                        "Monthly Income": float(income),
                        "Dependents": dependents,
                        "Occupation": occupation,
                        "City Tier": city_tier,
                        "Outstanding Debt": float(debt)
                    },
                    "Budget Summary": {
                        "Total Monthly Budget": float(total_budget),
                        "Monthly Savings": float(max(0, savings)),
                        "Savings %": float((max(0, savings)/income*100) if income > 0 else 0)
                    },
                    "Budget Breakdown": pred
                }
                
                # Download buttons
                exp_col1, exp_col2, exp_col3 = st.columns(3)
                
                with exp_col1:
                    json_data = json.dumps(export_summary, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📄 Download as JSON",
                        data=json_data,
                        file_name=f"BudgetBot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with exp_col2:
                    csv_data = df_display.to_csv(index=False)
                    st.download_button(
                        label="📊 Download as CSV",
                        data=csv_data,
                        file_name=f"BudgetBot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with exp_col3:
                    # Create a simple text report
                    report_text = f"""BUDGETBOT AI - BUDGET REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {model_choice}

USER PROFILE
Age: {age}
Monthly Income: ₹{int(income):,}
Dependents: {dependents}
Occupation: {occupation}
City Tier: {city_tier}
Outstanding Debt: ₹{int(debt):,}

BUDGET SUMMARY
Total Monthly Budget: ₹{int(total_budget):,}
Monthly Savings: ₹{int(max(0, savings)):,}
Savings %: {(max(0, savings)/income*100) if income > 0 else 0:.1f}%

BUDGET BREAKDOWN
"""
                    for _, row in df_display.iterrows():
                        report_text += f"\n{row['Category']}: ₹{int(row['Amount']):,} ({row['% of Budget']:.1f}%)"
                    
                    st.download_button(
                        label="📝 Download as Text",
                        data=report_text,
                        file_name=f"BudgetBot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.markdown("---")
                
                # Summary table
                st.write("**Budget Summary:**")
                summary_data = {
                    "Parameter": [
                        "Generated Date",
                        "AI Model Used",
                        "Monthly Income",
                        "Total Budget",
                        "Monthly Savings",
                        "Savings %"
                    ],
                    "Value": [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        model_choice,
                        f"₹{int(income):,}",
                        f"₹{int(total_budget):,}",
                        f"₹{int(max(0, savings)):,}",
                        f"{(max(0, savings)/income*100) if income > 0 else 0:.1f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.table(summary_df)

        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please verify your inputs and try again. If the problem persists, contact support.")


# ================================================================
# SCENARIO COMPARISON (Outside predict_clicked block)
# ================================================================
if st.session_state.budget_generated:
    if not st.session_state.get('show_comparison', False):
        st.markdown("---")
        st.subheader("⚖️ Compare Budget Scenarios")
        st.write("See how your budget allocation changes with and without debt")
        
        if debt > 0:  # Only show comparison if there's actual debt
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.write("**Scenario 1: With Current Debt**")
                st.info(f"💳 Debt Amount: ₹{int(debt):,}")
                emi_val = compute_emi(debt, annual_rate=0.12, years=emi_months)
                st.caption(f"Monthly EMI: ₹{int(emi_val):,}")
            
            with comp_col2:
                st.write("**Scenario 2: Debt-Free**")
                st.success("✨ No outstanding debt")
                st.caption("Compare allocation without debt burden")

            if st.button("📊 Generate Comparison", use_container_width=True, key="compare_btn"):
                st.session_state.show_comparison = True
                st.rerun()
        else:
            st.info("💡 No debt to compare. Add debt amount to see scenario comparison.")
    
    if st.session_state.get('show_comparison', False) and debt > 0:
        # Display comparison results in full width
        st.markdown("---")
        st.subheader("⚖️ Budget Comparison: With Loan vs Debt-Free")
        
        # Back button
        if st.button("← Back to Results", use_container_width=False):
            st.session_state.show_comparison = False
            st.rerun()
        
        model_comp = load_model_by_name(model_choice)
        
        with st.spinner("Analyzing both scenarios..."):
            # WITH DEBT - Scenario 1
            pred_with = predict_budget(
                user, preprocessor, model_comp, model_choice, TARGETS,
                num_cols=[c for c in feat_names if c in ['Age','Income','Dependents','debt','debt_to_income_ratio','expense_ratio','Loan_Repayment']],
                cat_cols=['Occupation','City_Tier']
            )
            
            # WITHOUT DEBT - Scenario 2 (intelligent reallocation)
            user_no_debt = user.copy()
            user_no_debt['debt'] = 0.0
            pred_no_debt = predict_budget(
                user_no_debt, preprocessor, model_comp, model_choice, TARGETS,
                num_cols=[c for c in feat_names if c in ['Age','Income','Dependents','debt','debt_to_income_ratio','expense_ratio','Loan_Repayment']],
                cat_cols=['Occupation','City_Tier']
            )
            
            # Calculate EMI freed up
            emi_freed = pred_with.get('Loan_Repayment', 0)
            
            # Smart reallocation: Distribute freed EMI to Savings and Education
            # Priority: Savings 60%, Education 40%
            if emi_freed > 0:
                savings_increase = emi_freed * 0.6
                education_increase = emi_freed * 0.4
                
                # Apply reallocation
                pred_reallocated = pred_no_debt.copy()
                pred_reallocated['Savings'] = max(0, pred_no_debt.get('Savings', 0) + savings_increase)
                pred_reallocated['Education'] = max(0, pred_no_debt.get('Education', 0) + education_increase)
                pred_reallocated['Loan_Repayment'] = 0
            else:
                pred_reallocated = pred_no_debt.copy()
            
            # Create comparison dataframes
            df_with = pd.DataFrame(list(pred_with.items()), columns=["Category", "Amount (₹)"])
            df_with['Amount (₹)'] = df_with['Amount (₹)'].astype(float).round(2)
            df_with['% of Budget'] = (df_with['Amount (₹)'] / df_with['Amount (₹)'].sum() * 100).round(2)
            df_with = df_with.sort_values('Amount (₹)', ascending=False).reset_index(drop=True)
            
            df_debt_free = pd.DataFrame(list(pred_reallocated.items()), columns=["Category", "Amount (₹)"])
            df_debt_free['Amount (₹)'] = df_debt_free['Amount (₹)'].astype(float).round(2)
            df_debt_free['% of Budget'] = (df_debt_free['Amount (₹)'] / df_debt_free['Amount (₹)'].sum() * 100).round(2)
            df_debt_free = df_debt_free.sort_values('Amount (₹)', ascending=False).reset_index(drop=True)
            
            # Display side-by-side tables
            st.markdown("---")
            st.write("**Budget Allocation Comparison:**")
            
            tab_col1, tab_col2 = st.columns(2)
            
            with tab_col1:
                st.write("### 💳 SCENARIO 1: WITH CURRENT LOAN")
                st.caption(f"Monthly Debt Payment (EMI): ₹{int(emi_freed):,}")
                
                # Format for display
                df_with_display = df_with.copy()
                df_with_display['Amount (₹)'] = df_with_display['Amount (₹)'].apply(lambda x: f"₹{int(x):,}")
                df_with_display['% of Budget'] = df_with_display['% of Budget'].apply(lambda x: f"{x:.2f}%")
                
                styled_with = df_with_display.style.set_properties(**{'text-align': 'left'})
                st.dataframe(styled_with, use_container_width=True, hide_index=True)
                
                total_with = df_with['Amount (₹)'].sum()
                savings_with = df_with[df_with['Category'] == 'Savings']['Amount (₹)'].sum()
                st.metric("Total Budget", f"₹{int(total_with):,}")
                st.metric("Savings", f"₹{int(savings_with):,}", f"{(savings_with/total_with*100):.1f}% of budget")
            
            with tab_col2:
                st.write("### ✨ SCENARIO 2: DEBT-FREE")
                st.caption(f"Freed EMI reallocated: Savings (60%) + Education (40%)")
                
                # Format for display
                df_debt_free_display = df_debt_free.copy()
                df_debt_free_display['Amount (₹)'] = df_debt_free_display['Amount (₹)'].apply(lambda x: f"₹{int(x):,}")
                df_debt_free_display['% of Budget'] = df_debt_free_display['% of Budget'].apply(lambda x: f"{x:.2f}%")
                
                styled_free = df_debt_free_display.style.set_properties(**{'text-align': 'left'})
                st.dataframe(styled_free, use_container_width=True, hide_index=True)
                
                total_free = df_debt_free['Amount (₹)'].sum()
                savings_free = df_debt_free[df_debt_free['Category'] == 'Savings']['Amount (₹)'].sum()

# AI functionality is handled in utils.py and main.py tabs.
# Definitions of legacy AI functions are removed to avoid confusion.

        # Comparison Summary (Rendered after chat logic but before footer)
    if st.session_state.get('show_comparison', False):
        st.markdown("---")
        st.subheader("📊 Comparison Summary")
            
        savings_increase = df_debt_free[df_debt_free['Category'] == 'Savings']['Amount (₹)'].sum() - df_with[df_with['Category'] == 'Savings']['Amount (₹)'].sum()
        education_increase = df_debt_free[df_debt_free['Category'] == 'Education']['Amount (₹)'].sum() - df_with[df_with['Category'] == 'Education']['Amount (₹)'].sum()
        savings_rate_with = (df_with[df_with['Category'] == 'Savings']['Amount (₹)'].sum() / income) * 100
        savings_rate_free = (df_debt_free[df_debt_free['Category'] == 'Savings']['Amount (₹)'].sum() / income) * 100
            
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
            
        with summary_col1:
            st.metric(
                "💰 Monthly EMI Freed",
                f"₹{int(emi_freed):,}",
                "Available after loan closure"
            )
            
        with summary_col2:
            st.metric(
                "🎯 Additional Savings",
                f"₹{int(savings_increase):,}",
                f"+{(savings_increase/income*100):.1f}% of income"
            )
            
        with summary_col3:
            st.metric(
                "📚 Education Investment",
                f"₹{int(education_increase):,}",
                f"+{(education_increase/income*100):.1f}% of income"
            )
            
        with summary_col4:
            st.metric(
                "📈 Improved Savings Rate",
                f"{savings_rate_free:.1f}%",
                f"+{(savings_rate_free - savings_rate_with):.1f}pp from {savings_rate_with:.1f}%"
            )
            
        # Key Insights
        st.markdown("---")
        st.write("**Key Financial Insights:**")
            
        insight_col1, insight_col2 = st.columns(2)
            
        with insight_col1:
            st.success(f"""
            **🎉 Benefits of Debt-Free Living:**
                
            • **Monthly cash freed:** ₹{int(emi_freed):,} available every month
            • **Increased savings:** ₹{int(savings_increase):,}/month → Build emergency fund faster
                • **Better education investment:** ₹{int(education_increase):,}/month → More financial security
                • **Higher savings rate:** {savings_rate_free:.1f}% (vs {savings_rate_with:.1f}% with debt)
                • **Improved credit score:** No loan payments boost financial health
                """)
            
        with insight_col2:
            st.info(f"""
            **💡 Reallocation Strategy:**
                
            • Fixed expenses remain unchanged:
            • Freed EMI (₹{int(emi_freed):,}) allocated as:
                  - 60% → Savings (₹{int(savings_increase):,})
                  - 40% → Education (₹{int(education_increase):,})
                • No luxury categories increased
                • Total budget remains at ₹{int(income):,}
                """)
            
        # Category-wise comparison table
        st.markdown("---")
        st.write("**Detailed Category Comparison:**")
            
        # Create detailed comparison
        df_detailed = pd.DataFrame({
                "Category": df_with['Category'],
                "With Loan (₹)": df_with['Amount (₹)'],
                "Debt-Free (₹)": df_debt_free.set_index('Category').loc[df_with['Category'], 'Amount (₹)'].values,
            })
            
        df_detailed['Change (₹)'] = df_detailed['Debt-Free (₹)'] - df_detailed['With Loan (₹)']
        df_detailed['% Change'] = (df_detailed['Change (₹)'] / df_detailed['With Loan (₹)'].replace(0, 1) * 100).round(2)
        df_detailed = df_detailed.sort_values('Change (₹)', ascending=False).reset_index(drop=True)
            
        # Format for display
        styled_detailed = df_detailed.style.format({
                'With Loan (₹)': '₹{:,.0f}',
                'Debt-Free (₹)': '₹{:,.0f}',
                'Change (₹)': '₹{:,.0f}',
                '% Change': '{:.2f}%'
            }).background_gradient(
                subset=['Change (₹)'],
                cmap='RdYlGn',
                vmin=df_detailed['Change (₹)'].min(),
                vmax=df_detailed['Change (₹)'].max()
            )
            
        st.dataframe(styled_detailed, use_container_width=True, hide_index=True)

# ============================================================================
# PERSISTENT AI CHAT ASSISTANT (ALWAYS AVAILABLE)
# ============================================================================
st.markdown("---")
st.header("🤖 Chat with BudgetBot AI")

# Build context safely
if st.session_state.get("budget_generated", False):
    try:
        ctx = f"User Profile: {user}. Total Budget: ₹{int(total_budget)}. Savings Potential: ₹{int(savings)}."
    except Exception:
        ctx = "User has partial budget data. Give general financial advice."
else:
    ctx = "User has not generated a budget yet. Give general financial advice."

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and logic
with st.form(key="chat_form", clear_on_submit=True):
    chat_col1, chat_col2 = st.columns([6, 1])
    with chat_col1:
        user_input = st.text_input("Ask me anything about your finances...", key="user_input_box", label_visibility="collapsed", placeholder="Ask me anything about your finances...")
    with chat_col2:
        submit_button = st.form_submit_button("Send", use_container_width=True)

    if submit_button and user_input:
        prompt = user_input
        st.session_state.messages.append({"role": "user", "content": prompt})

        if gemini_api_key:
            try:
                res = get_gemini_response(prompt, gemini_api_key, ctx)
                st.session_state.messages.append({"role": "assistant", "content": res})
            except Exception as e:
                st.error(f"❌ AI Error: {str(e)}")
        else:
            st.warning("⚠️ AI Assistant is disabled. Check your API configuration.")

# ============================================================================
# FOOTER (Always visible at the absolute bottom)
# ============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; color: #666; border-top: 1px solid #e0e0e0; font-family: sans-serif;">
    <p style="margin: 0; font-weight: bold; font-size: 1.1em;">🤖 BudgetBot AI © 2025</p>
    <p style="margin: 5px 0; font-size: 0.9em; color: #888;">Powered by Machine Learning | Your Personal AI Budget Advisor</p>
</div>
""", unsafe_allow_html=True)
