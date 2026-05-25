
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Silent Churn Detector",
    page_icon="🔕",
    layout="wide"
)

# ─── STYLES ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f0f; }
    .metric-card {
        background: #1e1e1e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #333;
    }
    .risk-high { color: #ff4b4b; font-size: 2rem; font-weight: bold; }
    .risk-med  { color: #ffa500; font-size: 2rem; font-weight: bold; }
    .risk-low  { color: #00c853; font-size: 2rem; font-weight: bold; }
    h1 { color: #ffffff; }
    h2, h3 { color: #cccccc; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD & PREPARE DATA ───────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drift features
    df['engagement_score'] = df['tenure'] / (df['MonthlyCharges'] + 1)
    df['charge_pressure']  = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

    service_cols = ['PhoneService','InternetService','OnlineSecurity',
                    'OnlineBackup','DeviceProtection','TechSupport',
                    'StreamingTV','StreamingMovies']
    for col in service_cols:
        df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    df['services_used']    = df[service_cols].sum(axis=1)
    df['loyalty_index']    = df['tenure'] * df['services_used']
    df['drift_risk_score'] = df['MonthlyCharges'] / (df['services_used'] + 1)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return df, model, scaler, X, y, X_test, y_test

df, model, scaler, X, y, X_test, y_test = load_and_train()

# ─── HEADER ────────────────────────────────────────────────────
st.title("🔕 Silent Churn Detector")
st.markdown("#### Detecting behavioral drift in customers *before* they leave")
st.markdown("---")

# ─── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📉 Drift Signals",
    "🧠 Model Insights",
    "🔍 Live Predictor"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Dataset Overview")

    total     = len(df)
    churned   = df['Churn'].sum()
    retained  = total - churned
    churn_pct = round((churned / total) * 100, 1)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Churned", f"{churned:,}", delta=f"{churn_pct}%", delta_color="inverse")
    col3.metric("Retained", f"{retained:,}")
    col4.metric("Model AUC Score", f"{round(roc_auc_score(y_test, model.predict(X_test)), 3)}")

    st.markdown("---")
    st.subheader("What is Silent Churn?")
    st.info("""
    **Silent Churn** happens when customers gradually disengage *before* they officially cancel.
    Traditional churn models predict who will leave. This model detects **behavioral drift** —
    the early warning signals like declining engagement, rising cost pressure, and reducing service usage
    that indicate a customer is mentally already gone.
    """)

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(20), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — DRIFT SIGNALS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Behavioral Drift Signals — Churners vs Non-Churners")
    st.markdown("These are the **custom engineered features** that separate silent churners from loyal customers.")

    churned_df  = df[df['Churn'] == 1]
    retained_df = df[df['Churn'] == 0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f0f')

    features = ['engagement_score', 'drift_risk_score', 'loyalty_index']
    titles   = ['Engagement Score', 'Drift Risk Score', 'Loyalty Index']
    explanations = [
        "Low engagement = customer is checked out",
        "High drift risk = paying a lot, using little",
        "Low loyalty = long customer but no services used"
    ]

    for i, (feat, title) in enumerate(zip(features, titles)):
        axes[i].set_facecolor('#1a1a1a')
        axes[i].hist(retained_df[feat], bins=30, alpha=0.6, color='#4caf50', label='Stayed')
        axes[i].hist(churned_df[feat],  bins=30, alpha=0.6, color='#f44336', label='Churned')
        axes[i].set_title(title, color='white', fontsize=13)
        axes[i].tick_params(colors='white')
        axes[i].legend()
        for spine in axes[i].spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.info("📉 **Engagement Score**\n\nLower values = customer barely using the product relative to what they pay")
    col2.warning("⚠️ **Drift Risk Score**\n\nHigher values = customer paying more but using fewer services")
    col3.error("🚨 **Loyalty Index**\n\nLower values = long-tenure customers who haven't expanded usage — quietly disengaged")

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Top 15 Features Driving Silent Churn")

    feature_names = X.columns
    importances   = model.feature_importances_
    indices       = np.argsort(importances)[::-1][:15]

    fig2, ax = plt.subplots(figsize=(14, 6))
    fig2.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')
    ax.bar(range(15), importances[indices], color='#e57373', alpha=0.9)
    ax.set_xticks(range(15))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', color='white')
    ax.set_ylabel("Importance Score", color='white')
    ax.set_title("Feature Importance", color='white', fontsize=14)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Model Performance")

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    fig3.patch.set_facecolor('#0f0f0f')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Not Churned','Churned'],
                yticklabels=['Not Churned','Churned'], ax=ax3)
    ax3.set_title("Confusion Matrix", color='white')
    ax3.tick_params(colors='white')
    st.pyplot(fig3)

# ══════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔍 Live Customer Risk Predictor")
    st.markdown("Enter a customer's details below to get their **silent churn risk score** in real time.")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charge = st.slider("Monthly Charges ($)", 20, 120, 65)
        total_charges  = st.number_input("Total Charges ($)", 0.0, 9000.0, 800.0)

    with col2:
        services_used  = st.slider("Number of Services Used", 0, 8, 3)
        contract       = st.selectbox("Contract Type", [0, 1, 2],
                                       format_func=lambda x: ["Month-to-Month","One Year","Two Year"][x])
        payment_method = st.selectbox("Payment Method", [0, 1, 2, 3],
                                       format_func=lambda x: ["Electronic Check","Mailed Check","Bank Transfer","Credit Card"][x])

    with col3:
        senior         = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        partner        = st.selectbox("Has Partner", [0, 1],    format_func=lambda x: "No" if x == 0 else "Yes")
        paperless      = st.selectbox("Paperless Billing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):

        # Build drift features from inputs
        engagement_score = tenure / (monthly_charge + 1)
        charge_pressure  = monthly_charge / (total_charges + 1)
        loyalty_index    = tenure * services_used
        drift_risk_score = monthly_charge / (services_used + 1)

        # Build a sample row matching training feature count
        sample = np.zeros(X.shape[1])
        col_names = list(X.columns)

        def set_feat(name, val):
            if name in col_names:
                sample[col_names.index(name)] = val

        set_feat('tenure',           tenure)
        set_feat('MonthlyCharges',   monthly_charge)
        set_feat('TotalCharges',     total_charges)
        set_feat('services_used',    services_used)
        set_feat('Contract',         contract)
        set_feat('PaymentMethod',    payment_method)
        set_feat('SeniorCitizen',    senior)
        set_feat('Partner',          partner)
        set_feat('PaperlessBilling', paperless)
        set_feat('engagement_score', engagement_score)
        set_feat('charge_pressure',  charge_pressure)
        set_feat('loyalty_index',    loyalty_index)
        set_feat('drift_risk_score', drift_risk_score)

        sample_scaled = scaler.transform(sample.reshape(1, -1))
        prob      = model.predict_proba(sample_scaled)[0][1]
        pred      = model.predict(sample_scaled)[0]
        prob_pct  = round(prob * 100, 1)

        st.markdown("---")
        st.subheader("Prediction Result")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churn Probability", f"{prob_pct}%")
        c2.metric("Engagement Score",  round(engagement_score, 3))
        c3.metric("Drift Risk Score",  round(drift_risk_score, 3))
        c4.metric("Loyalty Index",     round(loyalty_index, 1))

        if prob > 0.7:
            st.error(f"🚨 HIGH RISK ({prob_pct}%) — Immediate intervention recommended. Send a retention offer NOW.")
        elif prob > 0.4:
            st.warning(f"⚠️ MEDIUM RISK ({prob_pct}%) — Monitor closely. Consider a proactive check-in or discount.")
        else:
            st.success(f"✅ LOW RISK ({prob_pct}%) — Customer appears stable and engaged.")

        # Drift gauge chart
        fig4, ax4 = plt.subplots(figsize=(8, 2))
        fig4.patch.set_facecolor('#0f0f0f')
        ax4.set_facecolor('#1a1a1a')
        ax4.barh(['Churn Risk'], [prob_pct], color='#f44336' if prob > 0.7 else '#ffa500' if prob > 0.4 else '#4caf50', height=0.4)
        ax4.barh(['Churn Risk'], [100], color='#333', height=0.4, zorder=0)
        ax4.set_xlim(0, 100)
        ax4.set_xlabel("Risk %", color='white')
        ax4.tick_params(colors='white')
        for spine in ax4.spines.values():
            spine.set_edgecolor('#444')
        ax4.set_title(f"Risk Gauge: {prob_pct}%", color='white')
        plt.tight_layout()
        st.pyplot(fig4)
