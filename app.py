import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

st.set_page_config(layout="wide", page_title="Customer Churn Prediction", page_icon="📊")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 12px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: #fff;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
try:
    with open('churn_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found! Please run train_model.py first.")
    st.stop()

# --- Load Data ---
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    st.error("❌ Data file not found! Make sure 'WA_Fn-UseC_-Telco-Customer-Churn.csv' exists.")
    st.stop()

# --- Title ---
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>📊 Customer Churn Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px; color:gray;'>Visualize data patterns and predict churn 🚀</p>", unsafe_allow_html=True)
st.markdown("---")

# --- EDA Section (Only Graphs) ---
st.subheader("📊 Explore Customer Data")

feature_options = ['Contract', 'InternetService', 'gender', 'PaymentMethod', 'TechSupport']
selected_feature = st.selectbox("Select a feature to see its relationship with Churn:", feature_options)

fig = px.bar(
    df.groupby([selected_feature, 'Churn']).size().reset_index(name='count'),
    x=selected_feature,
    y='count',
    color='Churn',
    barmode='group',
    text='count',
    color_discrete_map={'No': '#4a90e2', 'Yes': '#e74c3c'}
)

fig.update_traces(textposition='outside')
fig.update_layout(
    title=dict(text=f"Churn Distribution by {selected_feature}", x=0.5, xanchor="center", font=dict(size=20)),
    xaxis_title=selected_feature,
    yaxis_title="Number of Customers",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(size=14),
    bargap=0.25
)

st.plotly_chart(fig, use_container_width=True)

# --- Prediction Section ---
st.markdown("---")
# --- KPI Section ---
kpi1, kpi2, kpi3 = st.columns(3)

# Calculate KPIs
churn_rate = df[df['Churn'] == 'Yes'].shape[0] / df.shape[0] * 100
avg_monthly_charges = df['MonthlyCharges'].mean()
avg_tenure = df['tenure'].mean()

# Style and display KPIs
kpi1.markdown(f"""
<div style="background-color: #ffffff; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #dee2e6;">
    <h3 style="color: #e74c3c;">Overall Churn Rate</h3>
    <p style="font-size: 2.5rem; font-weight: bold; color: #2c3e50;">{churn_rate:.2f}%</p>
</div>
""", unsafe_allow_html=True)

kpi2.markdown(f"""
<div style="background-color: #ffffff; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #dee2e6;">
    <h3 style="color: #4a90e2;">Avg. Monthly Bill</h3>
    <p style="font-size: 2.5rem; font-weight: bold; color: #2c3e50;">${avg_monthly_charges:.2f}</p>
</div>
""", unsafe_allow_html=True)

kpi3.markdown(f"""
<div style="background-color: #ffffff; border-radius: 12px; padding: 1.5rem; text-align: center; border: 1px solid #dee2e6;">
    <h3 style="color: #4a90e2;">Avg. Tenure</h3>
    <p style="font-size: 2.5rem; font-weight: bold; color: #2c3e50;">{avg_tenure:.1f} mo.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True) # Add some space

st.markdown("## 🔮 Predict Customer Churn")
st.markdown("Fill in customer details to estimate churn probability.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("👤 Gender", ['Male', 'Female'])
    contract = st.selectbox("📄 Contract", ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox("🌐 Internet Service", ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox("💳 Payment Method", [
        'Electronic check', 'Mailed check',
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    partner = st.selectbox("💍 Has a partner?", ['Yes', 'No'])
    dependents = st.selectbox("👶 Has dependents?", ['Yes', 'No'])

with col2:
    tenure = st.slider("📅 Tenure (months)", 0, 72, 24)
    monthly_charges = st.number_input("💵 Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
    total_charges = st.number_input("💰 Total Charges ($)", min_value=0.0, max_value=10000.0, value=1000.0)
    senior_citizen = st.selectbox("👴 Senior Citizen?", [0, 1])
    tech_support = st.selectbox("🛠 Tech Support", ['Yes', 'No', 'No internet service'])
    online_security = st.selectbox("🔒 Online Security", ['Yes', 'No', 'No internet service'])

# --- Prediction ---
if st.button("🚀 Predict Churn"):
    partner_numeric = 1 if partner == 'Yes' else 0
    dependents_numeric = 1 if dependents == 'Yes' else 0

    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner_numeric,
        'Dependents': dependents_numeric,
        'tenure': tenure,
        'PhoneService': 1,
        'MultipleLines': 'No',
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': tech_support,
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': contract,
        'PaperlessBilling': 1,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])
    prediction_proba = pipeline.predict_proba(input_df)[0][1]

    st.subheader("📌 Prediction Result")
    progress_value = int(prediction_proba * 100)
    st.progress(progress_value)

    if prediction_proba > 0.5:
        st.error(f"🚨 High Churn Risk: {prediction_proba:.2%}")
        st.warning("💡 Offer loyalty discounts, better contracts, or proactive support.")
    else:
        st.success(f"✅ Low Churn Risk: {prediction_proba:.2%}")
        st.info("🙌 Focus on customer engagement and upselling opportunities.")
