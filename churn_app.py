import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# ── Load the model
model = pickle.load(open('gb_model.pkl', 'rb'))

# ── Page config 
st.set_page_config(page_title='Customer Churn Prediction', page_icon='📡', layout='centered')

st.title('📡 Customer Churn Prediction App')
st.markdown('Fill in the customer details below and click **Predict** to check churn risk.')
st.divider()

# ── Input section 
col1, col2 = st.columns(2)

with col1:
    st.subheader('👤 Demographics')
    gender          = st.selectbox('Gender', ('Male', 'Female'))
    senior_citizen  = st.selectbox('Senior Citizen', ('No', 'Yes'))
    partner         = st.selectbox('Has Partner', ('Yes', 'No'))
    dependents      = st.selectbox('Has Dependents', ('Yes', 'No'))

with col2:
    st.subheader('📋 Account Info')
    tenure          = st.number_input('Tenure (months)', min_value=0, max_value=72, value=12)
    contract        = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    paperless       = st.selectbox('Paperless Billing', ('Yes', 'No'))
    payment_method  = st.selectbox('Payment Method', (
                        'Electronic check', 'Mailed check',
                        'Bank transfer (automatic)', 'Credit card (automatic)'))

st.divider()
col3, col4 = st.columns(2)

with col3:
    st.subheader('📞 Services')
    phone_service   = st.selectbox('Phone Service', ('Yes', 'No'))
    multiple_lines  = st.selectbox('Multiple Lines',
                        ('No phone service', 'No', 'Yes'))
    internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    online_security = st.selectbox('Online Security', ('No internet service', 'No', 'Yes'))
    online_backup   = st.selectbox('Online Backup',   ('No internet service', 'No', 'Yes'))

with col4:
    st.subheader('🛡️ Add-ons')
    device_protection = st.selectbox('Device Protection', ('No internet service', 'No', 'Yes'))
    tech_support      = st.selectbox('Tech Support',      ('No internet service', 'No', 'Yes'))
    streaming_tv      = st.selectbox('Streaming TV',      ('No internet service', 'No', 'Yes'))
    streaming_movies  = st.selectbox('Streaming Movies',  ('No internet service', 'No', 'Yes'))

st.divider()
col5, col6 = st.columns(2)
with col5:
    monthly_charges = st.number_input('Monthly Charges ($)', min_value=0.0, max_value=200.0, value=65.0)
with col6:
    total_charges   = st.number_input('Total Charges ($)',   min_value=0.0, max_value=10000.0, value=780.0)

# ── Encoding (mirrors notebook exactly) 
def build_input():
    # Binary label-encoded columns
    gender_enc      = 1 if gender == 'Male' else 0          # Male=1, Female=0
    partner_enc     = 1 if partner == 'Yes' else 0
    dependents_enc  = 1 if dependents == 'Yes' else 0
    phone_enc       = 1 if phone_service == 'Yes' else 0
    paperless_enc   = 1 if paperless == 'Yes' else 0

    # One-hot for MultipleLines (drop_first=True → base = 'No')
    ml_no_phone = 1 if multiple_lines == 'No phone service' else 0
    ml_yes      = 1 if multiple_lines == 'Yes' else 0

    # One-hot for InternetService (drop_first → base = 'DSL')
    is_fiber = 1 if internet_service == 'Fiber optic' else 0
    is_no    = 1 if internet_service == 'No' else 0

    # One-hot for OnlineSecurity (drop_first → base = 'No')
    os_no_service = 1 if online_security == 'No internet service' else 0
    os_yes        = 1 if online_security == 'Yes' else 0

    # OnlineBackup
    ob_no_service = 1 if online_backup == 'No internet service' else 0
    ob_yes        = 1 if online_backup == 'Yes' else 0

    # DeviceProtection
    dp_no_service = 1 if device_protection == 'No internet service' else 0
    dp_yes        = 1 if device_protection == 'Yes' else 0

    # TechSupport
    ts_no_service = 1 if tech_support == 'No internet service' else 0
    ts_yes        = 1 if tech_support == 'Yes' else 0

    # StreamingTV
    stv_no_service = 1 if streaming_tv == 'No internet service' else 0
    stv_yes        = 1 if streaming_tv == 'Yes' else 0

    # StreamingMovies
    sm_no_service = 1 if streaming_movies == 'No internet service' else 0
    sm_yes        = 1 if streaming_movies == 'Yes' else 0

    # Contract (drop_first → base = 'Month-to-month')
    contract_one = 1 if contract == 'One year' else 0
    contract_two = 1 if contract == 'Two year' else 0

    # PaymentMethod (drop_first → base = 'Bank transfer (automatic)')
    pm_cc      = 1 if payment_method == 'Credit card (automatic)' else 0
    pm_echeck  = 1 if payment_method == 'Electronic check' else 0
    pm_mailed  = 1 if payment_method == 'Mailed check' else 0

    row = {
        'gender': gender_enc,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': partner_enc,
        'Dependents': dependents_enc,
        'tenure': tenure,
        'PhoneService': phone_enc,
        'PaperlessBilling': paperless_enc,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_No phone service': ml_no_phone,
        'MultipleLines_Yes': ml_yes,
        'InternetService_Fiber optic': is_fiber,
        'InternetService_No': is_no,
        'OnlineSecurity_No internet service': os_no_service,
        'OnlineSecurity_Yes': os_yes,
        'OnlineBackup_No internet service': ob_no_service,
        'OnlineBackup_Yes': ob_yes,
        'DeviceProtection_No internet service': dp_no_service,
        'DeviceProtection_Yes': dp_yes,
        'TechSupport_No internet service': ts_no_service,
        'TechSupport_Yes': ts_yes,
        'StreamingTV_No internet service': stv_no_service,
        'StreamingTV_Yes': stv_yes,
        'StreamingMovies_No internet service': sm_no_service,
        'StreamingMovies_Yes': sm_yes,
        'Contract_One year': contract_one,
        'Contract_Two year': contract_two,
        'PaymentMethod_Credit card (automatic)': pm_cc,
        'PaymentMethod_Electronic check': pm_echeck,
        'PaymentMethod_Mailed check': pm_mailed,
    }
    return pd.DataFrame([row])

# ── Predict button 
st.markdown('')
if st.button('🔍 Predict Churn', use_container_width=True):
    df_input = build_input()

    # Scale all features (mirrors notebook: StandardScaler on full X)
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_input), columns=df_input.columns)

    prediction  = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]

    churn_prob = round(probability[1] * 100, 2)

    st.divider()
    if prediction == 1:
        st.error(f'⚠️ **Churn Predicted** — This customer is likely to leave.')
    else:
        st.success(f'✅ **No Churn** — This customer is likely to stay.')

    st.metric(label='Churn Probability', value=f'{churn_prob}%')

    # Risk gauge
    if churn_prob >= 60:
        risk_label, risk_color = 'High Risk 🔴', 'red'
    elif churn_prob >= 30:
        risk_label, risk_color = 'Medium Risk 🟡', 'orange'
    else:
        risk_label, risk_color = 'Low Risk 🟢', 'green'

    st.markdown(f'**Risk Level:** :{risk_color}[{risk_label}]')

    st.progress(int(churn_prob))

    # Tip
    st.info(
        '💡 **Tip:** If churn risk is high, consider offering a long-term contract '
        'discount, bundling Tech Support & Online Security, or a loyalty reward.'
    )
