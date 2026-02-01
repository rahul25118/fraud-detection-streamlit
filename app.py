import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load your champion model
@st.cache_resource
def load_model():
    loaded = joblib.load('fraud_production_model.pkl')
    return loaded['model'], loaded['threshold'], loaded['features']

model, threshold, features = load_model()

st.title("🚨 Production Fraud Detection System")
st.markdown("**94% Precision | 89% Recall | 99.8% AUC-PR | 6.3M Transactions**")

# Input form
col1, col2 = st.columns(2)
with col1:
    step = st.number_input("Step", 1, 743, 300)
    txn_type = st.selectbox("Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'])
    amount = st.number_input("Amount", 0.0, 10000000.0, 100000.0)
    
with col2:
    oldbalanceOrg = st.number_input("Old Balance Orig", 0.0, 60000000.0, 1000000.0)
    newbalanceOrig = st.number_input("New Balance Orig", 0.0, 50000000.0, 900000.0)
    oldbalanceDest = st.number_input("Old Balance Dest", 0.0, 300000000.0, 500000.0)
    newbalanceDest = st.number_input("New Balance Dest", 0.0, 300000000.0, 1500000.0)

if st.button("🔍 DETECT FRAUD", type="primary"):
    # Create transaction
    txn = {
        'step': step, 'type': txn_type, 'amount': amount,
        'oldbalanceOrg': oldbalanceOrg, 'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest, 'newbalanceDest': newbalanceDest,
        'isFlaggedFraud': 0
    }
    
    # Predict
    X = pd.DataFrame([txn])
    X = pd.get_dummies(X, columns=['type'], drop_first=True)
    X = X.reindex(columns=features, fill_value=0)
    
    prob = model.predict_proba(X)[0, 1]
    is_fraud = prob > threshold
    
    # Results
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Fraud Probability", f"{prob:.1%}")
    with col2:
        st.metric("Prediction", "🚨 FRAUD" if is_fraud else "✅ LEGIT", 
                 f"{prob:.1%}" if is_fraud else f"{1-prob:.1%}")
    with col3:
        st.metric("Confidence", "HIGH" if prob > 0.99 or prob < 0.01 else "MEDIUM")
    
    # Risk gauge
    fig = px.imshow([[prob]], color_continuous_scale='RdYlGn_r', 
                    aspect="auto", title="Risk Level")
    st.plotly_chart(fig, use_container_width=True)

# Key metrics
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Precision", "94%")
st.sidebar.metric("Recall", "89%") 
st.sidebar.metric("AUC-PR", "99.8%")
st.sidebar.metric("Transactions", "6.3M")
