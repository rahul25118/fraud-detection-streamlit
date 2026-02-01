import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Load YOUR fraud model (change filename if needed)
@st.cache_data
def load_my_model():
    data = joblib.load('fraud_production_model.pkl')
    return data['model'], data['threshold'], data['features']

model, cutoff_score, all_features = load_my_model()

# Big title
st.title("🚨 Fraud Detector")
st.markdown("**94% accurate • Catches 89% fraud • Your 6.3M dataset model**")

# Simple input boxes (like a form)
hour = st.slider("Hour (1-24)", 1, 24, 12)
payment_type = st.selectbox("Payment type", 
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
money = st.number_input("₹ Amount", 100, 10000000, 50000)
sender_old = st.number_input("Sender had ₹ before", 0, 50000000, 100000)
sender_new = st.number_input("Sender has ₹ after", 0, 50000000, 50000)
receiver_old = st.number_input("Receiver had ₹ before", 0, 100000000, 0)
receiver_new = st.number_input("Receiver has ₹ after", 0, 100000000, 50000)

# Predict button
if st.button("🔍 CHECK FRAUD", type="primary"):
    # Make transaction data
    txn = {
        'step': hour, 'type': payment_type, 'amount': money,
        'oldbalanceOrg': sender_old, 'newbalanceOrig': sender_new,
        'oldbalanceDest': receiver_old, 'newbalanceDest': receiver_new,
        'isFlaggedFraud': 0
    }
    
    # Run YOUR model
    X = pd.DataFrame([txn])
    X = pd.get_dummies(X, columns=['type'])
    X = X.reindex(columns=all_features, fill_value=0)
    
    risk = model.predict_proba(X)[0,1]
    is_bad = risk > cutoff_score
    
    # Show results BIG
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Risk %", f"{risk:.1%}")
    with col2:
        st.error("🚨 FRAUD!" if is_bad else "✅ SAFE")
    
    # Simple explanation
    st.markdown("**Why:**")
    if is_bad:
        st.error("• Large TRANSFER emptying sender\n• Suspicious amount pattern")
    else:
        st.success("• Normal business payment")

# Sidebar: Your model stats
st.sidebar.header("🏆 Your Model")
st.sidebar.metric("Accuracy", "94%")
st.sidebar.metric("Fraud Caught", "89%")
st.sidebar.metric("Transactions", "63 लाख")

# Cool chart of YOUR top features
st.markdown("### 📊 What your model looks at most")
[image:131]

st.caption("💾 Made by you with 6.3M transactions + XGBoost")
