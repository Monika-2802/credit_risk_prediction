import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import io

st.set_page_config(layout="wide")
st.title("Credit Risk Prediction Dashboard")
st.markdown("Use this tool to predict if a loan applicant is a **Good** or **Bad** credit risk using the German Credit dataset.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("german_credit_data.csv") 

df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df['Saving accounts'].fillna('no_info', inplace=True)
df['Checking account'].fillna('no_info', inplace=True)

cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
df_encoded['Credit per month'] = df_encoded['Credit amount'] / df_encoded['Duration']


df_encoded['Credit_Risk'] = np.where(
    ((df['Checking account'].isin(['little', 'no_info'])) &
     (df['Saving accounts'].isin(['little', 'no_info'])) &
     (df['Credit amount'] > df['Credit amount'].median()) &
     (df['Duration'] > df['Duration'].median())),
    0, 1
)


X = df_encoded.drop(columns=['Credit_Risk'])
y = df_encoded['Credit_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

st.subheader(" Model Performance")
st.text("Classification Report")
st.code(classification_report(y_test, y_pred))

st.text("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

st.subheader(" Feature Importance")
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig2, ax2 = plt.subplots()
feat_imp.plot(kind='barh', ax=ax2)
ax2.set_title("Top 10 Influential Features")
ax2.invert_yaxis()
st.pyplot(fig2)

st.subheader("Bulk Predictions for All Applicants")

df_bulk = pd.read_csv(uploaded_file) if uploaded_file else df.copy()
df_bulk.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
df_bulk['Saving accounts'].fillna('no_info', inplace=True)
df_bulk['Checking account'].fillna('no_info', inplace=True)
df_encoded_bulk = pd.get_dummies(df_bulk, columns=cat_cols, drop_first=True)
df_encoded_bulk['Credit per month'] = df_encoded_bulk['Credit amount'] / df_encoded_bulk['Duration']

for col in X.columns:
    if col not in df_encoded_bulk:
        df_encoded_bulk[col] = 0
df_encoded_bulk = df_encoded_bulk[X.columns]

df_scaled = scaler.transform(df_encoded_bulk)
df_bulk['Predicted Credit Risk'] = np.where(model.predict(df_scaled) == 1, 'Good', 'Bad')

st.dataframe(df_bulk[['Credit amount', 'Duration', 'Checking account', 'Saving accounts', 'Predicted Credit Risk']])

csv_download = df_bulk.to_csv(index=False).encode('utf-8')
st.download_button("Download Predictions", data=csv_download, file_name="credit_risk_predictions.csv", mime="text/csv")

st.subheader(" Predict Credit Risk for a New Applicant")
with st.form("manual_input"):
    credit_amount = st.number_input("Credit Amount", min_value=100, value=1500)
    duration = st.slider("Duration (months)", 6, 60, 24)
    credit_per_month = credit_amount / duration
    checking_no_info = st.checkbox("Checking account: no_info", value=True)
    saving_no_info = st.checkbox("Saving account: no_info", value=True)

    inputs = {
        'Age': 35,
        'Job': 2,
        'Credit amount': credit_amount,
        'Duration': duration,
        'Credit per month': credit_per_month,
        'Checking account_no_info': int(checking_no_info),
        'Saving accounts_no_info': int(saving_no_info),
    }

    for col in X.columns:
        if col not in inputs:
            inputs[col] = 0

    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df[X.columns])

    if st.form_submit_button("Predict"):
        prediction = model.predict(input_scaled)[0]
        risk = "Good" if prediction == 1 else "Bad"
        st.success(f"Predicted Credit Risk: **{risk}**")
