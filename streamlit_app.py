import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("NASA Exoplanet Classifier")

uploaded_file = st.file_uploader("Upload KOI CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing زي ما عملت
    df = df.dropna(axis=1, how='all')
    drop_cols = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname']
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    df['koi_disposition'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

    X = df.drop(columns=['koi_disposition', 'koi_pdisposition'])
    y = df['koi_disposition']

    X = X.fillna(X.mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)

    st.subheader("Test Accuracy")
    acc = accuracy_score(y_test, y_pred)
    st.write(acc)

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(pd.DataFrame(report).transpose())

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0,1], yticklabels=[0,1], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Feature Importance
    st.subheader("Top 15 Feature Importances")
    importances = rf.feature_importances_
    features = X.columns
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)[:15]
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, ax=ax2)
    st.pyplot(fig2)

    # Show predictions on X_test
    st.subheader("Sample Predictions")
    sample_df = pd.DataFrame(X_test, columns=X.columns)
    sample_df['prediction'] = y_pred
    st.dataframe(sample_df.head(20))
