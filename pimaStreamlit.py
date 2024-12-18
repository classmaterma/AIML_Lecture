import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib as plt
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, precision_recall_curve,
                             roc_curve, auc
                             )




st.title("Pima Prediction")


uploaded_files = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_files is not None:
    data = pd.read_csv(uploaded_files)
    st.write("File Uploaded")
    
    if st.checkbox("Show Raw"):
        st.write(data)
        
    for col in data.columns[:-1]:
        if data[col].min() == 0 and col != "Pregnancies":
            if data[col].dtype == "int64":
                mean_value = int(data[col][data[col] != 0].mean())
            elif data[col].dtype == "float64":
                mean_value = data[col][data[col]!=0].mean()
                
            data[col] = data[col].replace(0, mean_value)
    
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.write("Processed Data (0 values replaced with mean)")
    st.write(data.describe())
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    if st.checkbox("Show Features"):
        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train, y_train)
        feature_importance = pd.DataFrame({
            "Feature" : X.columns,
            "Importance" : rf_model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
            
        
        st.write(feature_importance)
        
    selected_features = st.multiselect(
        "Select Features",
        options = list(X.columns),
        default = list(X.columns)
    )
    
    if selected_features:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        classifier_name = st.selectbox(
            "Select Classifier",
            ["Logistic Regression", "Random Forest", "Decision Tree"]
        )
        
        if classifier_name == "Logistic Regression":
            model = LogisticRegression(solver="liblinear")
        elif classifier_name == "Random Forest":
            model = RandomForestClassifier()
        elif classifier_name == "Decision Tree":
            model = DecisionTreeClassifier()
        
        model.fit(X_train_selected, y_train)
        y_scores = model.predict_proba(X_test_selected)[:, 1]

        y_pred = (y_scores > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        st.write (f"{classifier_name} Accuracy: {accuracy:.2f}")
        
        threshold = st.slider("Adjust", 0.0, 1.0, 0.5, 0.01)
        
        y_pred_threshold = (y_scores > threshold).astype(int)        
        accuracy_threshold = accuracy_score(y_test, y_pred_threshold)
        precision_threshold = precision_score(y_test, y_pred_threshold)
        recall_threshold = recall_score(y_test, y_pred_threshold)

        st.write(f"Performance with threshold {threshold:.2f}")
        st.write(f" - Accuracy: {accuracy_threshold:.2f}")
        st.write(f" - Precision: {precision_threshold:.2f}")
        st.write(f" - Recall: {recall_threshold:.2f}")

