import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Streamlit Config

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")


# File Upload

st.sidebar.header("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Training Data (CSV)", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Test Data (CSV/Excel)", type=["csv", "xlsx"])

if train_file and test_file:
    # Load datasets
    train_df = pd.read_csv(train_file)

    if test_file.name.endswith(".csv"):
        test_df = pd.read_csv(test_file)
    else:
        test_df = pd.read_excel(test_file)

    
    # Preprocessing
    
    for df in [train_df, test_df]:
        if "customerID" in df.columns:
            df.drop("customerID", axis=1, inplace=True)

    for df in [train_df, test_df]:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df.fillna(0, inplace=True)

    encoders = {}
    for col in train_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        encoders[col] = le
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col])

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    
    # Train Models
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    summary_data = []
    conf_matrices = {}

    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        # Extract hyperparameters as string
        hyperparams = {k: v for k, v in model.get_params().items() if not k.startswith('random_state')}
        hyperparams_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])

        # Epochs: For models that have n_estimators or max_iter
        if hasattr(model, "n_estimators"):
            epochs = model.n_estimators
        elif hasattr(model, "max_iter"):
            epochs = model.max_iter
        else:
            epochs = "N/A"

        summary_data.append({
            "Model": name,
            "Classifier": model.__class__.__name__,
            "Hyperparameters": hyperparams_str,
            "Epochs": epochs,
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{precision*100:.2f}%",
            "Recall": f"{recall*100:.2f}%",
            "F1 Score": f"{f1*100:.2f}%"
        })

        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    results_df = pd.DataFrame(summary_data)
    st.subheader("📈 Performance Summary")
    st.dataframe(results_df)

    # Save to Excel
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, index=False, sheet_name="Model_Results")
    st.download_button("📥 Download Results (Excel)", buffer.getvalue(),
                       file_name="model_performance_summary.xlsx")

    # Best model
    results_df["AccFloat"] = results_df["Accuracy"].str.rstrip("%").astype(float)
    best_model_name = results_df.loc[results_df["AccFloat"].idxmax()]["Model"]
    best_model = models[best_model_name]
    st.success(f"Best Model: **{best_model_name}**")

    # Confusion Matrix
    st.subheader("Confusion Matrix Viewer")
    choice = st.selectbox("Select Model", results_df["Model"].tolist())
    cm = conf_matrices[choice]
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {choice}")
    st.pyplot(fig)

    
    # Prediction Form
    
    st.subheader("🧑‍💻 Predict Churn for a Customer")

    with st.form("customer_form"):
        gender = st.selectbox("Gender", encoders["gender"].classes_)
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", encoders["MultipleLines"].classes_)
        internet_service = st.selectbox("Internet Service", encoders["InternetService"].classes_)
        online_security = st.selectbox("Online Security", encoders["OnlineSecurity"].classes_)
        online_backup = st.selectbox("Online Backup", encoders["OnlineBackup"].classes_)
        device_protection = st.selectbox("Device Protection", encoders["DeviceProtection"].classes_)
        tech_support = st.selectbox("Tech Support", encoders["TechSupport"].classes_)
        streaming_tv = st.selectbox("Streaming TV", encoders["StreamingTV"].classes_)
        streaming_movies = st.selectbox("Streaming Movies", encoders["StreamingMovies"].classes_)
        contract = st.selectbox("Contract", encoders["Contract"].classes_)
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", encoders["PaymentMethod"].classes_)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

        submit = st.form_submit_button("Predict")

    if submit:
        input_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [1 if senior == "Yes" else 0],
            "Partner": [1 if partner == "Yes" else 0],
            "Dependents": [1 if dependents == "Yes" else 0],
            "tenure": [tenure],
            "PhoneService": [1 if phone_service == "Yes" else 0],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [1 if paperless == "Yes" else 0],
            "PaymentMethod": [payment],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
        })

        # Encode using saved encoders
        for col in encoders:
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(
                    lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
                )
                input_data[col] = encoders[col].transform(input_data[col])

        # Scale
        input_scaled = scaler.transform(input_data)

        # Prediction
        pred = best_model.predict(input_scaled)[0]
        prob = best_model.predict_proba(input_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Customer likely to churn. Probability: {prob:.2%}")
        else:
            st.success(f"✅ Customer not likely to churn. Probability: {prob:.2%}")

else:
    st.info("Upload both Training and Test datasets to begin.")
