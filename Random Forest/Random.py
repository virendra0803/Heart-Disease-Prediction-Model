import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score

# Load the dataset (Heart Disease dataset from UCI)

def load_data():
    # url = 'https://raw.githubusercontent.com/plotly/datasets/master/heart.csv'
    data = pd.read_csv("C:/Users/hp/Documents/new-heart-model-demo/final model done/Random Forest/heart1.csv")
    return data

# Train the model and return the trained model along with test data
def train_model(df):
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, y_test, y_pred, X_test, scaler

# User Input for Prediction
def user_input_features():
    st.sidebar.header('Input Features for Prediction')
    age = st.sidebar.slider('Age', 29, 77, 54)
    sex = st.sidebar.selectbox('Sex(Male : 1 / Female : 0)', (0, 1))
    cp = st.sidebar.selectbox('Chest Pain Type ( Typical: 0 / Atypical: 1 / Non-Anginal Pain: 2 / Asymptomatic: 3)', (0, 1, 2, 3))
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 94, 200, 130)
    chol = st.sidebar.slider('Serum Cholesterol (chol)', 126, 564, 246)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar(> 120 mg/dl : 1 / < 120 mg/dl : 0)', (0, 1))
    restecg = st.sidebar.selectbox('Resting Electrocardiographic (Normal: 0 / ST-T Wave Abnormality: 1 / Ventricular Abnormality: 2)', (0, 1, 2))
    thalach = st.sidebar.slider('Maximum Heart Rate (thalach)', 71, 202, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina (NO : 0 / Yes : 1)', (0, 1))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment (Up : 0 / Flat : 1 / Down : 2)', (0, 1, 2))
    # ca = st.sidebar.slider('Number of Major Vessels Colored by Fluoroscopy (ca)', 0, 3, 0)
    # thal = st.sidebar.selectbox('Thalassemia (thal)', (0, 1, 2, 3))

    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            # 'ca': ca,
            # 'thal': thal 
            }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Main Function
def main():
    st.title("Heart Disease Prediction")
    
    # Load and preprocess data
    df = load_data()

    # Model training and evaluation
    model, accuracy, y_test, y_pred, X_test, scaler = train_model(df)

    # User Input for Prediction
    input_df = user_input_features()

    # Predict based on user input
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    st.subheader("Prediction Output")
    if prediction[0] == 1:
        st.write("**Heart Disease Predicted**")
    else:
        st.write("**No Heart Disease Predicted**")

    # Visualize the process of calculating the prediction
    st.subheader("Prediction Probability Visualization")
    labels = ['No Heart Disease', 'Heart Disease']
    proba = prediction_proba[0]
    plt.figure(figsize=(6, 4))
    plt.barh(labels, proba, color=['green', 'red'])
    plt.xlim([0, 1])
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    st.pyplot(plt)

    # Confusion Matrix
    st.subheader(f"Model Accuracy: {accuracy:.2f}")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    st.pyplot(plt)

    # ROC Curve
    st.subheader("ROC Curve")
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    st.pyplot(plt)

    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    st.pyplot(plt)

if __name__ == '__main__':
    main()
