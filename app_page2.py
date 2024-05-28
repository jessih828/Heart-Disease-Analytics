import streamlit as st
import pandas as pd
import pickle

# Function to load your trained model
def load_model():
    with open('heart_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def prob(prediction_proba):
    for prob in prediction_proba:
        if prob < 0.5:
            return 'low'
        elif prob >= 0.5 and prob < 0.75:
            return 'medium'
        else:
            return 'high'

def heart_disease_prediction_page():
    st.title("Heart Disease Prediction")
    st.markdown("""
    ### Welcome to the Heart Disease Prediction App
    This app uses machine learning to predict the likelihood of a heart Disease based on user input parameters. 
    Please provide the following details to get your prediction.
    """)

    st.header('Input Parameters')
    age = st.slider('Age', 29, 77, 50)
    sex = st.selectbox('Sex', ('Male', 'Female'))
    ChestPainType = st.selectbox('Chest Pain Type', 
                                 ('Typical angina (TA)', 
                                  'Atypical angina (ATA)', 
                                  'Non-anginal pain (NAP)', 
                                  'Asymptomatic (No chest pain)'))
    RestingBP = st.slider('Resting Blood Pressure', 94, 200, 120)
    chol = st.slider('Cholesterol Level', 126, 564, 240)
    FastingBS = st.selectbox('Fasting Blood Sugar > 120 mg/dl?', ('False', 'True'))
    restecg = st.selectbox('Resting Electrocardiogram Results', 
                           ('Normal', 
                            'Abnormal ST-T wave (ST)',
                            'Probable or definite left ventricular hypertrophy'))
    MaxHR = st.slider('Max Heart Rate Achieved', 60, 202, 145)
    ExerciseAngina = st.selectbox('Exercise-Induced Angina', ('No', 'Yes'))
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slp = st.selectbox('Slope of Peak Exercise ST Segment', ('Up', 'Flat', 'Down'))

    # Converting categorical variables to numerical values
    FastingBS_dict = {'False': 0, 'True': 1}

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex_M': [1 if sex == 'Male' else 0],
        'ChestPainType_ATA': [1 if ChestPainType == 'Atypical angina (ATA)' else 0],
        'ChestPainType_NAP': [1 if ChestPainType == 'Non-anginal pain (NAP)' else 0],
        'ChestPainType_TA': [1 if ChestPainType == 'Typical angina (TA)' else 0],
        'RestingBP': [RestingBP],
        'Cholesterol': [chol],
        'FastingBS': [FastingBS_dict[FastingBS]],
        'MaxHR': [MaxHR],
        'Oldpeak': [oldpeak],
        'RestingECG_Normal': [1 if restecg == 'Normal' else 0],
        'RestingECG_ST': [1 if restecg == 'Abnormal ST-T wave (ST)' else 0],
        'ExerciseAngina_Y': [1 if ExerciseAngina == 'Yes' else 0],
        'ST_Slope_Flat': [1 if slp == 'Flat' else 0],
        'ST_Slope_Up': [1 if slp == 'Up' else 0]
    })

    # Reorder the columns to match the order expected by the model
    input_data = input_data[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 
                             'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA', 
                             'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']]

    st.subheader('User Input Parameters')
    st.write(input_data)

        # Load the trained model
    model = load_model()

    if model:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        st.subheader('Heart Disease Prediction')
        probability = prediction_proba[0][1] * 100
        risk_level = prob([prediction_proba[0][1]])

        # Determine the risk level message and color
        if risk_level == "low":
            message = f"The probability of the patient having heart disease is {probability:.2f}% (low risk)"
            color = "#4FB783"
        elif risk_level == "medium":
            message = f"The probability of the patient having heart disease is {probability:.2f}% (medium risk)"
            color = "#005A9C"
        else:
            message = f"The probability of the patient having heart disease is {probability:.2f}% (high risk)"
            color = "#A91D3A"

        # Display the prediction with larger font and appropriate color
        st.markdown(f"<span style='color:{color}; font-weight:bold;'>{message}</span>", unsafe_allow_html=True)
    
if __name__ == "__main__":
    heart_disease_prediction_page()