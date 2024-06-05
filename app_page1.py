import streamlit as st

def introduction_page():
    st.title("Heart Disease Prediction")

    st.subheader("App Overview")
    st.write("""
    Welcome to the Heart Disease Prediction app! This app predicts the likelihood of a patient having heart disease based on clinical features.
    In this app, you can find the following sections:
             
    1. Data Exploration
        - Overview of the dataset (Here you can find the explanation on the clinical features used in the model)
        - Data visualization
             
    2. Machine Learning Models
        - Predictive models used to predict heart disease
        - Evaluation metrics used to assess model performance   
             
    3. Heart Disease Prediction
        - Input patient data to predict the likelihood of heart disease
        - Display the prediction results and risk level
             
    **By using these models and evaluation metrics, I aim to demonstrate my machine learning skills and the importance of hyperparameter tuning in building effective predictive models**
    """)

    st.subheader("Why Predict Heart Disease?")
    st.write("""
    Heart disease is one of the leading causes of death globally. Early prediction and intervention can significantly improve patient outcomes and reduce mortality rates.
    Predicting heart disease allows for timely medical interventions, lifestyle changes, and continuous monitoring, which can prevent severe complications and save lives.
    According to the World Health Organization, cardiovascular diseases are responsible for approximately 31% of all global deaths. 
    More information can be found [here](https://ourworldindata.org/cardiovascular-diseases).
    """)

if __name__ == "__main__":
    introduction_page()