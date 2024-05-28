import streamlit as st

def introduction_page():
    st.title("Heart Disease Prediction")

    st.subheader("Why Predict Heart Disease?")
    st.write("""
    Heart disease is one of the leading causes of death globally. Early prediction and intervention can significantly improve patient outcomes and reduce mortality rates.
    Predicting heart disease allows for timely medical interventions, lifestyle changes, and continuous monitoring, which can prevent severe complications and save lives.
    According to the World Health Organization, cardiovascular diseases are responsible for approximately 31% of all global deaths. 
    More information can be found [here](https://ourworldindata.org/cardiovascular-diseases).
    """)

    st.subheader("Machine Learning Models")
    st.write("""
    In this section, I will showcase various machine learning models to predict heart disease based on clinical features. 
    These models include:

    - **Decision Trees**
    - **Pruned Decision Trees**
    - **Random Forests**
    - **Grid Search Optimized Random Forests**
    - **XGBoost**
    - **Grid Search Optimized XGBoost**

    I will evaluate the models using the following metrics:

    - **AUC Curve**: AUC-ROC is a important metric for understanding the performance of a classification model. It represents the ability of the model to distinguish between classes.
    - **Confusion Matrix**: This tool provides a detailed breakdown of the model's performance by showing the TP, TN, FP, and FN. It is essential for me to understand the types of errors the model is making and for further refinement.

    By using these models and evaluation metrics, I aim to demonstrate my machine learning skills and the importance of hyperparameter tuning in building effective predictive models.
    """)

if __name__ == "__main__":
    introduction_page()