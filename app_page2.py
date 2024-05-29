import streamlit as st
import pandas as pd
import plotly.subplots as sp
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv('heart_data.csv')

results = {
    'Decision Trees (Pruned)': {
        'conf_matrix': [[68, 14],[21, 81]],
        'accuracy': 0.80,
        'precision': 0.85,
        'recall': 0.79,
        'f1_score': 0.82,
        'specificity': 0.82,
        'roc_curve': {'fpr': [0.0, 0.036585365853658534, 0.04878048780487805, 0.06097560975609756, 0.06097560975609756, 0.0975609756097561, 0.10975609756097561, 0.10975609756097561, 0.13414634146341464, 0.17073170731707318, 0.17073170731707318, 0.2073170731707317, 0.2926829268292683, 0.3170731707317073, 0.3902439024390244, 0.4024390243902439, 0.4024390243902439, 0.4268292682926829, 0.47560975609756095, 0.5365853658536586, 1.0], 
                      'tpr': [0.0, 0.47058823529411764, 0.49019607843137253, 0.5196078431372549, 0.5392156862745098, 0.5686274509803921, 0.6666666666666666, 0.7058823529411765, 0.7647058823529411, 0.7941176470588235, 0.8529411764705882, 0.8627450980392157, 0.9117647058823529, 0.9215686274509803, 0.9215686274509803, 0.9215686274509803, 0.9411764705882353, 0.9509803921568627, 0.9509803921568627, 0.9607843137254902, 1.0], 
                      'roc_auc': 0.88},
        'cross_validated_roc_auc': "0.8970 ± 0.0235",
        'cross_validated_accuracy': "0.8225 ± 0.0071",
        'cross_validated_precision': "0.8188 ± 0.0362",
        'cross_validated_recall': "0.8171 ± 0.0658",
        'cross_validated_f1_score': "0.8595 ± 0.0245",
        'summary': "The pruned Decision Tree model demonstrates robust performance with an accuracy of 0.80 and a ROC AUC of 0.88 on the test set. The cross-validation results further confirm the model\'s consistency and reliability. These metrics suggest that the model generalizes well across different subsets of the data, maintaining a balance between precision and recall. However, to further enhance the model\'s performance and robustness, future steps could include exploring ensemble methods such as Random Forest or XGBoost, conducting additional hyperparameter tuning, or incorporating more diverse and comprehensive datasets. These actions could potentially lead to even better generalization and predictive accuracy."
    },

    'Random Forest': {
        'conf_matrix': [[67, 14],[12, 90]],
        'accuracy': 0.85,
        'precision': 0.85,
        'recall': 0.88,
        'f1_score': 0.86,
        'roc_curve': {'fpr': [0.0, 0.012195121951219513, 0.012195121951219513, 0.012195121951219513, 0.012195121951219513, 0.012195121951219513, 0.012195121951219513, 0.012195121951219513, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.04878048780487805, 0.04878048780487805, 0.06097560975609756, 0.06097560975609756, 0.06097560975609756, 0.07317073170731707, 0.07317073170731707, 0.08536585365853659, 0.08536585365853659, 0.0975609756097561, 0.0975609756097561, 0.10975609756097561, 0.12195121951219512, 0.12195121951219512, 0.13414634146341464, 0.14634146341463414, 0.14634146341463414, 0.18292682926829268, 0.18292682926829268, 0.1951219512195122, 0.1951219512195122, 0.24390243902439024, 0.2682926829268293, 0.2682926829268293, 0.3048780487804878, 0.32926829268292684, 0.3902439024390244, 0.4268292682926829, 0.45121951219512196, 0.4634146341463415, 0.4634146341463415, 0.4878048780487805, 0.524390243902439, 0.5365853658536586, 0.5609756097560976, 0.573170731707317, 0.5975609756097561, 0.6219512195121951, 0.6585365853658537, 0.7073170731707317, 0.7804878048780488, 0.8414634146341463, 0.9146341463414634, 1.0],
                      'tpr': [0.0, 0.0392156862745098, 0.058823529411764705, 0.09803921568627451, 0.11764705882352941, 0.19607843137254902, 0.21568627450980393, 0.29411764705882354, 0.3235294117647059, 0.3627450980392157, 0.37254901960784315, 0.4019607843137255, 0.4117647058823529, 0.43137254901960786, 0.4411764705882353, 0.45098039215686275, 0.47058823529411764, 0.5098039215686274, 0.5392156862745098, 0.5588235294117647, 0.5882352941176471, 0.6078431372549019, 0.6372549019607843, 0.6470588235294118, 0.6862745098039216, 0.696078431372549, 0.7156862745098039, 0.7352941176470589, 0.7450980392156863, 0.7647058823529411, 0.7647058823529411, 0.7745098039215687, 0.7745098039215687, 0.8137254901960784, 0.8137254901960784, 0.8235294117647058, 0.8529411764705882, 0.8529411764705882, 0.8725490196078431, 0.8823529411764706, 0.8823529411764706, 0.9019607843137255, 0.9019607843137255, 0.9215686274509803, 0.9215686274509803, 0.9313725490196079, 0.9607843137254902, 0.9607843137254902, 0.9607843137254902, 0.9607843137254902, 0.9607843137254902, 0.9803921568627451, 0.9901960784313726, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      'roc_auc': 0.93},
        'cross_validated_roc_auc': "0.9140 ± 0.0283",
        'cross_validated_accuracy': "0.8518 ± 0.0236",
        'cross_validated_precision': "0.8572 ± 0.0436",
        'cross_validated_recall': "0.8859 ± 0.0168",
        'cross_validated_f1_score': "0.8669 ± 0.0215",
        'summary': (
            "The Random Forest model demonstrates strong performance with an accuracy of 0.8587 and a ROC AUC of 0.9330 on the test set. "
            "The cross-validation results further confirm the model's consistency and reliability. These metrics suggest that the model "
            "generalizes well across different subsets of the data, maintaining a balance between precision and recall. However, to further "
            "enhance the model's performance and robustness, future steps could include exploring additional hyperparameter tuning, or "
            "incorporating more diverse and comprehensive datasets. These actions could potentially lead to even better generalization "
            "and predictive accuracy.")
    },


    'Random Forest (Grid Search)': {
        'conf_matrix': [[68, 14],[12, 90]],
        'accuracy': 0.88,
        'precision': 0.88,
        'recall': 0.90,
        'f1_score': 0.89,
        'specificity': 0.85,
        'cross_validated_roc_auc': "0.9220 ± 0.0282",
        'cross_validated_accuracy': "0.8573 ± 0.0267",
        'cross_validated_precision': "0.8565 ± 0.0391",
        'cross_validated_recall': "0.8938 ± 0.0197",
        'cross_validated_f1_score': "0.8800 ± 0.0240",
        'roc_curve': {'fpr': [0.0, 0.0, 0.0, 0.0, 0.0, 0.012195121951219513, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.024390243902439025, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.036585365853658534, 0.04878048780487805, 0.04878048780487805, 0.07317073170731707, 0.07317073170731707, 0.08536585365853659, 0.08536585365853659, 0.0975609756097561, 0.0975609756097561, 0.0975609756097561, 0.10975609756097561, 0.12195121951219512, 0.14634146341463414, 0.14634146341463414, 0.14634146341463414, 0.18292682926829268, 0.18292682926829268, 0.1951219512195122, 0.1951219512195122, 0.2073170731707317, 0.2073170731707317, 0.21951219512195122, 0.24390243902439024, 0.34146341463414637, 0.3780487804878049, 0.3902439024390244, 0.3902439024390244, 0.4268292682926829, 0.45121951219512196, 0.47560975609756095, 0.5, 0.5121951219512195, 0.5121951219512195, 0.5365853658536586, 0.5609756097560976, 0.573170731707317, 0.5975609756097561, 0.7073170731707317, 0.7195121951219512, 0.7682926829268293, 0.7804878048780488, 0.9024390243902439, 0.926829268292683, 1.0], 
                      'tpr': [0.0, 0.00980392156862745, 0.0196078431372549, 0.08823529411764706, 0.12745098039215685, 0.14705882352941177, 0.1568627450980392, 0.18627450980392157, 0.19607843137254902, 0.24509803921568626, 0.27450980392156865, 0.29411764705882354, 0.3235294117647059, 0.3431372549019608, 0.37254901960784315, 0.39215686274509803, 0.4411764705882353, 0.4803921568627451, 0.5196078431372549, 0.5294117647058824, 0.5686274509803921, 0.5980392156862745, 0.6176470588235294, 0.6568627450980392, 0.6862745098039216, 0.7156862745098039, 0.7254901960784313, 0.7549019607843137, 0.7549019607843137, 0.7843137254901961, 0.7843137254901961, 0.7941176470588235, 0.803921568627451, 0.8137254901960784, 0.8333333333333334, 0.8431372549019608, 0.8431372549019608, 0.8431372549019608, 0.8823529411764706, 0.9019607843137255, 0.9019607843137255, 0.9215686274509803, 0.9215686274509803, 0.9313725490196079, 0.9313725490196079, 0.9411764705882353, 0.9411764705882353, 0.9411764705882353, 0.9411764705882353, 0.9607843137254902, 0.9705882352941176, 0.9803921568627451, 0.9803921568627451, 0.9803921568627451, 0.9803921568627451, 0.9803921568627451, 0.9803921568627451, 0.9901960784313726, 0.9901960784313726, 0.9901960784313726, 0.9901960784313726, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                      'roc_auc': 0.93},
        'summary': (
            "The Random Forest model with Grid Search demonstrates strong performance with an accuracy of 0.8804 and a ROC AUC of 0.9314 on the test set. "
            "The cross-validation results further confirm the model's consistency and reliability, with a mean cross-validated ROC AUC of 0.9220 ± 0.0282, "
            "accuracy of 0.8573 ± 0.0267, precision of 0.8565 ± 0.0391, recall of 0.8938 ± 0.0197, and F1 score of 0.8800 ± 0.0240. "
            "These metrics suggest that the model generalizes well across different subsets of the data, maintaining a balance between precision and recall. "
            "However, to further enhance the model's performance and robustness, future steps could include exploring additional hyperparameter tuning or "
            "incorporating more diverse and comprehensive datasets. These actions could potentially lead to even better generalization and predictive accuracy."
            )

    },

    'XGBClassifier': {
        'conf_matrix': [[71, 11],[10, 92]],
        'accuracy': 0.88,
        'precision': 0.89,
        'recall': 0.90,
        'f1_score': 0.89,
        'specificity': 0.86,
        'roc_auc': 0.93,
        'cross_validated_roc_auc': "0.9154 ± 0.0215",
        'cross_validated_accuracy': "0.8496 ± 0.0237",
        'cross_validated_precision': "0.8546 ± 0.0360",
        'cross_validated_recall': "0.8799 ± 0.0236",
        'cross_validated_f1_score': "0.8664 ± 0.0196",
        'roc_curve': {'fpr': [0.0, 0.0, 0.0, 0.012195121951219513, 0.012195121951219513, 0.024390243902439025, 0.024390243902439025, 0.04878048780487805, 0.04878048780487805, 0.06097560975609756, 0.06097560975609756, 0.07317073170731707, 0.07317073170731707, 0.0975609756097561, 0.0975609756097561, 0.10975609756097561, 0.10975609756097561, 0.13414634146341464, 0.13414634146341464, 0.14634146341463414, 0.14634146341463414, 0.2073170731707317, 0.2073170731707317, 0.21951219512195122, 0.21951219512195122, 0.23170731707317074, 0.23170731707317074, 0.25609756097560976, 0.25609756097560976, 0.2682926829268293, 0.2682926829268293, 0.2804878048780488, 0.2804878048780488, 0.3048780487804878, 0.3048780487804878, 0.5121951219512195, 0.5121951219512195, 0.9512195121951219, 0.9512195121951219, 1.0],
                      'tpr': [0.0, 0.00980392156862745, 0.13725490196078433, 0.13725490196078433, 0.4117647058823529, 0.4117647058823529, 0.5686274509803921, 0.5686274509803921, 0.7745098039215687, 0.7745098039215687, 0.8235294117647058, 0.8235294117647058, 0.8627450980392157, 0.8627450980392157, 0.8725490196078431, 0.8725490196078431, 0.8921568627450981, 0.8921568627450981, 0.9019607843137255, 0.9019607843137255, 0.9117647058823529, 0.9117647058823529, 0.9215686274509803, 0.9215686274509803, 0.9313725490196079, 0.9313725490196079, 0.9411764705882353, 0.9411764705882353, 0.9509803921568627, 0.9509803921568627, 0.9607843137254902, 0.9607843137254902, 0.9705882352941176, 0.9705882352941176, 0.9803921568627451, 0.9803921568627451, 0.9901960784313726, 0.9901960784313726, 1.0, 1.0],
                      'roc_auc': 0.939},
        'summary': (
            "The XGBoost model demonstrates strong performance with an accuracy of 0.88 and a ROC AUC of 0.9471 on the test set. "
            "The cross-validation results further confirm the model's consistency and reliability, with a mean cross-validated ROC AUC of 0.9219 ± 0.0282, "
            "accuracy of 0.8573 ± 0.0267, precision of 0.8565 ± 0.0391, recall of 0.8938 ± 0.0197, and F1 score of 0.8800 ± 0.0240. "
            "These metrics suggest that the model generalizes well across different subsets of the data, maintaining a balance between precision and recall. "
            "However, to further enhance the model's performance and robustness, future steps could include exploring additional hyperparameter tuning or "
            "incorporating more diverse and comprehensive datasets. These actions could potentially lead to even better generalization and predictive accuracy."
        )
    },

    'XGBClassifier (Grid Search)': {
        'conf_matrix': [[70, 12],[11, 91]],
        'accuracy': 0.88,
        'precision': 0.88,
        'recall': 0.88,
        'f1_score': 0.89,
        'specificity': 0.93,
        'cross_validated_roc_auc': "0.929955348476024 ± 0.020801285088247048",
        'cross_validated_accuracy': "0.8790864813494892 ± 0.016648261784001597",
        'cross_validated_precision': "0.8799233509929671 ± 0.03295534918290866",
        'cross_validated_recall': "0.907610172781984 ± 0.028561606424072978",
        'cross_validated_f1_score': "0.8926840718133123 ± 0.013538822096832978",
        'roc_curve': {'fpr': [0.0, 0.0, 0.0, 0.012195121951219513, 0.012195121951219513, 0.024390243902439025, 0.024390243902439025, 0.036585365853658534, 0.036585365853658534, 0.04878048780487805, 0.04878048780487805, 0.06097560975609756, 0.06097560975609756, 0.07317073170731707, 0.07317073170731707, 0.08536585365853659, 0.08536585365853659, 0.0975609756097561, 0.0975609756097561, 0.14634146341463414, 0.14634146341463414, 0.23170731707317074, 0.23170731707317074, 0.24390243902439024, 0.24390243902439024, 0.2682926829268293, 0.2682926829268293, 0.5365853658536586, 0.5365853658536586, 0.5853658536585366, 0.5853658536585366, 0.6097560975609756, 0.6097560975609756, 1.0], 
                      'tpr': [0.0, 0.00980392156862745, 0.029411764705882353, 0.029411764705882353, 0.46078431372549017, 0.46078431372549017, 0.6078431372549019, 0.6078431372549019, 0.6176470588235294, 0.6176470588235294, 0.6666666666666666, 0.6666666666666666, 0.7941176470588235, 0.7941176470588235, 0.803921568627451, 0.803921568627451, 0.8137254901960784, 0.8137254901960784, 0.8921568627450981, 0.8921568627450981, 0.9215686274509803, 0.9215686274509803, 0.9411764705882353, 0.9411764705882353, 0.9509803921568627, 0.9509803921568627, 0.9705882352941176, 0.9705882352941176, 0.9803921568627451, 0.9803921568627451, 0.9901960784313726, 0.9901960784313726, 1.0, 1.0], 
                      'roc_auc': 0.937},
        'summary': (" ")
    }
}


def about_the_dataset():
    st.title("Exploring ML Models for Heart Disease Prediction")
    st.subheader("Data Exploration")

    # Summary and detailed data explanation
    with st.expander("Data Explanation"):
        st.write("""
        - **Age**: Age of the patient
        - **Sex**: Sex of the patient (M = Male, F = Female)
        - **ExerciseAngina**: Exercise-induced angina - Whether the patient experiences chest pain induced by exercise (1 = yes; 0 = no)
        - **ChestPainType**:
            - Value 1: Typical angina (standard chest pain related to the heart)
            - Value 2: Atypical angina (chest pain not typically associated with the heart)
            - Value 3: Non-anginal pain (chest pain not related to the heart)
            - Value 4: Asymptomatic (no chest pain)
        - **RestingBP**: Resting blood pressure (in mm Hg)
        - **Cholesterol**: Cholesterol in mg/dl fetched via BMI sensor
        - **FastingBS**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        - **RestingECG**: Resting electrocardiographic results
            - Value 0: Normal
            - Value 1: Having ST-T wave abnormality (showing issues with heart waves)
            - Value 2: Showing probable or definite left ventricular hypertrophy by Estes' criteria (Results indicating possible or definite thickening of the heart muscle)
        - **MaxHeartRate**: Maximum heart rate the patient achieves during an exercise test
        """)
    
    with st.expander("Statistical Summary"):
        st.write(df.describe())

    with st.expander("Data Visualization"):
        color_discrete_map = {0: '#FA7070', 1: '#72A0C1'}

        fig = sp.make_subplots(rows=3, cols=3, subplot_titles=[
            'Age Distribution', 'Sex Distribution', 'Exercise Induced Angina',
            'Chest Pain Type', 'Resting Blood Pressure by Target',
            'Fasting Blood Sugar', 'Resting Electrocardiographic Results'
        ])

        # Check column names are correct and adjust if necessary
        expected_columns = ['Age', 'Sex', 'ExerciseAngina', 'ChestPainType', 'RestingBP', 'FastingBS', 'RestingECG', 'HeartDisease']
        if not all(column in df.columns for column in expected_columns):
            st.error("Error: One or more expected columns are missing in the dataset.")
            st.stop()

        # Add Age Histogram
        age_hist = px.histogram(df, x='Age', color='HeartDisease', color_discrete_map=color_discrete_map, marginal="box")
        for trace in age_hist.data:
            fig.add_trace(trace, row=1, col=1)

        # Add Sex Count Plot
        sex_hist = px.histogram(df, x='Sex', color='HeartDisease', color_discrete_map=color_discrete_map)
        for trace in sex_hist.data:
            fig.add_trace(trace, row=1, col=2)

        # Add Exercise Induced Angina Count Plot
        exng_hist = px.histogram(df, x='ExerciseAngina', color='HeartDisease', color_discrete_map=color_discrete_map)
        for trace in exng_hist.data:
            fig.add_trace(trace, row=1, col=3)

        # Add Chest Pain Type Count Plot
        cp_hist = px.histogram(df, x='ChestPainType', color='HeartDisease', color_discrete_map=color_discrete_map)
        for trace in cp_hist.data:
            fig.add_trace(trace, row=2, col=1)

        # Add Resting Blood Pressure Box Plot by Target
        trtbps_box = px.box(df, x='HeartDisease', y='RestingBP', color='HeartDisease', color_discrete_map=color_discrete_map)
        for trace in trtbps_box.data:
            fig.add_trace(trace, row=2, col=2)

        # Add Fasting Blood Sugar Count Plot
        fbs_hist = px.histogram(df, x='FastingBS', color='HeartDisease', color_discrete_map=color_discrete_map)
        for trace in fbs_hist.data:
            fig.add_trace(trace, row=2, col=3)

        # Add Resting Electrocardiographic Results Count Plot
        restecg_hist = px.histogram(df, x='RestingECG', color='HeartDisease', color_discrete_map=color_discrete_map)
        for trace in restecg_hist.data:
            fig.add_trace(trace, row=3, col=1)

        # Update layout for the subplot figure
        fig.update_layout(height=900, width=900, title_text="Heart Disease Data Visualization", showlegend=False)

        # Display the plot
        st.plotly_chart(fig)

    st.subheader("Machine Learning Models Analysis")

    st.write("""
    In this section, I explore different machine learning models to predict heart disease based on clinical features. 
    I have used models such as decision trees, random forests, XGBoost, and their optimized versions using grid search.
    You can compare the performance of these models based on accuracy, ROC curves, and confusion matrices, and by clicking the "Use Grid Search" bottom, you can see the optimized version of the model.
    """)

    # Interactive Model Selection
    st.write("#### Models Used")
    model_choice = st.selectbox("Select a model", list(results.keys()))
    # show the model accuracy
    st.write(f"**Model Accuracy**: {results[model_choice]['accuracy']}")

    st.write("click to show: ")

    # Confusion Matrix
    if st.checkbox('Show Confusion Matrix'):
        conf_matrix = results[model_choice]['conf_matrix']
        x = ['Positive (1)', 'Negative (0)']
        y = ['Positive (1)', 'Negative (0)']

        z_text = [['TP: {}'.format(conf_matrix[0][0]), 'FP: {}'.format(conf_matrix[0][1])], 
                ['FN: {}'.format(conf_matrix[1][0]), 'TN: {}'.format(conf_matrix[1][1])]]

        conf_fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=x,
            y=y,
            colorscale='Teal',
            showscale=True,
            text=z_text,
            texttemplate="%{text}",
            hoverinfo="skip"
        ))

        conf_fig.update_layout(title=f'Confusion Matrix for {model_choice}', xaxis_title='Actual Values', yaxis_title='Predicted Values')
        st.plotly_chart(conf_fig)

    # show matrix values
    if st.checkbox('Show Model Metrics'):
        st.write(f"**Accuracy**: {results[model_choice]['accuracy']}")
        st.write(f"**Precision**: {results[model_choice]['precision']}")
        st.write(f"**Recall**: {results[model_choice]['recall']}")
        st.write(f"**F1 Score**: {results[model_choice]['f1_score']}")
        st.write(f"**Specificity**: {results[model_choice]['specificity']}")

    # ROC Curve
    if st.checkbox('Show ROC Curve'):
        roc_data = results[model_choice]['roc_curve']
        
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=roc_data['fpr'], y=roc_data['tpr'], mode='lines', name=f'ROC curve (area = {roc_data["roc_auc"]:.2f})'))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='No Skill', line=dict(dash='dash')))
        roc_fig.update_layout(title=f'ROC Curve for {model_choice}', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        
        st.plotly_chart(roc_fig)

    # show Cross-Validation Results
    if st.checkbox('Show Cross-Validation Results'):
        st.write(f"**Cross-Validated ROC AUC**: {results[model_choice]['cross_validated_roc_auc']}")
        st.write(f"**Cross-Validated Accuracy**: {results[model_choice]['cross_validated_accuracy']}")
        st.write(f"**Cross-Validated Precision**: {results[model_choice]['cross_validated_precision']}")
        st.write(f"**Cross-Validated Recall**: {results[model_choice]['cross_validated_recall']}")
        st.write(f"**Cross-Validated F1 Score**: {results[model_choice]['cross_validated_f1_score']}")

    # sumamry of the model
    if st.checkbox('Show Model Summary'):
        st.write(f"{results[model_choice]['summary']}")

    
if __name__ == "__main__":
    about_the_dataset()