import numpy as np
import pandas as pd
import streamlit as st

from classifier.knn import KNN
from sklearn.metrics import accuracy_score
from classifier.decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from eda.basic_statistics import BasicStatisticalAnalysis
from classifier.multilayerpercepteron import NeuralNetwork
from preprocess.convert_to_dataframe import ConvertDataset
from feature_engineering.feature_engineering import BasicFeatureEngineering


st.set_page_config(
    page_title='Scikit-Learn Streamlit App',
    page_icon='🔥',  
)

# Defining main function
def main():
    # Create a streamlit container that contain sidebar and two columns
    main_container = st.container()
    side_bar_container = st.container()
    main_page_container = st.container()
    dataframe_container = st.container()

    classification_methods_tuple = (
        'Decision Tree',
        'KNN',
        'Neural Network'        
    )

    with main_container:
        # Create a sidebar with a button to show/hide the sidebar
        with side_bar_container:
            st.sidebar.title('Scikit-learn Playground App')
            st.sidebar.markdown('---')

            # Create a sidebar that choose datasets that will be used in the app
            st.sidebar.subheader('Datasets')
            dataset_type = st.sidebar.selectbox(
                'Choose a dataset',
                ('Iris', 'Wine', 'Breast Cancer')
            )

            # Convert 'Breast Cancer' to breast_cancer
            if dataset_type == 'Breast Cancer':
                dataset_type = 'breast_cancer'

            # Call class that convert the dataset to dataframe
            dataset = ConvertDataset(dataset_type.lower())
            dataframe = dataset.convert_to_dataframe()
            st.sidebar.markdown('---')

            # Create a sidebar with two buttons
            st.sidebar.subheader('Process')
            radio_bt_app = st.sidebar.radio('Choose One', ('Exploratory Data Analysis', 'Machine Learning'))
            st.sidebar.markdown('---')

            # Create a sidebar with two buttons
            if radio_bt_app == 'Machine Learning':
                st.sidebar.subheader('Classifier')
                radio_ml_model = st.sidebar.radio('Choose One', classification_methods_tuple)
                st.sidebar.markdown('---')

        with main_page_container:
            st.markdown('Welcome to 🔥 Scikit-learn Playground App!. You can used this playground to explore and learn about machine learning algorithms. The algorithm is implemented by using Scikit-learn library.')
            if radio_bt_app == 'Exploratory Data Analysis':
                st.markdown('## Exploratory Data Analysis')
                st.markdown('---')
                st.markdown('### Data Description')
                bs = BasicStatisticalAnalysis(dataframe)
                bs.describe()
                st.markdown('### Correlation Matrix')
                bs.correlation()
                st.markdown('### Features Importances')     
                fi = BasicFeatureEngineering(dataframe)
                fi.feature_importance()
            else:
                st.markdown('## Machine Learning')
                st.text('')
                st.markdown('---')

                if dataset_type == 'Iris':
                    st.markdown('### Iris Dataset')
                elif dataset_type == 'Wine':
                    st.markdown('### Wine Dataset')
                elif dataset_type == 'breast_cancer':
                    st.markdown('### Breast Cancer Dataset')

                # Show the dataset    
                st.dataframe(dataframe.head())

                # Create slider for split dataset
                st.markdown('### Split Dataset')
                split_dataset_slider = st.slider('Data Training', min_value=10, max_value=100, value=80, step=5)
                X_train, X_test, y_train, y_test = split_dataset(dataframe, test_size=split_dataset_slider/100)
                st.markdown('---')

                if radio_ml_model == 'Decision Tree':
                    st.markdown('### Decision Tree')
                    decision_tree = train_and_predict(
                        classifier=DecisionTree(), 
                        X_train=X_train,
                        X_test=X_test, 
                        y_train=y_train,
                        y_test=y_test
                    )                    
                    st.markdown('---')
                elif radio_ml_model == 'KNN':
                    st.markdown('### KNN')
                    knn = train_and_predict(
                        classifier=KNN(), 
                        X_train=X_train,
                        X_test=X_test, 
                        y_train=y_train,
                        y_test=y_test
                    )
                    st.markdown('---')
                elif radio_ml_model == 'Neural Network':
                    st.markdown('### Neural Network')
                    neural_network = train_and_predict(
                        classifier=NeuralNetwork(), 
                        X_train=X_train,
                        X_test=X_test, 
                        y_train=y_train,
                        y_test=y_test
                    )
                    st.markdown('---')
                
# This function is not implemented yet
def prediction_dataframe(prediction_results):
    df = pd.DataFrame()
    pass

def train_and_predict(classifier=None, X_train=None, X_test=None, y_train=None, y_test=None) -> object:
    clf = classifier
    bt_train = st.button('Train and Predict')
    if bt_train:
        results = clf.fit_and_predict(X_train, y_train, X_test)
        st.success('Training Success')
        acc_score = accuracy_score(y_test, results)
        st.write('Accuracy: {:.2f}'.format(acc_score))
    return clf

def split_dataset(dataframe, test_size=0.2) -> object:
    X = dataframe.drop('target', axis=1)
    y = dataframe['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

if __name__=="__main__":
    main()         