# Create a class that convert sklearn datasets to pandas dataframe
# The dataframe should contain all dataset features and target
import pandas as pd
import streamlit as st

# Classfication Datasets
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
# Regression Datasets
from sklearn.datasets import load_diabetes

class ConvertDataset:
    def __init__(self, data) -> None:
        self.data = data
        self.dataframe = None

    @st.cache(suppress_st_warning=True)
    def convert_to_dataframe(self) -> pd.DataFrame:
        if self.data == 'iris':
            self.dataframe = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
            self.dataframe['target'] = load_iris().target
        elif self.data == 'wine':
            self.dataframe = pd.DataFrame(load_wine().data, columns=load_wine().feature_names)
            self.dataframe['target'] = load_wine().target
        elif self.data == 'breast_cancer':
            self.dataframe = pd.DataFrame(load_breast_cancer().data, columns=load_breast_cancer().feature_names)
            self.dataframe['target'] = load_breast_cancer().target
        elif self.data == 'diabetes':
            self.dataframe = pd.DataFrame(load_diabetes().data, columns=load_diabetes().feature_names)
            self.dataframe['target'] = load_diabetes().target
        else:
            print('Invalid dataset type')
            return

        return self.dataframe
