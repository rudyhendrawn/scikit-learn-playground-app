import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor


class BasicFeatureEngineering:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def drop_columns(self, columns=None) -> None:
        if columns is None:
            pass
        else:
            st.write(self.dataframe.drop(columns=columns))

    def feature_importance(self) -> pd.DataFrame:
        feature_selector = RandomForestRegressor()
        feature_selector.fit(self.dataframe.iloc[:, :-1], self.dataframe.iloc[:, -1])
        importances = feature_selector.feature_importances_
        idx = np.argsort(importances)
        feature_importance_df = pd.DataFrame(
            data={'feature': self.dataframe.columns[idx].tolist(), 'importance': importances[idx]})
        fig = px.bar(feature_importance_df, x='feature', y='importance',color='importance')
        st.plotly_chart(fig, use_container_width=True)

        return feature_importance_df
        
