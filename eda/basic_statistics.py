import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

class BasicStatisticalAnalysis:
    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

    def describe(self) -> None:
        if self.dataframe.columns.size < 6:
            st.write(self.dataframe.describe())
        else:
            st.write(self.dataframe.describe().T)

    def correlation(self) -> None:
        fig = px.imshow(np.round(self.dataframe.corr(), 2), text_auto=True, aspect='auto')
        st.plotly_chart(fig, use_container_width=True)