import streamlit as st
from modeling.prediction import prediction_display
import time
import plotly.express as px

st.set_page_config(
    page_title="Real-Time Volatility Dashboard",
    page_icon="✅",
    layout="wide",
)

st.markdown("<h1 style='text-align: center; color: Black;'> Real-Time Volatility Dashboard </h1>", unsafe_allow_html=True)


data = prediction_display()

cols = st.empty()
placeholder = st.empty()

def update_chart():
    for i in range(480):
        with cols.container():
            if i % 10 == 0:
                col1, col2, col3, col4, col5 = st.columns(5,gap='large')

                index=[i+1, i+2, i+5, i+10, i+30]
                old_index = [i, i+1, i+4, i+9, i+29]

                key_value = [data.iloc[j]['prediction'] for j in index]
                old_value = [data.iloc[j]['prediction'] for j in old_index]

                col1.metric(
                        label="In 1 minute",
                        value=round(float(key_value[0]), 4),
                        delta=f"{round(float((key_value[0]-old_value[0])/key_value[0]), 4)}%")

                col2.metric(
                        label="In 2 minute",
                        value=round(float(key_value[1]), 4),
                        delta=f"{round(float((key_value[1]-old_value[1])/key_value[1]), 4)}%")

                col3.metric(
                        label="In 5 minutes",
                        value=round(float(key_value[2]),4),
                        delta=f"{round(float((key_value[2]-old_value[2])/key_value[2]),4)}%")

                col4.metric(
                        label="In 10 minutes",
                        value=round(float(key_value[3]),4),
                        delta=f"{round(float((key_value[3]-old_value[3])/key_value[3]),4)}%")

                col5.metric(
                        label="In 30 minutes",
                        value=round(float(key_value[4]),4),
                        delta=f"{round(float((key_value[4]-old_value[4])/key_value[4]),4)}%")

            if i % 5 == 0:
                to_plot = data[i:i+100]
                fig = px.line(to_plot, x="time(GMT)",y="prediction",width=1800, height=1000)

                placeholder.write(fig)

            time.sleep(1)


update_chart()
