import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px

# Flower icon
st.set_page_config(page_title="Iris Dataset", page_icon="🌺", layout="wide")

def main():

    models_json = json.load(open('models.json', 'r'))

    model_names = list(models_json.keys())
    model_data = list(models_json.values())

    model_paths = [data['path'] for data in model_data]
    models = [load_model(path) for path in model_paths]

    df = pd.read_csv('datasets/iris.csv')

    mean_values = df.drop('target', axis=1).mean(axis=0)

    col1, col2 = st.columns(2)

    with col1:
        sl = st.number_input('Sepal Length', value=mean_values['sepal length (cm)'])
        sw = st.number_input('Sepal Width', value=mean_values['sepal width (cm)'])

    with col2:
        pl = st.number_input('Petal Length', value=mean_values['petal length (cm)'])
        pw = st.number_input('Petal Width', value=mean_values['petal width (cm)'])

    sepals = df[['sepal length (cm)', 'sepal width (cm)', 'target']]
    petals = df[['petal length (cm)', 'petal width (cm)', 'target']]

    chart_cols = st.columns(2)

    with chart_cols[0]:
        fig = px.scatter(sepals, x='sepal length (cm)', y='sepal width (cm)', color='target')
        fig.add_scatter(x=[sl], y=[sw], mode='markers', marker=dict(color='red'), name='New')
        st.plotly_chart(fig)

    with chart_cols[1]:
        fig = px.scatter(petals, x='petal length (cm)', y='petal width (cm)', color='target')
        fig.add_scatter(x=[pl], y=[pw], mode='markers', marker=dict(color='red'), name='New')
        st.plotly_chart(fig)

    with st.columns(3)[1]:
        selected_model_name = st.selectbox('Select the model', model_names)

    selected_model = models[model_names.index(selected_model_name)]

    with st.columns(3)[1]:
        if st.button('Predict'):
            prediction = selected_model.predict([[sl, sw, pl, pw]])[0]
            st.title(prediction)

def load_model(path):
    return pickle.load(open(path, 'rb'))

if __name__ == '__main__':
    main()
