import streamlit as st
import pickle

def main():

    knn = pickle.load(open('knn.pkl', 'rb'))

    sl = st.number_input('Sepal Length')
    sw = st.number_input('Sepal Width')
    pl = st.number_input('Petal Length')
    pw = st.number_input('Petal Width')

    if st.button('Predict'):
        prediction = knn.predict([[sl, sw, pl, pw]])[0]
        st.write(f'The prediction is: {prediction}')

if __name__ == '__main__':
    main()
