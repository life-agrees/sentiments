import streamlit as st
import pandas as pd
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from io import StringIO

st.set_page_config(page_title="Sentiment Analysis Tool ",page_icon=":smiley:",layout="wide")
@st.cache_data

def load_data():
    with open('model_class/model.pkl','rb')as file:
        model=pickle.load(file)
    with open('model_class/tfidf.pkl','rb')as file:
        tfidf=pickle.load(file)
    with open('model_class/le.pkl','rb')as file:
        le=pickle.load(file)
    
    return model,tfidf,le

model,tfidf,le=load_data()



st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #FFFFFF;
    }
    .main {
        background: #1E1E1E;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        max-width: 800px;
        margin: 2rem auto;
        text-align: center;
    }
    h1 {
        color: #BB86FC;
        text-align: center;
        font-family: 'Arial', sans-serif;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 18px;
        color: #BBBBBB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-size: 18px;
        padding: 10px;
        background: #333333;
        color: #FFFFFF;
        border: 1px solid #444444;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stFileUploader label {
        font-size: 18px;
        margin-top: 1rem;
        color: #FFFFFF;
    }
    .stButton button {
        font-size: 18px;
        padding: 10px 20px;
        background-color: #BB86FC;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #A66CFF;
    }
    .result-positive {
        color: #4CAF50;
        font-weight: bold;
        font-size: 20px;
        margin-top: 1rem;
    }
    .result-negative {
        color: #FF5252;
        font-weight: bold;
        font-size: 20px;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)




st.markdown('<h1>Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="description">
        Welcome to the Sentiment Analysis App! This application allows you to analyze the sentiment of your reviews. Whether you are reviewing hotels, restaurants, products, or any other service, our tool can help determine if the sentiment expressed in your review is positive or negative.<br><br>
        <strong>Instructions:</strong><br>
        - Enter your review in the text area below or upload a file containing reviews.<br>
        - Click the <b>Submit</b> button to see the sentiment analysis result.<br>
        - Sentiments will be highlighted.<br><br>
        We welcome all kinds of reviews!
    </div>
    """,
    unsafe_allow_html=True
)


def predict_review(text):
    vector = tfidf.transform([text]).toarray()
    prediction=model.predict(vector)
    return le.inverse_transform(prediction)[0]



st.markdown('## Enter a Single Review')
user_intent = st.text_area('Enter your review', height=200)

st.sidebar.markdown('## Or Upload a File Containing Reviews')
uploaded_file = st.sidebar.file_uploader("Choose a file",type=['txt','csv'])

button = st.button('Submit')


if button:
    if user_intent:
        result = predict_review(user_intent)
        if result == 'Positive':
            st.markdown('<p class="result-positive">Positive Review</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result-negative">Negative Review</p>', unsafe_allow_html=True)
    elif uploaded_file is not None:
        if uploaded_file.type == 'text/plain':
            stringio = StringIO(uploaded_file.getvalue().decode('utf-8'))
            reviews = stringio.readlines()
            st.sidebar.write('Preview of Uploaded File:')
            st.sidebar.text_area('file content',value=''.join(reviews[:5]),height=100)
            st.sidebar.write('Number of reviews: {}'.format(len(reviews)))

        elif uploaded_file.type == 'text/csv':
            df = pd.read_csv(uploaded_file)
            reviews = df.iloc[:, 0].tolist()
            st.sidebar.write('Preview of Uploaded File:')
            st.sidebar.dataframe(df.head)
            st.sidebar.write('Number of reviews: {}'.format(len(reviews)))

        for review in reviews:
            result = predict_review(review.strip())
            if result == 'Positive':
                st.markdown('<p class="result-positive">Positive Review</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="result-negative">Negative Review</p>', unsafe_allow_html=True)
    else:
        st.error("Please Enter A Review or Upload A File")




st.markdown(
    """
    <style>
    .stTextArea textarea {
        font-size: 18px;
        padding: 10px;
        margin-top: 10px;
    }
    .stFileUploader label {
        font-size: 18px;
        margin-top: 1rem;
        color: #FFFFFF;
    }
    .stButton button {
        font-size: 18px;
        padding: 10px 20px;
        background-color: #BB86FC;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        margin-top: 10px;
    }
    .stButton button:hover {
        background-color: #A66CFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)