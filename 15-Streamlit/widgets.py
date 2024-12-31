import streamlit as st
import pandas as pd

st.title("Streamlit Widget App")

name = st.text_input("Enter your name")
if name:
    st.write(f"Hello, {name}")

age = st.slider("Select your age: ",0,100,25)
st.write(f"Your age is {age}")

options = ["Python","C Language","Data Science"]
choice = st.selectbox("Please one course from the following:",options)
st.write(f"You chose is {choice}")

data = {
    "Name":["John","Jane","Jake","Jill"],
    "Age":[28,24,35,40],
    "City":["New York","Los Angeles","Chicago","Houstan"]
}
df = pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)

file_uploader = st.file_uploader("Please upload data")
if file_uploader is not None:
    df = pd.read_csv("sampledata.csv")
    st.write(df)