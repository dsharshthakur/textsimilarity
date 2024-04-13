import streamlit as st
from Helping import TextSimilarity, model

# main title
st.title("Text Similarity Check")
st.markdown("<br>", unsafe_allow_html=True)

# text inputs
text1 = st.text_input("Enter first text:")
text2 = st.text_input("Enter second text:")

st.markdown("<br>", unsafe_allow_html=True)

# calculate button
col1, col2, col3 = st.columns(3)
with col2:
    btn = st.button("Calculate")

# perform calculation and display output
if btn:
    score = TextSimilarity(text1, text2, model)
    similarity = round(score.calculate(), 2)
    col1, col2, col3 = st.columns(3)

    # if similarity score is negative
    if similarity < 0:
        similarity = 0

    # display final output
    st.success({"similarity_score" :similarity})
