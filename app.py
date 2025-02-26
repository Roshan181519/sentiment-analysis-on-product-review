import streamlit as st
from newsenti import preprocess_text, model  # Importing your existing functions and model

# Streamlit UI
st.title("Sentiment Analysis")

# Input text box
user_review = st.text_area("Enter a Review:")

if st.button("Predict Sentiment"):
    if user_review:
        # Preprocess the input text
        processed_review = preprocess_text(user_review)
        
        # Make prediction
        sentiment = model.predict([processed_review])[0]
        
        # Display result
        st.write(f"The sentiment is: {sentiment}")
    else:
        st.write("Please enter a review.")
