import streamlit as st
import os
import sys

# Simple test to verify deployment works
st.title("🚀 RAG System - Loading...")

try:
    # Try to import and run the main app
    exec(open('app.py').read())
except Exception as e:
    st.error(f"Error loading main app: {e}")
    st.write("**Fallback Mode - Basic Functionality**")
    
    # Basic Streamlit app as fallback
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a document", type=['pdf', 'txt', 'docx'])
    
    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
    
    st.header("🔍 Ask Questions")
    question = st.text_input("Enter your question:")
    
    if question:
        st.write(f"You asked: {question}")
        st.info("Main RAG system loading... Please wait or refresh the page.")
    
    st.sidebar.write("**System Status:**")
    st.sidebar.write("✅ Streamlit: Working")
    st.sidebar.write("⏳ RAG System: Loading...")
    
    st.write("---")
    st.write("**If you see this message, the basic deployment is working.**")
    st.write("The full RAG system will load shortly. Please refresh the page in 1-2 minutes.")