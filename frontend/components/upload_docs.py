import streamlit as st
from utils.api import upload_docs

def render_upload_sidebar():
    st.sidebar.header("Upload medical documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDFs", type='pdf', accept_multiple_files=True)
    if uploaded_files and st.sidebar.button("Upload DB"):
        result = upload_docs(uploaded_files)
        if result.status_code == 200:
            st.sidebar.success("Uploaded PDFs successfully")
        else:
            st.sidebar.error(f"Error: {result.text}")
