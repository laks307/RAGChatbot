import streamlit as st
from components.chatUI import render_chat_ui
from components.history_msg_download import render_history_msg_download
from components.upload_docs import render_upload_sidebar

st.set_page_config(page_title='AI Medical Assistant', layout='wide')
st.title('  🩺 LAKS Medical Chatbot ')

render_chat_ui()
render_upload_sidebar()
render_history_msg_download()