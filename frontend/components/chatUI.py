# from multiprocessing.connection import answer_challenge

import streamlit as st
# from pycparser.ply.yacc import resultlimit

from utils.api import ask_query

def render_chat_ui():
    st.subheader(" 💬 Chat with AI bot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg['role']).markdown(msg['content'])

    user_input = st.chat_input("Type ur message ...")
    if user_input:
        st.chat_message('user').markdown(user_input)
        st.session_state.messages.append({'role':'user', 'content':user_input})

        result = ask_query(user_input)
        if result.status_code == 200:
            answer = result.json()['response']
            st.chat_message('assistant').markdown(answer)
            st.session_state.messages.append({'role':'assistant', 'content': answer})
        else:
            st.error(f"Error: {result.text}")
