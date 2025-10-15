import streamlit as st


def render_history_msg_download():
    if st.session_state.get('message'):
        download_msg = [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
        download_msg = "\n\n".join(download_msg)
        st.download_button("Download Chat History", download_msg, file_name = "chat_history.txt", mime = 'text/plain')