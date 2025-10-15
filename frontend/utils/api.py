from config import API_URL
import requests


def upload_docs(files):
    files_list = [('files', (f.name, f.read(), 'application/pdf')) for f in files]
    return requests.post(f"{API_URL}/upload-docs/", files=files_list)

def ask_query(question):
    return requests.post(f"{API_URL}/ask-query/", data={'question': question})