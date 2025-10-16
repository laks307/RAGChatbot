from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from exception_handling import error_handling
from routes.upload_docs import router as upload_docs
from routes.ask_query import router as ask_query

app = FastAPI(title = 'chatbot', description = 'chatbot using langchain, fastapi, streamlit')

app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*'],
    allow_credentials = ['*']
)

# middleware exception handling
app.middleware('http')(error_handling)

# routes

# 1.to upload file in chatbot
app.include_router(upload_docs)
# 2. to post query
app.include_router(ask_query)


