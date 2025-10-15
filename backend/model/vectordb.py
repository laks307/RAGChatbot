import os
from typing import Optional, List
from pydantic import Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
#import pinecone
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from backend.log import logger
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from backend.model.llm import invoke_chain

from sqlalchemy.testing.suite.test_reflection import metadata
from backend.log import logger

load_dotenv()

UPLOAD_DOCS =  './upload_docs'
os.makedirs(UPLOAD_DOCS, exist_ok=True)

google_api_key = os.getenv("GOOGLE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
embed_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', google_api_key=google_api_key)

    # initialize pinecone
import time
logger.debug("initializing pinecone")
vectordb = Pinecone(pinecone_api_key=pinecone_api_key)
spec = ServerlessSpec(cloud = 'aws', region = 'us-east-1')
existing_index = [i['name'] for i in vectordb.list_indexes()]
logger.debug(f"existing index: {existing_index}")
if pinecone_index_name not in existing_index:
    vectordb.create_index(
            name=pinecone_index_name,
            spec=spec,
            metric='dotproduct',
            dimension=3072
        )
    while not vectordb.describe_index(pinecone_index_name).status["ready"]:
        time.sleep(1)

logger.debug("pinecone index created")
logger.debug("initializing index")
index = vectordb.Index(pinecone_index_name)
logger.debug("index initialized")
  #  return vectordb

def embed_docs(files):
    # vectordb = initialize_pinecone()


    # logger.debug("index initialized, proceeding to embeddings")
    embed_model = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', google_api_key=google_api_key)
    logger.debug(f"embedding model: {embed_model.model}")
    file_path = []

    for file in files:
        path = Path(UPLOAD_DOCS)/file.filename
        file_path.append(str(path))
        with open(path, 'wb') as f:
            f.write(file.file.read())
    logger.debug("files uploaded locally")
    for file in file_path:
        loader = PyPDFLoader(file)
        docs = loader.load()
        logger.debug(f"docs: loaded")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        logger.debug(f"chunks: splitted")

        text = [i.page_content for i in chunks]
        metadata = [i.metadata for i in chunks]
        print(metadata[0])
        for i in range(len(chunks)):
            temp = dict()
            temp = metadata[i]
            temp['text'] = text[i].replace('\n', '')
            metadata[i] = temp
        print('==================')
        print('after adding text')
        print(metadata[0])
        id = [f"{Path(file).stem} - {i}" for i in range(len(chunks))]
        logger.debug(f"proceeding with embeddings")
        try:
            embeddings = embed_model.embed_documents(text)
            print(len(embeddings))
            print(len(embeddings[0]))
        except Exception as e:
            logger.error(f"embeddings failed: {e}")
        logger.debug(f"embeddings: done")
        try:
            with tqdm(total=len(embeddings), desc='uploading embeds to pinecone') as pbar:
                index.upsert(vectors=zip(id, embeddings, metadata))
                pbar.update(len(embeddings))
        except Exception as e:
            logger.error(f"uploading embeddings to DB failed: {e}")

    logger.debug("embeds uploaded to pinecone")


def embed_query(query):
    #cvectordb = initialize_pinecone()
    #index = vectordb.index(pinecone_index_name)
    logger.debug(f"coming here")
    try:
        #if not embed_query:
        embed_query = embed_model.embed_query(query)
        #nlogger.debug(f"embed_query: {embed_query}")
    except Exception as e:
        logger.error(f"query failed: {e}")
    logger.debug(f"embeding query done")

    rel_docs = index.query(vector=embed_query, top_k=1, include_metadata=True)
    logger.debug(f"rel_docs: fetched - {rel_docs}")
    docs = [
        Document(page_content=i['metadata'].get('text',''), metadata=i['metadata'])
        for i in rel_docs['matches']
    ]
    print(docs)

    logger.debug(f'relevant docs extracted from pinecone - {len(docs)}')

    class SimpleRetriever(BaseRetriever):
        tags: Optional[List[str]] = Field(default_factory=list)
        metadata: Optional[dict] = Field(default_factory=dict)

        def __init__(self, documents: List[Document]):
            super().__init__()
            self._docs = documents

        def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
            return self._docs

    retriever = SimpleRetriever(documents=docs)
    logger.debug(f"invoking llm and chain")
    try:

        chain = invoke_chain(retriever)
        logger.debug(f"chain: invoked")
        print(chain)
        result = invoke_llm(chain, query)
        print(result)
    except Exception as e:
        logger.error(f"while invoking llm failed: {e}")
    return result










