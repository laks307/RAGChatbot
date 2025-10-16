from csv import excel
from sys import exception
from model.vectordb import embed_query
from model.llm import invoke_chain
from model.query_handlers import query_chain
from log import logger
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi.responses import JSONResponse
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pydantic import Field
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
from typing import List, Optional
import os


router = APIRouter()

@router.post('/ask-query/')
async def ask_query(question: str = Form(...)):
    try:
        logger.info(f"user query: {question}")

        # Embed model + Pinecone setup
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
        embed_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        embedded_query = embed_model.embed_query(question)
        res = index.query(vector=embedded_query, top_k=3, include_metadata=True)
        logger.info(f"result from pinecone: {res}")
        docs = [
            Document(
                page_content=match["metadata"].get("text", ""),
                metadata=match["metadata"]
            ) for match in res["matches"]
        ]

        class SimpleRetriever(BaseRetriever):
            tags: Optional[List[str]] = Field(default_factory=list)
            metadata: Optional[dict] = Field(default_factory=dict)

            def __init__(self, documents: List[Document]):
                super().__init__()
                self._docs = documents

            def _get_relevant_documents(self, query: str) -> List[Document]:
                return self._docs

        retriever = SimpleRetriever(docs)
        chain = invoke_chain(retriever)
        result = query_chain(chain, question)

        logger.info("query successful")
        return result

    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})