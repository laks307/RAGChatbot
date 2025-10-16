from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from log import logger
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

def get_system_msg():
    msg = """
            You are a helpful Medical assistant. Respond politely.
            You will be given context delimited by triple angular brackets and question delimited by triple backticks. 
            Answer the question only based on the given context.
            If you dont find any answer relevant to the question, then simply respond "Info not found in doc"
            
            Context:
            <<<{context}>>>
            
            Question:
            '''{question}'''         
            """
    return msg


def invoke_chain(retriever):
    llm = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature = 0, api_key=api_key)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are **MediBot**, an AI-powered assistant trained to help users understand medical documents and health-related questions.

    Your job is to provide clear, accurate, and helpful responses based **only on the provided context**.

    ---

    🔍 **Context**:
    {context}

    🙋‍♂️ **User Question**:
    {question}

    ---

    💬 **Answer**:
    - Respond in a calm, factual, and respectful tone.
    - Use simple explanations when needed.
    - If the context does not contain the answer, say: "I'm sorry, but I couldn't find relevant information in the provided documents."
    - Do NOT make up facts.
    - Do NOT give medical advice or diagnoses.
    """
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )


