import os
import io
import re
import asyncio
import traceback
from typing import List, Dict
import requests
import tempfile
from contextlib import asynccontextmanager

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, status, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from pydantic_settings import BaseSettings

# RAG-specific imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, BaseRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# LLM and Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

# --- Configuration ---
# This class now securely loads your keys from the .env file.
class Settings(BaseSettings):
    gemini_api_key: str
    groq_api_key: str
    auth_token: str
    embedding_model: str = "models/text-embedding-004"
    llm_model: str = "llama3-70b-8192"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# --- Global State and Cache ---
class AppState:
    llm: ChatGroq = None
    reranker: HuggingFaceCrossEncoder = None
    retriever_cache: Dict[str, BaseRetriever] = {}

app_state = AppState()

# --- FastAPI Lifespan Events for Pre-loading ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the LLM and Reranker model on startup
    print("Application starting up...")
    app_state.llm = ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=0.1,
        max_tokens=1024
    )
    print("Loading local reranker model... (This may take a moment on first run)")
    app_state.reranker = HuggingFaceCrossEncoder(model_name=settings.reranker_model_name)
    print("Models loaded successfully.")
    yield
    # Clean up resources on shutdown (optional)
    print("Application shutting down...")
    app_state.retriever_cache.clear()

# --- FastAPI App Instance ---
app = FastAPI(
    title="HackRX 6.0 - High-Performance RAG Solution",
    description="A highly optimized RAG pipeline with pre-loaded models and caching to meet <30s latency.",
    version="5.0.0",
    lifespan=lifespan
)

# --- Security ---
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(auth_scheme)):
    if credentials.credentials != settings.auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )
    return credentials

# --- Pydantic Data Models ---
class DocumentInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class AnswersResponse(BaseModel):
    answers: List[str]

# --- Core RAG Logic ---

def clean_pdf_text(text: str) -> str:
    """Cleans up common text extraction artifacts from PDFs."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;?()-:%$€£@\'"]', '', text)
    return text.strip()

async def get_or_create_retriever(document_url: str) -> BaseRetriever:
    """
    Retrieves a pre-built retriever from the cache or creates a new one
    if the document URL has not been processed before.
    """
    if document_url in app_state.retriever_cache:
        print(f"Retriever found in cache for: {document_url}")
        return app_state.retriever_cache[document_url]

    print(f"Retriever not in cache. Initializing for: {document_url}")
    try:
        response = requests.get(str(document_url))
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        os.remove(temp_file_path)

        if not docs:
            raise ValueError("No text could be extracted from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.page_content = clean_pdf_text(chunk.page_content)
        print(f"Document split into {len(chunks)} chunks.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.gemini_api_key,
            task_type="retrieval_document"
        )

        # Increase K to cast a wider net for the reranker
        faiss_vectorstore = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 15})
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 15
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )
        
        compressor = DocumentCompressorPipeline(transformers=[app_state.reranker])
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=ensemble_retriever
        )
        
        app_state.retriever_cache[document_url] = compression_retriever
        print("Retriever created and cached successfully.")
        return compression_retriever
    except Exception as e:
        print(f"ERROR during retriever initialization: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize retriever for document: {str(e)}"
        )

async def generate_answer(question: str, retriever: BaseRetriever, semaphore: asyncio.Semaphore) -> str:
    """Generates a single answer string for a given question."""
    async with semaphore:
        try:
            print(f"\nProcessing question: {question}")
            retrieved_docs = await retriever.ainvoke(question)
            
            if not retrieved_docs:
                print("No relevant documents found.")
                return "Information not available in the document."

            context = "\n\n".join([f"--- Context Chunk ---\n{doc.page_content}" for doc in retrieved_docs])
            
            prompt = f"""You are a specialized Question-Answering system for insurance documents. Provide a precise, factual, and direct answer based ONLY on the provided context.

            **CRITICAL INSTRUCTIONS:**
            1.  **Answer Solely from Context:** Do not use any external knowledge.
            2.  **Direct Answer:** Provide a direct and complete answer.
            3.  **Handle Missing Information:** If the context does not contain the answer, respond with: "Information not available in the document."
            4.  **No Formatting:** The answer must be a single, clean paragraph.

            **Context:**
            {context}

            **Question:** {question}

            **Answer:**
            """
            response = await app_state.llm.ainvoke(prompt)
            processed_answer = response.content.strip().replace('\n', ' ').strip()
            print(f"Generated Answer: {processed_answer}")
            return processed_answer
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
            return f"Error processing question: {str(e)}"

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=AnswersResponse, dependencies=[Depends(verify_token)])
async def run_rag_pipeline(data: DocumentInput):
    """
    Main API endpoint to run the RAG pipeline.
    It initializes the system with the provided document URL and answers all questions.
    """
    try:
        retriever = await get_or_create_retriever(str(data.documents))
        semaphore = asyncio.Semaphore(10)
        tasks = [generate_answer(q, retriever, semaphore) for q in data.questions]
        all_answers = await asyncio.gather(*tasks)
        return AnswersResponse(answers=all_answers)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"FATAL Error in /run endpoint: {e}")
        print(f"Full traceback: {error_details}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# --- Root Endpoint for Health Check ---
@app.get("/", summary="Health Check")
async def root():
    return {"status": "ok", "title": app.title, "version": app.version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
