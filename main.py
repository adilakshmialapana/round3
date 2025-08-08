import os
import io
import re
import asyncio
import traceback
from typing import List, Dict
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

import fitz  # PyMuPDF for PDF extraction

from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables from .env file located in the working directory
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

if not GEMINI_API_KEY or not AUTH_TOKEN:
    raise RuntimeError("Please set GEMINI_API_KEY and AUTH_TOKEN in your .env file")

app = FastAPI(
    title="HackRX 6.0 - Enhanced Gemini 2.5 Flash RAG",
    description="Optimized RAG aiming for ≥90% accuracy and ≤15s response for 10 concurrent queries.",
    version="3.0.0"
)

security = HTTPBearer()

_vectorstore_cache: Dict[str, BaseRetriever] = {}

class DocumentInput(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answers(BaseModel):
    answers: List[str]

def clean_pdf_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,;?!:\'\"()-]', '', text)
    text = re.sub(r'(\.|,|;|\?){2,}', r'\1', text)
    return text.strip()

def load_pdf_and_split(document_bytes: bytes) -> List[Document]:
    doc = fitz.open(stream=document_bytes, filetype="pdf")
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        if text.strip():
            clean_text = clean_pdf_text(text)
            pages.append(Document(page_content=clean_text, metadata={"page": i + 1}))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    return chunks

async def get_or_create_retriever(document_url: str) -> BaseRetriever:
    if document_url in _vectorstore_cache:
        return _vectorstore_cache[document_url]

    try:
        import requests
        response = requests.get(document_url)
        response.raise_for_status()
        pdf_bytes = response.content

        chunks = load_pdf_and_split(pdf_bytes)
        if not chunks:
            raise ValueError("No extractable text found in PDF.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GEMINI_API_KEY,
            task_type="retrieval_document"
        )

        faiss_store = FAISS.from_documents(chunks, embeddings)
        faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 20})

        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = 20

        ensemble = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=[0.8, 0.2]
        )

        _vectorstore_cache[document_url] = ensemble
        return ensemble

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retriever setup failed: {str(e)}"
        )

_llm_instance = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.0,
    max_output_tokens=1024,
    convert_system_message_to_human=True
)

FEW_SHOT_EXAMPLES = """
Example 1:
Context:
--- Context Page 3 ---
The grace period for premium payment under the National Parivar Mediclaim Plus Policy is 30 days without penalty.

Question:
What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?

Answer:
The grace period for premium payment under the National Parivar Mediclaim Plus Policy is 30 days without penalty.

Example 2:
Context:
--- Context Page 7 ---
Pre-existing diseases will be covered after a waiting period of 48 months.

Question:
What is the waiting period for pre-existing diseases (PED)?

Answer:
The waiting period for pre-existing diseases (PED) is 48 months.
"""

async def answer_question(question: str, retriever: BaseRetriever, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        try:
            docs = retriever.get_relevant_documents(question)
            if not docs:
                return "Information not available in the document."

            context_chunks = []
            char_limit = 18000
            current_len = 0
            for doc in docs:
                content = f"--- Context Page {doc.metadata.get('page')} ---\n{doc.page_content}\n"
                if current_len + len(content) > char_limit:
                    break
                context_chunks.append(content)
                current_len += len(content)

            context_text = "\n".join(context_chunks)

            prompt = f"""
You are an expert insurance policy assistant.

Rules:
- Answer only from the context below.
- If info is missing, say "Information not available in the document."
- Use full sentences, no formatting or lists.
- Provide concise, clear answers.
- Use only the provided information.

Few-shot examples:
{FEW_SHOT_EXAMPLES}

Context:
{context_text}

Question:
{question}

Answer:
""".strip()

            response = await _llm_instance.ainvoke(prompt)
            answer = response.content.strip().replace('\n', ' ').strip()
            return answer

        except Exception as e:
            return f"Error processing question: {str(e)}"

@app.post("/api/v1/hackrx/run", response_model=Answers, status_code=status.HTTP_200_OK)
async def run_hackrx(data: DocumentInput, credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")

    retriever = await get_or_create_retriever(str(data.documents))
    if not retriever:
        raise HTTPException(status_code=500, detail="Retriever Initialization Failed.")

    semaphore = asyncio.Semaphore(10)

    tasks = [answer_question(q, retriever, semaphore) for q in data.questions]
    results = await asyncio.gather(*tasks)

    return Answers(answers=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
