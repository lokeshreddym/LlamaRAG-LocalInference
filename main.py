import os
import chromadb
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama
import PyPDF2
import nltk
from typing import List
import torch
from tqdm import tqdm
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
from nltk.corpus import stopwords
import string

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download("stopwords", quiet=True)

# Initialize FastAPI app
app = FastAPI()

# Model configuration
embedding_model_id = "BAAI/bge-small-en-v1.5"
generation_model_path = ".\Llama-2-7B-Chat-GGUF\llama-2-7b-chat.Q4_K_M.gguf"  # Download this file

# Initialize embedding model
embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
embedding_model = AutoModel.from_pretrained(embedding_model_id)

# Initialize Llama model for generation
llm = Llama(model_path=generation_model_path, n_ctx=2048, n_threads=1)

# Move embedding model to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model.to(device)

# Initialize Chroma client and collection
CHROMA_DATA_PATH = "./chroma_path"
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
collection = chroma_client.get_or_create_collection("general_documents")

def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

def preprocess_text(text: str, chunk_size: int = 512) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        if current_length + len(sentence) > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embedding(text: str) -> List[float]:
    inputs = embedding_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

def add_to_chroma(text_segments: List[str], batch_size: int = 100):
    for i in tqdm(range(0, len(text_segments), batch_size), desc="Adding to Chroma"):
        batch = text_segments[i:i+batch_size]
        batch_embeddings = [get_embedding(segment) for segment in batch]
        
        collection.add(
            embeddings=batch_embeddings,
            documents=batch,
            ids=[f"doc_{j}" for j in range(i, min(i+batch_size, len(text_segments)))]
        )

def retrieve_relevant_passages(query: str, top_k: int = 3) -> List[str]:
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results['documents'][0]

def generate_response(context: List[str], query: str) -> str:
    prompt = f"""Human: You are a helpful assistant. Based on the following context, answer the question. If the information is not in the context, say 'I don't have enough information to answer this question.'

Context: {' '.join(context)}

Question: {query}

Assistant: """

    response = llm(prompt, max_tokens=512, stop=["Human:", "\n"], echo=False)
    return response['choices'][0]['text'].strip()

def rag_pipeline(query: str) -> str:
    relevant_passages = retrieve_relevant_passages(query)
    response = generate_response(relevant_passages, query)
    return response.strip()

class Query(BaseModel):
    question: str

@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    else:
        # For other file types, you might need to implement specific extraction methods
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    processed_text = preprocess_text(text)
    add_to_chroma(processed_text)
    
    return {"message": "Document processed and added to the database successfully"}

@app.post("/query")
async def query(query: Query):
    result = rag_pipeline(query.question)
    return {"query": query.question, "response": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)