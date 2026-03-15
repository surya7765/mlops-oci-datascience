from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import oci
import faiss
import numpy as np
import torch
import tempfile
import json
import threading
import os
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

app = FastAPI(title="DevOps Assistant API")

# --------------------------------
# REQUEST MODELS
# --------------------------------

class ChatRequest(BaseModel):
    query: str

# --------------------------------
# OCI SETUP
# --------------------------------

signer = oci.auth.signers.get_resource_principals_signer()
object_storage = oci.object_storage.ObjectStorageClient({}, signer=signer)
namespace = object_storage.get_namespace().data
bucket_name = "mlops-llm-data"

# --------------------------------
# LOAD FAISS INDEX
# --------------------------------

response = object_storage.get_object(namespace, bucket_name, "models/latest.faiss")
with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(response.data.content)
    index_path = tmp.name

index = faiss.read_index(index_path)
index.nprobe = 10
faiss.omp_set_num_threads(4)
print("Index loaded:", index.ntotal)

# --------------------------------
# LOAD CHUNK METADATA
# --------------------------------

response = object_storage.get_object(namespace, bucket_name, "models/chunk_metadata.json")
chunk_metadata = json.loads(response.data.content.decode("utf-8"))
print("Metadata loaded:", len(chunk_metadata))

documents = list(chunk_metadata.values())

tokenized_docs = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

print("BM25 index ready")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# --------------------------------
# LOAD MODELS (LOCAL PATHS)
# --------------------------------

# Works both locally and in deployment
base_dir = os.path.dirname(os.path.realpath(__file__))

# If running as a notebook/script directly, fall back to rag_deployment path
if not os.path.exists(os.path.join(base_dir, "tinyllama")):
    base_dir = "/home/datascience/rag-pipeline/rag_deployment"
    
embedding_model = SentenceTransformer(
    os.path.join(base_dir, "minilm"),
    local_files_only=True
)
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(base_dir, "rag_deployment/tinyllama"))
# llm = AutoModelForCausalLM.from_pretrained(
#     os.path.join(base_dir, "tinyllama"),
#     torch_dtype=torch.float32
# )
tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(base_dir, "tinyllama"),
    local_files_only=True
)
llm = AutoModelForCausalLM.from_pretrained(
    os.path.join(base_dir, "tinyllama"),
    torch_dtype=torch.float32,
    local_files_only=True
)
llm.eval()
print("Models loaded")

# --------------------------------
# EMBEDDING CACHE (LRU, max 100)
# --------------------------------

class LRUCache:
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        self.cache[key] = value

embedding_cache = LRUCache(max_size=100)

def get_query_embedding(query):
    cached = embedding_cache.get(query)
    if cached is not None:
        return cached
    embedding = embedding_model.encode(query)
    embedding = np.array(embedding).astype("float32").reshape(1, -1)
    embedding_cache.set(query, embedding)
    return embedding

# --------------------------------
# CONTEXT RETRIEVAL
# --------------------------------

def retrieve_context(query):

    query_embedding = get_query_embedding(query)

    k_vector = 5
    k_keyword = 5

    # Vector search
    distances, indices = index.search(query_embedding, k_vector)

    vector_docs = []

    for idx in indices[0]:
        vector_docs.append(chunk_metadata.get(str(idx), ""))

    # Keyword search
    tokenized_query = query.split()

    keyword_docs = bm25.get_top_n(
        tokenized_query,
        documents,
        n=k_keyword
    )

    # Merge candidates
    candidate_docs = list(set(vector_docs + keyword_docs))

    # Re-ranking
    pairs = [(query, doc) for doc in candidate_docs]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(candidate_docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, score in ranked_docs[:3]]

    combined_context = "\n\n".join(top_docs)

    return combined_context

# --------------------------------
# PROMPT BUILDER
# --------------------------------

def build_prompt(query, context):
    return (
        "<|system|>\n"
        "You are a helpful DevOps assistant.\n"
        "Answer the user's question using ONLY the provided context.\n"
        "If the answer is not in the context, say \"I don't know\".\n"
        "<|user|>\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n"
        "<|assistant|>\n"
    )

# --------------------------------
# HEALTH & STATUS
# --------------------------------

@app.get("/health")
def health():
    return {"status": "running"}

@app.get("/status")
def status():
    return {
        "vectors": index.ntotal,
        "metadata": len(chunk_metadata),
        "model": "TinyLlama",
        "cache_size": len(embedding_cache.cache),
        "status": "running"
    }

# --------------------------------
# STANDARD CHAT (NON STREAMING)
# --------------------------------

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        context = retrieve_context(request.query)
        if not context:
            return {"answer": "I don't know — no relevant context found."}
        prompt = build_prompt(request.query, context)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return {"answer": answer.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------
# STREAMING GENERATORS
# --------------------------------

def generate_plain_stream(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True  # handles prompt skipping automatically
    )
    thread = threading.Thread(
        target=llm.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    )
    thread.start()
    for token in streamer:
        yield token
    thread.join()
    

def generate_json_stream(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = threading.Thread(
        target=llm.generate,
        kwargs=dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    )
    thread.start()
    for token in streamer:
        yield json.dumps({"token": token}) + "\n"
    thread.join()

# --------------------------------
# STREAMING ENDPOINTS
# --------------------------------

@app.post("/chat_stream")
def chat_stream(request: ChatRequest):
    try:
        context = retrieve_context(request.query)
        prompt = build_prompt(request.query, context)
        return StreamingResponse(
            generate_plain_stream(prompt),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_stream_json")
def chat_stream_json(request: ChatRequest):
    try:
        context = retrieve_context(request.query)
        prompt = build_prompt(request.query, context)
        return StreamingResponse(
            generate_json_stream(prompt),
            media_type="application/json"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))