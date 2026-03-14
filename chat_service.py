from fastapi import FastAPI
import oci
import faiss
import numpy as np
import torch
import tempfile
import json

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="DevOps Assistant API")

# OCI setup
signer = oci.auth.signers.get_resource_principals_signer()
object_storage = oci.object_storage.ObjectStorageClient({}, signer=signer)

namespace = object_storage.get_namespace().data
bucket_name = "mlops-llm-data"

# ------------------------
# LOAD FAISS INDEX
# ------------------------

response = object_storage.get_object(
    namespace,
    bucket_name,
    "models/latest.faiss"
)

with tempfile.NamedTemporaryFile(delete=False) as tmp:
    tmp.write(response.data.content)
    index_path = tmp.name

index = faiss.read_index(index_path)

print("Index loaded:", index.ntotal)

index.nprobe = 10

# ------------------------
# LOAD CHUNK METADATA
# ------------------------

response = object_storage.get_object(
    namespace,
    bucket_name,
    "models/chunk_metadata.json"
)

chunk_metadata = json.loads(
    response.data.content.decode("utf-8")
)

print("Metadata loaded:", len(chunk_metadata))

# ------------------------
# LOAD MODELS
# ------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(model_name)
llm.eval()

# ------------------------
# HEALTH
# ------------------------

@app.get("/health")
def health():
    return {"status": "running"}

# ------------------------
# CHAT
# ------------------------

@app.post("/chat")
def chat(query: str):

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    k = 5
    distances, indices = index.search(query_embedding, k)

    contexts = []

    distance_threshold = 1.5

    for rank, idx in enumerate(indices[0]):
    
        if distances[0][rank] > distance_threshold:
            continue

        contexts.append(chunk_metadata.get(str(idx), ""))

    contexts = contexts[:3]
    combined_context = "\n\n".join(contexts)

    prompt = f"""
<|system|>
You are a helpful DevOps assistant.

Answer the user's question using ONLY the provided context.
If the answer is not in the context, say "I don't know".

<|user|>
Context:
{combined_context}

Question:
{query}

<|assistant|>
"""
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
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return {"answer": answer}


@app.get("/status")
def status():

    return {
        "vectors": index.ntotal,
        "metadata": len(chunk_metadata),
        "model": "TinyLlama",
        "status": "running"
    }





    