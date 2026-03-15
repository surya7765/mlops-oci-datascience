import json
import logging
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# LOAD MODEL (RUNS ONCE)
# ------------------------
def load_model():
    # Files (latest.faiss, chunk_metadata.json) are bundled in the model artifact.
    # No OCI download needed — they are present in the same directory as score.py.

    logger.info("Loading FAISS index...")
    model_dir = os.path.dirname(os.path.realpath(__file__))

    index = faiss.read_index(os.path.join(model_dir, "latest.faiss"))

    logger.info("Loading chunk metadata...")

    with open(os.path.join(model_dir, "chunk_metadata.json")) as f:
        chunk_metadata = json.load(f)

    logger.info("Loading embedding model...")
    # embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(os.path.join(model_dir, "minilm"))


    # CPU-only deployment on OCI Data Science free tier.
    device = "cpu"
    logger.info(f"Device: {device}")

    logger.info("Loading TinyLlama...")
    
    model_name = os.path.join(model_dir, "tinyllama")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # explicit for CPU
    )
    llm.eval()

    logger.info("All models loaded successfully.")

    return {
        "index": index,
        "metadata": chunk_metadata,
        "embedder": embedding_model,
        "tokenizer": tokenizer,
        "llm": llm,
        "device": device,
    }


# ------------------------
# PREDICT FUNCTION
# ------------------------
def predict(data, model):
    try:
        query = data.get("query", "").strip()
        if not query:
            return {"error": "Query must be a non-empty string."}

        embedding_model = model["embedder"]
        index = model["index"]
        chunk_metadata = model["metadata"]
        tokenizer = model["tokenizer"]
        llm = model["llm"]

        # Encode query — encode() returns float32 numpy array directly
        query_embedding = embedding_model.encode([query]).astype("float32")

        # Search FAISS
        k = 5
        distances, indices = index.search(query_embedding, k)

        # Guard: FAISS returns -1 when k > number of stored vectors
        contexts = []
        for idx in indices[0]:
            if idx == -1:
                continue
            key = str(idx)
            if key in chunk_metadata:
                contexts.append(chunk_metadata[key])
            else:
                logger.warning(f"Index {idx} not found in chunk_metadata.")

        if not contexts:
            return {"answer": "I don't know — no relevant context was found."}

        # Guard context length: ~4 chars per token, reserve 300 tokens for prompt + answer
        max_context_chars = (1024 - 300) * 4
        combined_context = "\n\n".join(contexts)
        if len(combined_context) > max_context_chars:
            logger.warning(
                f"Context truncated from {len(combined_context)} to {max_context_chars} chars."
            )
            combined_context = combined_context[:max_context_chars]

        # TinyLlama chat prompt format
        prompt = (
            "<|system|>\n"
            "You are a helpful DevOps assistant.\n"
            "Answer the user's question using ONLY the provided context.\n"
            "If the answer is not in the context, say \"I don't know\".\n"
            "<|user|>\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{query}\n"
            "<|assistant|>\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        # No .to(device) needed — CPU tensors are already on CPU

        logger.info("Running inference on CPU (may take 30-90s on free tier)...")

        with torch.no_grad():
            outputs = llm.generate(
                **inputs,
                max_new_tokens=80,          # reduced from 80 to cut CPU latency
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {"answer": answer.strip()}

    except Exception as e:
        logger.exception("predict() failed")
        return {"error": str(e)}