import numpy as np
import requests
import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List


# Logging
logging.basicConfig(level=logging.DEBUG)



class SearchRequest(BaseModel):
    docs: List[str]
    query: str



class SearchResponse(BaseModel):
    matches: List[str]


API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
AIPROXY_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjMwMDIwMzRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.KEQjxQbjAIHY8_0l-WpiOL_KrBslnPTFZnexib9N6qc"

def get_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {AIPROXY_TOKEN}",
        }
        payload = {
            "model": "text-embedding-3-small",  # Update model if needed
            "input": texts,
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    except Exception as e:
        logging.error(f"Error fetching embeddings: {e}")
        raise


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a1_arr = np.array(v1)
    a2_arr = np.array(v2)
    if not a1_arr.any() or not a2_arr.any():
        raise ValueError("One or both vectors are empty")
    return np.dot(a1_arr, a2_arr) / (
        np.linalg.norm(a1_arr) * np.linalg.norm(a2_arr)
    )



    


