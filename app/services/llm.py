from langchain.llms import Ollama
from fastapi import HTTPException

ollama_model = Ollama(model="3.2")

def generate_response(prompt: str) -> str:
    try:
        response = ollama_model(prompt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))