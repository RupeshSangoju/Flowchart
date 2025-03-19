import requests
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class DiagramRequest(BaseModel):
    user_input: str
    diagram_type: str

def summarize_text(text):
    try:
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error summarizing text: {str(e)}"

def generate_mermaid_syntax(user_input, diagram_type):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY is not set in environment variables"
    
    api_url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    summarized_text = summarize_text(user_input)
    prompt = f"Convert the following text into a {diagram_type} using Mermaid.js syntax: {summarized_text}"
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response from API")
    except requests.RequestException as e:
        return f"Error: Failed to connect to Perplexity API - {str(e)}"

@app.get("/", response_model=dict)
@app.head("/")
def home():
    return {"message": "Mermaid Diagram API is running!"}

@app.post("/generate_diagram")
def generate_diagram(request: DiagramRequest):
    return {"mermaid_syntax": generate_mermaid_syntax(request.user_input, request.diagram_type)}