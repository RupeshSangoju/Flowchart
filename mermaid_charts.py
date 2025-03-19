import requests
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from transformers import BartForConditionalGeneration, BartTokenizer

load_dotenv()

app = FastAPI()

class DiagramRequest(BaseModel):
    user_input: str
    diagram_type: str

def summarize_text(text):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_mermaid_syntax(user_input, diagram_type):
    api_url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    summarized_text = summarize_text(user_input)
    print(f"\nSummarized Text:\n{summarized_text}\n")  # Print the summarized text
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
        "search_domain_filter": None,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": None,
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response from API")
    else:
        return f"Error: {response.status_code} - {response.text}"

@app.get("/")
def home():
    return {"message": "Mermaid Diagram API is running!"}


@app.post("/generate_diagram")
def generate_diagram(request: DiagramRequest):
    return {"mermaid_syntax": generate_mermaid_syntax(request.user_input, request.diagram_type)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use the Render-assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)
