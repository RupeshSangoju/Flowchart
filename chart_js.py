import requests
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChartRequest(BaseModel):
    user_input: str
    chart_type: str

def generate_chartjs_syntax(user_input, chart_type):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        logger.error("PERPLEXITY_API_KEY not set")
        return "Error: PERPLEXITY_API_KEY not set in environment variables"
    
    api_url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = f"Convert the following data into a {chart_type} chart using Chart.js syntax: {user_input}"
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raises exception for 4xx/5xx errors
        data = response.json()
        result = data.get("choices", [{}])[0].get("message", {}).get("content", "Error: No response from API")
        logger.info(f"Generated Chart.js syntax: {result}")
        return result
    except requests.RequestException as e:
        logger.error(f"API error: {str(e)}")
        return f"Error: Failed to connect to Perplexity API - {str(e)}"

@app.get("/", response_model=dict)
@app.head("/")
def home():
    return {"message": "Chart.js API is running!"}

@app.post("/generate_chart")
def generate_chart(request: ChartRequest):
    logger.info(f"Received request: {request.user_input}, {request.chart_type}")
    try:
        result = generate_chartjs_syntax(request.user_input, request.chart_type)
        return {"chart_syntax": result}
    except Exception as e:
        logger.error(f"Error in generate_chart: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))