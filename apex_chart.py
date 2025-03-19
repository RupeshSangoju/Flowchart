import requests
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

class ChartRequest(BaseModel):
    user_input: str
    chart_type: str

def generate_apexcharts_syntax(user_input, chart_type):
    api_url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    prompt = f"Convert the following data into an {chart_type} chart using ApexCharts syntax: {user_input}"
    
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700,
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

@app.post("/generate_chart")
def generate_chart(request: ChartRequest):
    return {"chart_syntax": generate_apexcharts_syntax(request.user_input, request.chart_type)}

if __name__ == "__main__":
    user_input = input("Enter your data or text: ")
    chart_type = input("Enter chart type (Line, Bar, Pie, Donut, Area, Radar, Scatter, Bubble, Heatmap, Mixed): ")
    print("\nGenerated ApexCharts Syntax:\n")
    print(generate_apexcharts_syntax(user_input, chart_type))
    uvicorn.run(app, host="0.0.0.0", port=8000)
