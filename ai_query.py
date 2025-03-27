import requests
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from function_ai import ftools
import os
import json
from fastapi.responses import JSONResponse


def query_gpt(user_input: str) -> Dict[str, Any]:    
    EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"
    tools = ftools(user_input)
    

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": user_input},
            {
                "role": "system",
                "content": """
                
        You are an assistant capable of executing various tasks. 
        You are an expert in identifying contextual matches from following functions, , even if the phrasing is different first analyze all and decide.
        Use the following functions for specific tasks:  
        
        - fg1_1: What is the output of code -s?
        - fg1_2: Send a HTTPS request to https://httpbin.org/get?
        - fg1_3: run npx -y prettier@3.4.2
        - fg1_4: formulas in Google Sheets
        - fg1_5: formulas in Excel
        - fg1_6: value in the hidden input
        - fg1_7: are there in the date range
        - fg1_8: the value in the "answer" column of the CSV file?
        - fg1_9: Sort this JSON array
        - fg1_10: convert it into a single JSON object
        - fg1_11: select elements using CSS selectors Sum of data-value attributes
        - fg1_17: How many lines are different
        - fg1_12: What is the sum of all values associated with these symbols?
        - fg1_13: Enter the raw Github URL of email\.json so we can verify it
        - fg1_14: What does running cat * | sha256sum in that folder show in bash?
        - fg1_15: Use ls with options to list all files in the folderWhat's the total size of all file
        - fg1_16: What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?
        - fg1_18: "What is the total sales of all the items in the"
        
        Always return relative paths for system directory locations.           
    
                """
            }
        ],
        "tools": tools,
        "tool_choice": "auto",

        
    }
    
    try:
        response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"GPT query failed: {str(e)}")
    
