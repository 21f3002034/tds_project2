import requests
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from function_ai import ftools
import os
import json
from fastapi.responses import JSONResponse


async def query_gpt(user_input: str) -> Dict[str, Any]:    
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
        - fg1_18: There is a tickets table in a SQLite database that has columns type, What is the total sales of all the items in the Write SQL to calculate it.
        - fg2_1: Write documentation in Markdown for
        - fg2_2: compress it losslessly to an image that is less than 1,500 bytes. By losslessly, we mean that every pixel in the new image should be identical to the original image. Upload your losslessly compressed image
        - fg2_3: What is the GitHub Pages URL
        - fg2_4: Run this program on Google Colab What is the result? (It should be a 5-character string)
        - fg2_5: Create a new Google Colab notebook calculate the number of pixels What is the result? (It should be a number)
        - fg2_6: What is the Vercel URL
        - fg2_7: Trigger the action and make sure it is the most recent action.
        - fg2_8: What is the Docker image URL
        - fg2_9: What is the API URL endpoint for FastAPI /api
        - fg2_10: What is the ngrok URL
        - fg3_1: Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text
        - fg3_2: How many input tokens does it use up (Number of tokens)
        - fg3_3: What is the JSON body we should send to https://api.openai.com/v1/chat/completions for this
        - fg3_4: Write just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content (text and image URL)
        - fg3_5: Your task is to write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding
        - fg3_6: Your task is to write a Python function most_similar(embeddings)
        - fg3_7: What is the API URL endpoint for your implementation? It might look like: /similarity
        - fg3_8: What is the API URL endpoint for your implementation? It might look like: /execute
        - fg3_9: Write a prompt that will get the LLM to say Yes
        - fg4_1: What is the total number of ducks across players on page number ESPN Cricinfo's ODI batting stats
        - fg4_2: Utilize IMDb's advanced web search
        - fg4_3: Create an API endpoint (e.g., /api/outline) that accepts a country query parameter
        - fg4_4: What is the JSON weather forecast description for
        - fg4_5: What is the minimum latitude of the bounding box of the city
        - fg4_6: What is the link to the latest Hacker News post mentioning
        - fg4_7: Enter the date (ISO 8601, e.g., "2024-01-01T00:00:00Z") when the newest user joined GitHub
        - fg4_8: Trigger the workflow and wait for it to complete
        - fg4_9: Retrieve the PDF file containing the student marks table By automating the extraction and analysis
        - fg4_10: What is the markdown content of the PDF
        - fg5_1: Clean this Excel data,The total margin is defined as What is the total margin for transactions before, Clean this Excel data
        - fg5_2: data analyst at EduTrack Systems, your task is to process this text file How many unique students are there in the file 
        - fg5_3: What is the number of successful GET requests for pages under 
        - fg5_4: Across all requests under  web log entry 
        - fg5_5: Use phonetic clustering algorithm
        - fg5_6:  your task is to develop a program that will:, What is the total sales value?
        - fg5_7: How many times does DX appear as a key?
        - fg5_8: Write a DuckDB SQL query
        - fg5_9: What is the text of the transcript
        - fg5_10: Upload the reconstructed image        
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
    
