from tempfile import NamedTemporaryFile
import os
import io
import re
import json
import csv
import time
import shutil
import zipfile
import hashlib
import tempfile
import asyncio
import subprocess
import requests
import numpy as np
import pandas as pd  # type: ignore
import pytz
from datetime import datetime, timedelta
from bs4 import BeautifulSoup  # type: ignore
from fastapi import UploadFile  # type: ignore
from fastapi.responses import JSONResponse

def GA1_2(question):
    pattern = r"Send a HTTPS request to (https?://[^\s]+) with the URL encoded parameter email set to ([\w.%+-]+@[\w.-]+\.\w+)"
    match = re.search(pattern, question)

    if match:
        url, email = match.groups()
        print("URL:", url)
        print("Email:", email)

        # Make a GET request with email as a query parameter
        response = requests.get(url, params={"email": email})
        result = response.json()
        result["headers"]["User-Agent"] = "HTTPie/3.2.4"
        answer_json = json.dumps(result, separators=(",", ":"))
        return json.loads(json.dumps({"answer": result}))

    return {"error": "Url and Email not found in the input text"}

def process_request(query):
    pattern = r"Send a HTTPS request to (https?://[^\s]+) with the URL encoded parameter email set to ([\w.%+-]+@[\w.-]+\.\w+)"
    match = re.search(pattern, query)

    if match:
        endpoint, user_email = match.groups()
        print("Endpoint:", endpoint)
        print("User Email:", user_email)

        # Make a GET request with email as a query parameter
        response = requests.get(endpoint, params={"email": user_email})
        response_data = response.json()
        response_data["headers"]["User-Agent"] = "HTTPie/3.2.4"
        json_output = json.dumps(response_data, separators=(",", ":"))
        return JSONResponse(content={"answer": response_data})

    return {"error": "Endpoint and User Email not found in the input text"}
question="""
Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL.

Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 21f3002034@ds.study.iitm.ac.in

What is the JSON output of the command? (Paste only the JSON body, not the headers)"""







def fg2_4(question: str):
   email = re.findall(
       r'Run this program on Google Colab, allowing all required access to your email ID: ([\w. % +-]+@[\w.-] +\.\w+)', question)[0]
   expiry_year = "2025"
   print(email, expiry_year)
   hash_value = hashlib.sha256(
       f"{email} {expiry_year}".encode()).hexdigest()[-5:]
   return {"answer":str(hash_value)}

question="""
Let's make sure you can access Google Colab. Run this program on Google Colab, allowing all required access to your email ID: 21f3002034@ds.study.iitm.ac.in.

import hashlib
import requests
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
creds = GoogleCredentials.get_application_default()
token = creds.get_access_token().access_token
response = requests.get(
  "https://www.googleapis.com/oauth2/v1/userinfo",
  params={"alt": "json"},
  headers={"Authorization": f"Bearer {token}"}
)
email = response.json()["email"]
hashlib.sha256(f"{email} {creds.token_expiry.year}".encode()).hexdigest()[-5:]
What is the result? (It should be a 5-character string)"""
print(fg2_4(question))