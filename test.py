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







def GA1_9(question):
    json_pattern = r"\[.*?\]|\{.*?\}"
    sort_pattern = r"Sort this JSON array of objects by the value of the (\w+) field.*?tie, sort by the (\w+) field"

    json_match = re.search(json_pattern, question, re.DOTALL)
    sort_match = re.search(sort_pattern, question, re.DOTALL)

    if json_match and sort_match:
        try:
            json_data = json.loads(json_match.group())
            sort_keys = [sort_match.group(1), sort_match.group(2)]
            # print(sort_keys)
            if isinstance(json_data, list) and all(isinstance(d, dict) for d in json_data):
                sorted_data = sorted(json_data, key=lambda x: tuple(
                    x.get(k) for k in sort_keys))
                return json.dumps(sorted_data, separators=(",", ":"))
            else:
                return json.dumps(json_data, separators=(",", ":"))

        except json.JSONDecodeError:
            return None

    return None

question="""
Let's make sure you know how to use JSON. Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines.

[{"name":"Alice","age":42},{"name":"Bob","age":60},{"name":"Charlie","age":54},{"name":"David","age":6},{"name":"Emma","age":20},{"name":"Frank","age":7},{"name":"Grace","age":41},{"name":"Henry","age":10},{"name":"Ivy","age":29},{"name":"Jack","age":46},{"name":"Karen","age":85},{"name":"Liam","age":58},{"name":"Mary","age":25},{"name":"Nora","age":11},{"name":"Oscar","age":19},{"name":"Paul","age":49}]"""

print(GA1_9(question))