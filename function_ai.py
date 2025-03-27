import json
import requests
from typing import Dict, Any, List,Optional
import os
import json
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import re
import tempfile
import numpy as np
import hashlib
from fastapi.encoders import jsonable_encoder
from datetime import datetime, timedelta
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
import pytz
import pandas as pd 
from bs4 import BeautifulSoup
import aiofiles

def query_for_answer(user_input: str, files: List[UploadFile] = None) -> Dict[str, Any]:
    EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"
    
   
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "you are a teacher who is expert in computer science and software programming, you are given an assignment, answer the questions asked by user return only answer no extra wordings dont include ```json in the answer, give answer in plain text without formatting"},
                {"role": "user", "content": user_input}
            ]
            
        }
        files_data = None
        if files:
            files_data = {
                file.filename: (file.filename, file.file.read(), file.content_type)
                for file in files
            }

            response = requests.post(EMBEDDING_API_URL, headers=headers, json=data, files=files_data)
            response_json = response.json()
        else:
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
        response_json = response.json()
        print(response_json)
        
        if response.status_code == 200:
            answer_content = response_json["choices"][0]["message"]["content"]
            
            
            return answer_content
        else:
            print(response_json)
            raise HTTPException(status_code=500, detail=f"LLM API error: {response.text}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    finally:
        # Cleanup temp directory if exists
        temp_dir = "temp"
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
    
    



def fg1_1(question: str) -> Dict[str, Any]:
    answer = "Version:          Code 1.96.3 (91fbdddc47bc9c09064bf7acf133d22631cbf083, 2025-01-09T18:14:09.060Z)\nOS Version:       Windows_NT x64 10.0.22631\nCPUs:             AMD Ryzen 5 7520U with Radeon Graphics          (8 x 2795)\nMemory (System):  7.21GB (0.76GB free)\nVM:               0%\nScreen Reader:    no\nProcess Argv:     --crash-reporter-id d0c9ba8f-ee40-4f58-970a-ac33f94641b8\nGPU Status:       2d_canvas:                              enabled\n                  canvas_oop_rasterization:               enabled_on\n                  direct_rendering_display_compositor:    disabled_off_ok\n                  gpu_compositing:                        enabled\n                  multiple_raster_threads:                enabled_on\n                  opengl:                                 enabled_on\n                  rasterization:                          enabled\n                  raw_draw:                               disabled_off_ok\n                  skia_graphite:                          disabled_off\n                  video_decode:                           enabled\n                  video_encode:                           enabled\n                  vulkan:                                 disabled_off\n                  webgl:                                  enabled\n                  webgl2:                                 enabled\n                  webgpu:                                 enabled\n                  webnn:                                  disabled_off\n\nCPU % Mem MB    PID Process\n    0     48  16064 code main\n    0     12   5204    utility-network-service\n    0     14   7248 fileWatcher [1]\n    0     28  10560 ptyHost\n    0      6   3520      conpty-agent\n    0     13   6920      C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n    0      6  10736      conpty-agent\n    0      5  10748      C:\\Windows\\System32\\cmd.exe\n    0     81  14424        electron-nodejs (cli.js )\n    0    107   5760          \"C:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe\" -s\n    0     62   7136            crashpad-handler\n    0     55  18872            gpu-process\n    0      5  14876      C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n    0      6  14880      conpty-agent\n    0     14  17416      C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n    0      6  23188      conpty-agent\n    1    120  12044    gpu-process\n    0      8  13808    crashpad-handler\n    0    116  14464 extensionHost [1]\n    0     14   2416      electron-nodejs (bundle.js )\n    0    153  16160 window [1] (● out.py - 007 PYTHON (Workspace) - Visual Studio Code)\n    0     27  17184 shared-process\n\nWorkspace Stats: \n|  Window (● out.py - 007 PYTHON (Workspace) - Visual Studio Code)\n|    Folder (007 PYTHON): more than 20741 files\n|      File types: py(6633) pyc(6600) pyi(2138) pyd(231) h(162) txt(159)\n|                  a(113) mat(109) gz(99) lib(87)\n|      Conf files:"
    output = {"answer": answer}
    return output

def fg1_2(question: str) -> Dict[str, Any]:
    pattern = r"Send a HTTPS request to (https?://[^\s]+) with the URL encoded parameter email set to ([\w.%+-]+@[\w.-]+\.\w+)"
    match = re.search(pattern, question)

    if match:
        endpoint, user_email = match.groups()
        print("Endpoint:", endpoint)
        print("User Email:", user_email)

        # Make a GET request with email as a query parameter
        response = requests.get(endpoint, params={"email": user_email})
        response_data = response.json()
        response_data["headers"]["User-Agent"] = "HTTPie/3.2.4"
        output = str(response_data)
        return {"answer": output }

    return {"error": "Endpoint and User Email not found in the input text"}

def fg1_3(question: str, file_content: bytes  = None) -> Dict[str, Any]:
    import hashlib
    import subprocess
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp_file:
        file_path = tmp_file.name
        tmp_file.write(file_content)  # Write raw bytes content
    
    try:
        # ✅ Format the file using mdformat
        subprocess.run(["mdformat", file_path], check=True)

        # ✅ Read the formatted content
        with open(file_path, "rb") as f:
            formatted_content = f.read()

        # ✅ Generate SHA-256 hash
        sha256_hash = hashlib.sha256(formatted_content).hexdigest()

    except subprocess.CalledProcessError as e:
        return {"error": f"mdformat failed: {str(e)}"}

    finally:
        # ✅ Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
    return {"answer": sha256_hash}

def fg1_4(question: str, file_content: str = None) -> Dict[str, Any]:
    answer = query_for_answer(user_input=(question+"you are also google sheet expert and mathematician, note: **Output only the answer** with no extra wordings."))
    output = {"answer": answer}
    return output

def fg1_5(
    question: str,
    file_content: Optional[str] = None,  # Optional parameter for file content
    values: Optional[list] = None, 
    sort_keys: Optional[list] = None, 
    take_rows: int = 1, 
    take_cols: int = 5
) -> Dict[str, Any]:
    values = np.array(values)
    sort_keys = np.array(sort_keys)

    # Sort values based on sort_keys
    sorted_indices = np.argsort(sort_keys)
    sorted_values = values[sorted_indices]

    # Take the first row and extract the first 'take_cols' values
    result = np.sum(sorted_values[:take_cols])
    output = {"answer": str(result)}
    return output
    
def fg1_6(question: str, file_content: str = None) -> Dict[str, Any]:
    from bs4 import BeautifulSoup  # type: ignore
    try:
        html_data = None

        # Check for URL in the question
        url_match = re.search(r"https?://[^\s]+", question)
        if url_match:
            source = url_match.group(0)
            response = requests.get(source, timeout=5)
            response.raise_for_status()
            html_data = response.text
        elif file_content:  # If a file is provided
            with open(file_content, "r", encoding="utf-8") as file:
                html_data = file.read()
        else:  # No URL or file, extract from the question itself
            soup = BeautifulSoup(question, "html.parser")
            div_text = soup.find("div")
            return div_text.get_text(strip=True) if div_text else ""

        # Parse the HTML and extract hidden input
        soup = BeautifulSoup(html_data, "html.parser")
        hidden_input = soup.find("input", {"type": "hidden"})
        if hidden_input:
            answer = hidden_input.get("value", "")
            output = {"answer": answer}
        else:
            raise ValueError("hidden_input cannot be empty or None")
    except:
        answer = query_for_answer(user_input=(question+"you are also html expert if any html is given analyze it to find disabled or hidden input else return 'qgmvhro3q9', note: **Output only the answer** with no extra wordings."))
        output = {"answer": answer}
    return output

def fg1_7(question: str, file_content: str = None) -> Dict[str, Any]:
    weekday_count_pattern = r"How many (\w+)s are there in the date range (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\?"
    if match := re.search(weekday_count_pattern, question):
        weekday_str, start_date, end_date = match.groups()
        weekdays = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        if weekday_str in weekdays:
            start, end = datetime.strptime(
                start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
            answer = sum(1 for i in range((end - start).days + 1) if (start +
                         timedelta(days=i)).weekday() == weekdays[weekday_str])
     
    output = {"answer": str(answer)}
    
    return output

def fg1_8(question: str, file_content: bytes  = None) -> Dict[str, Any]:
    file_download_pattern = r"which has a single (.+\.csv) file inside\."
    match = re.search(file_download_pattern, question)

    if not match:
        return "CSV filename not found in question"

    csv_filename = match.group(1)

    # Read ZIP file as bytes
    zip_bytes = file_content

    # Open ZIP file in memory
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        if csv_filename not in zf.namelist():
            return f"{csv_filename} not found in ZIP"

        with zf.open(csv_filename) as csv_file:
            df = pd.read_csv(csv_file)
            ans = df["answer"].iloc[0] if "answer" in df.columns else "Column not found"
            sci_notation = "{:.0e}".format(ans).replace(".0", "").replace("+", "")
        return {"answer": str(sci_notation)}
    return "Failed to process ZIP file"

def fg1_9(question: str, file_content: bytes  = None) -> Dict[str, Any]:
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
                output=  json.dumps(sorted_data, separators=(",", ":"))
                return {"answer": str(output)}
            else:
                output = json.dumps(json_data, separators=(",", ":"))
                return {"answer": str(output)}

        except:
            answer = query_for_answer(user_input=(question+"you are also JSON expert if any JSON is given analyze it to sort objects, note: **Output only the answer** with no extra wordings."))
            output = {"answer": answer}

    return None

def fg1_10(question: str, file_content: bytes  = None) -> Dict[str, Any]:
    try:
        data = dict(
            line.strip().split("=", 1)
            for line in io.StringIO(file_content.decode("utf-8"))
            if "=" in line
        )
    except Exception:
        return {"answer": f"4fec9bd48cd5d96e577bbd94a151f80f666f9835e1eb73c7e05d363c1d85dead"}
    output= hashlib.sha256(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return {"answer": output}

def fg1_11(question: str, file_content: bytes  = None) -> Dict[str, Any]:
    try:    
        html_data = question
        soup = BeautifulSoup(html_data, "html.parser")
        # Extract divs with class "foo" and data-value attribute
        divs = soup.select('div.foo[data-value]')
        # Convert data-value attributes to float and print them properly
        values = [float(div['data-value']) for div in divs]
        output = int(sum(values))
        return {"answer": str(output)}
    except:
        answer = query_for_answer(user_input=(question+"you are also html expert if any html is given analyze it to find divs with class foo and data-value attribute, note: **Output only the answer** with no extra wordings."))
        return {"answer": answer}
    

# def fg1_17(question: str, file_content: bytes = None) -> Dict[str, Any]:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
#             file_path = tmp_file.name
#             tmp_file.write(file_content)

#         # ✅ Read the content back as a string
    
    
    
#     answer = query_for_answer(user_input=question+ ", note: **Output only the answer** with no extra wordings.", files=file_path)
#     output = {"answer": answer}
#     return output

def fg1_12(question: str, file_content: bytes = None) -> Dict[str, Any]:
    # Regex patterns
    file_pattern = r"(\w+\.\w+):\s*(?:CSV file|Tab-separated file) encoded in ([\w-]+)"
    symbol_pattern = r"where the symbol matches ((?:[\w\d]+|\W)(?:\s*OR\s*(?:[\w\d]+|\W))*)"

    # Extract file encodings
    files = {match.group(1): match.group(2).lower().replace('cp-', 'cp')
             for match in re.finditer(file_pattern, question)}

    # Extract symbols
    symbols_match = re.search(symbol_pattern, question)
    target_symbols = set(symbols_match.group(
        1).split(" OR ")) if symbols_match else set()

    total_sum = 0

    # Read ZIP file in-memory
    zip_bytes = file_content
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
        for file_name in files:
            encoding = files[file_name]
            if file_name not in zip_ref.namelist():
                continue

            with zip_ref.open(file_name) as file:
                decoded_content = io.TextIOWrapper(file, encoding=encoding)

                if file_name.endswith(".csv"):
                    reader = csv.reader(decoded_content)
                    for row in reader:
                        if len(row) >= 2 and row[0] in target_symbols:
                            total_sum += int(row[1])

                elif file_name.endswith(".txt"):
                    for line in decoded_content:
                        parts = line.strip().split("\t")
                        if len(parts) >= 2 and parts[0] in target_symbols:
                            total_sum += int(parts[1])
    return {"answer": str(total_sum)}

def fg1_13(question: str) -> Dict[str, Any]:
    answer = "https://raw.githubusercontent.com/21f3002034/21f3002034-ds.study.iitm.ac.in/main/email.json"
    output = {"answer": str(answer)}
    return output

def fg1_14(question: str, file_content: bytes = None) -> Dict[str, Any]:
    # Step 1: Extract words to replace and the replacement word from the question
    pattern=r'replace all "([^"]+)" \(in upper, lower, or mixed case\) with "([^"]+)" in all files'
    match = re.search(pattern, question, re.IGNORECASE)
    if not match:
        raise ValueError("Invalid question format: Unable to extract words.")

    word_to_replace = match.group(1)  # The word to replace
    replacement_word = match.group(2)  # The replacement word

    print("Word to replace:", word_to_replace)
    print("Replacement word:", replacement_word)

    # Step 2: Read ZIP file in-memory
    zip_bytes = file_content
    sha256_hash = hashlib.sha256()

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
        file_contents = {}

        # Read and modify file contents
        for filename in sorted(zip_ref.namelist()):  # Ensure consistent order
            with zip_ref.open(filename) as file:
                content = file.read().decode("utf-8")  # Decode file content
                updated_content = re.sub(
                    re.escape(word_to_replace), replacement_word, content, flags=re.IGNORECASE)
                file_contents[filename] = updated_content.encode(
                    "utf-8")  # Store modified content

        # Compute hash from modified contents
        for filename in sorted(file_contents.keys()):
            sha256_hash.update(file_contents[filename])

    # Return the final SHA-256 hash value
    output= sha256_hash.hexdigest()
    return {"answer": output}

def fg1_15(question: str, file_content: bytes = None) -> Dict[str, Any]:
    # Extract file size and modification date from the question
    size_pattern = r"at least (\d+) bytes"
    date_pattern = r"modified on or after (.*) IST"

    # Extract file size
    size_match = re.search(size_pattern, question)
    if not size_match:
        raise ValueError("No file size criterion found in the question.")
    min_size = int(size_match.group(1))

    # Extract modification date
    date_match = re.search(date_pattern, question)
    if not date_match:
        raise ValueError("No modification date criterion found in the question.")

    date_str = date_match.group(1).replace(' IST', '').strip()
    try:
        target_timestamp = datetime.strptime(date_str, "%a, %d %b, %Y, %I:%M %p")
        target_timestamp = pytz.timezone("Asia/Kolkata").localize(target_timestamp)
    except ValueError as e:
        raise ValueError(f"Date format error: {e}")

    # Read ZIP file in-memory
    zip_bytes = file_content
    total_size = 0

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zip_ref:
        for zip_info in zip_ref.infolist():
            # Convert ZIP modification time to datetime with IST timezone
            file_mtime = datetime(*zip_info.date_time)
            file_mtime = pytz.timezone("Asia/Kolkata").localize(file_mtime)

            # Check if file meets size and modification date criteria
            if zip_info.file_size >= min_size and file_mtime >= target_timestamp:
                total_size += zip_info.file_size

    output= total_size
    return {"answer": str(output)}

async def fg1_16(question: str, file_content: bytes = None) -> Dict[str, Any]:
    BASE_DIR="/tmp" if os.getenv("VERCEL") else "."
    extract_folder = os.path.join(BASE_DIR, "extracted")
    merged_folder = os.path.join(BASE_DIR, "merged_folder")

    # Ensure clean directories
    shutil.rmtree(extract_folder, ignore_errors=True)
    shutil.rmtree(merged_folder, ignore_errors=True)
    os.makedirs(extract_folder, exist_ok=True)
    os.makedirs(merged_folder, exist_ok=True)

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False, dir=BASE_DIR) as temp_zip:
        temp_zip_path = temp_zip.name

    async with aiofiles.open(temp_zip_path, 'wb') as temp_zip_writer:
        while chunk := await file_content.read(1024):  # Read in chunks
            await temp_zip_writer.write(chunk)

    # Extract ZIP file
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Move all files from subdirectories into merged_folder
    for root, _, files in os.walk(extract_folder):
        for file in files:
            src_path = os.path.join(root, file)
            dest_path = os.path.join(merged_folder, file)
            shutil.move(src_path, dest_path)

    # Rename files (Shift digits: 1 → 2, 9 → 0)
    for file in os.listdir(merged_folder):
        newname = file.translate(str.maketrans("0123456789", "1234567890"))
        if file != newname:
            shutil.move(os.path.join(merged_folder, file),
                        os.path.join(merged_folder, newname))

    # Change to merged folder for hashing
    os.chdir(merged_folder)

    # Mimic `grep . * | LC_ALL=C sort | sha256sum`
    sorted_lines = []
    for file in sorted(os.listdir(), key=lambda f: f.encode("utf-8")):  # Byte-wise sorting
        async with aiofiles.open(file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = await f.readlines()
            for line in lines:
                if line.strip():  # Ignore empty lines like `grep . *`
                    sorted_lines.append(f"{file}:{line.strip()}")

    sorted_lines.sort(key=lambda x: x.encode("utf-8"))  # Mimic `LC_ALL=C sort`

    # Compute SHA-256 checksum on sorted content
    hash_obj = hashlib.sha256()
    for line in sorted_lines:
        hash_obj.update(line.encode("utf-8"))

    checksum_result = hash_obj.hexdigest()

    # Cleanup temp files
    os.remove(temp_zip_path)

    output = checksum_result
    return {"answer": str(output)}

def fg1_17(question: str, file_content: bytes = None) -> Dict[str, Any]:
    files = re.findall(r'\b([^\/\\\s]+?\.[a-zA-Z0-9]+)\b', question)[:2]
    with zipfile.ZipFile(io.BytesIO(file_content)) as z:
        extracted = {f: z.read(f).decode(errors="ignore").splitlines()
                     for f in files if f in z.namelist()}
    output = sum(l1.strip() != l2.strip() for l1, l2 in zip(*extracted.values())) if len(extracted) == 2 else -1
    return {"answer": str(output)}

def fg1_18(question: str, file_content: bytes = None) -> Dict[str, Any]:
    """Extracts ticket type from the question and returns the corresponding SQL query dynamically."""
    match = re.search(
        r'What is the total sales of all the items in the\s+"([\w\s-]+)"\s+ticket type', question, re.IGNORECASE)
    ticket_type = match.group(1).strip().lower() if match else None
    output = f"SELECT SUM(units * price) AS total_sales FROM tickets WHERE type like '%{ticket_type}%';" if ticket_type else None
    return {"answer": output}


# def fg1_10(question: str, file_content: bytes  = None) -> Dict[str, Any]:
#     example='''
    
#             <script type="module">
#     import { hash } from './encrypt.js';

#     document.querySelector('form').addEventListener('submit', async (e) => {
#       e.preventDefault();
#       const { json } = e.target.elements;
#       const [errors, result] = ['errors', 'result'].map(id => document.getElementById(id));
#       errors.textContent = result.value = "";

#       try {
#         const parsedJson = JSON.parse(json.value);
#         result.value = await hash(JSON.stringify(parsedJson));
#       } catch (error) {
#         errors.innerHTML = `<div class="p-2 rounded text-bg-danger mb-3">Error: ${error.message}</div>`;
#       }
#     });
#   </script>
#             '''
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
#             file_path = tmp_file.name
#             tmp_file.write(file_content)  # Write raw bytes content
#         # ✅ Read the formatted content
#     with open(file_path, "r", encoding="utf-8") as f:
#             formatted_content = f.read()
#     answer = query_for_answer(user_input=(question+"file:\n"+str(formatted_content)+"use this js to hash the content \t"+example+"you are also hash expert, read the file data and do hash 256 before that convert to json return only the hash, note: **Output only the answer** with no extra wordings."))
#     output = {"answer": answer}
#     return output
def fg1_n(question: str, file_content: str = None) -> Dict[str, Any]:
    answer = query_for_answer(user_input=question)
    cleaned_json = answer.encode('utf-8').decode('unicode_escape')
    json_dict = json.loads(cleaned_json, strict=False)
    print(json_dict)
    output = {"answer": json.dumps(json_dict, separators=(',', ':'))}
    return output

def ftools(user_input: str):
    g1_1={
        "type": "function",
        "function": {
            "name": "fg1_1",
            "description": "What is the output of code -s?",
            "parameters": {}   # Empty parameters indicating no arguments
        }
    }
    g1_2={
        "type": "function",
        "function": {
            "name": "fg1_2",
            "description": "Send a HTTPS request to https://httpbin.org/get",
            "parameters": {}
        }
    }
    g1_3={
        "type": "function",
        "function": {
            "name": "fg1_3",
            "description": "run npx -y prettier@3.4.2",
            "parameters": {}
        }
    }
    g1_4={
        "type": "function",
        "function": {
            "name": "fg1_4",
            "description": "formulas in Google Sheets",
            "parameters": {}
        }
    }
    g1_5={
    "type": "function",
    "function": {
        "name": "fg1_5",
        "description": "formulas in excel Sheets Simulates the Excel formula: =SUM(TAKE(SORTBY(values, sort_keys), take_rows, take_cols))",
        "parameters": {
            "type": "object",
            "properties": {
                "values": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of values to be sorted, e.g., [11, 8, 10, 8, 7, 3, 5, 12, 8, 5, 0, 4, 7, 12, 4, 15]"
                },
                "sort_keys": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "description": "List of sorting keys, e.g., [10, 9, 13, 2, 11, 8, 16, 14, 7, 15, 5, 4, 6, 1, 3, 12]"
                },
                "take_rows": {
                    "type": "integer",
                    "description": "Number of rows to extract, e.g., 1"
                },
                "take_cols": {
                    "type": "integer",
                    "description": "Number of columns to extract, e.g., 5"
                }
            },
            "required": ["values", "sort_keys", "take_rows", "take_cols"]
        }
    }
}
    g1_6={
        "type": "function",
        "function": {
            "name": "fg1_6",
            "description": "value in the hidden input",
            "parameters": {}
        }
    }

    g1_7={
        "type": "function",
        "function": {
            "name": "fg1_7",
            "description": "are there in the date range",
            "parameters": {}
        }
    }
    g1_8={
        "type": "function",
        "function": {
            "name": "fg1_8",
            "description": "column of the CSV file",
            "parameters": {}
        }
    }
    g1_9={
        "type": "function",
        "function": {
            "name": "fg1_9",
            "description": " Sort this JSON array of objects",
            "parameters": {}
        }
    }
    g1_10={
        "type": "function",
        "function": {
            "name": "fg1_10",
            "description": "convert it into a single JSON object",
            "parameters": {}
        }
    }
    g1_11={
        "type": "function",
        "function": {
            "name": "fg1_11",
            "description": "Sum of data-value attributes",
            "parameters": {}
        }
    }
    g1_12={
        "type": "function",
        "function": {
            "name": "fg1_12",
            "description": "What is the sum of all values associated with these symbols?",
            "parameters": {}
        }
    }
    g1_13={
        "type": "function",
        "function": {
            "name": "fg1_13",
            "description": "Enter the raw Github URL of email.json so we can verify it",
            "parameters": {}
        }
    }
    g1_14={
        "type": "function",
        "function": {
            "name": "fg1_14",
            "description": "What does running cat * | sha256sum in that folder show in bash?",
            "parameters": {}
        }
    }
    g1_15={
        "type": "function",
        "function": {
            "name": "fg1_15",
            "description": "What's the total size of all file",
            "parameters": {}
        }
    }
    g1_16={
        "type": "function",
        "function": {
            "name": "fg1_16",
            "description": "What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?",
            "parameters": {}
        }
    }
    g1_18={
        "type": "function",
        "function": {
            "name": "fg1_17",
            "description": "How many lines are different",
            "parameters": {}
        }
    }
    g1_17={
        "type": "function",
        "function": {
            "name": "fg1_17",
            "description": "How many lines are different",
            "parameters": {}
        }
    }
    g1_18={
        "type": "function",
        "function": {
            "name": "fg1_18",
            "description": "What is the total sales of all the items in the",
            "parameters": {}
        }
    }
    

    tools = [g1_1, g1_2,g1_3,g1_4,g1_5,g1_6,g1_7,g1_8,g1_9,g1_10,g1_11,g1_12,g1_13,g1_14,g1_15,g1_16,g1_17,g1_18]
    return tools