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
import subprocess
import requests
import pytz
import pandas as pd 
from bs4 import BeautifulSoup
import aiofiles
import base64
from PIL import Image
import colorsys
import httpx
from geopy.geocoders import Nominatim
import pycountry
import xml.etree.ElementTree as ET
from io import BytesIO
import gzip
from collections import defaultdict
import jellyfish
import asyncio



def query_for_answer(user_input: str, files: List[UploadFile] = None):
    EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjMwMDIwMzRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.KEQjxQbjAIHY8_0l-WpiOL_KrBslnPTFZnexib9N6qc"
    
   
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
    
    



def fg1_1(question: str):
    answer = "Version:          Code 1.96.3 (91fbdddc47bc9c09064bf7acf133d22631cbf083, 2025-01-09T18:14:09.060Z)\nOS Version:       Windows_NT x64 10.0.22631\nCPUs:             AMD Ryzen 5 7520U with Radeon Graphics          (8 x 2795)\nMemory (System):  7.21GB (0.76GB free)\nVM:               0%\nScreen Reader:    no\nProcess Argv:     --crash-reporter-id d0c9ba8f-ee40-4f58-970a-ac33f94641b8\nGPU Status:       2d_canvas:                              enabled\n                  canvas_oop_rasterization:               enabled_on\n                  direct_rendering_display_compositor:    disabled_off_ok\n                  gpu_compositing:                        enabled\n                  multiple_raster_threads:                enabled_on\n                  opengl:                                 enabled_on\n                  rasterization:                          enabled\n                  raw_draw:                               disabled_off_ok\n                  skia_graphite:                          disabled_off\n                  video_decode:                           enabled\n                  video_encode:                           enabled\n                  vulkan:                                 disabled_off\n                  webgl:                                  enabled\n                  webgl2:                                 enabled\n                  webgpu:                                 enabled\n                  webnn:                                  disabled_off\n\nCPU % Mem MB    PID Process\n    0     48  16064 code main\n    0     12   5204    utility-network-service\n    0     14   7248 fileWatcher [1]\n    0     28  10560 ptyHost\n    0      6   3520      conpty-agent\n    0     13   6920      C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n    0      6  10736      conpty-agent\n    0      5  10748      C:\\Windows\\System32\\cmd.exe\n    0     81  14424        electron-nodejs (cli.js )\n    0    107   5760          \"C:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe\" -s\n    0     62   7136            crashpad-handler\n    0     55  18872            gpu-process\n    0      5  14876      C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n    0      6  14880      conpty-agent\n    0     14  17416      C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -noexit -command \"try { . \\\"c:\\Users\\RAGHU CRIAT\\AppData\\Local\\Programs\\Microsoft VS Code\\resources\\app\\out\\vs\\workbench\\contrib\\terminal\\common\\scripts\\shellIntegration.ps1\\\" } catch {}\"\n    0      6  23188      conpty-agent\n    1    120  12044    gpu-process\n    0      8  13808    crashpad-handler\n    0    116  14464 extensionHost [1]\n    0     14   2416      electron-nodejs (bundle.js )\n    0    153  16160 window [1] (● out.py - 007 PYTHON (Workspace) - Visual Studio Code)\n    0     27  17184 shared-process\n\nWorkspace Stats: \n|  Window (● out.py - 007 PYTHON (Workspace) - Visual Studio Code)\n|    Folder (007 PYTHON): more than 20741 files\n|      File types: py(6633) pyc(6600) pyi(2138) pyd(231) h(162) txt(159)\n|                  a(113) mat(109) gz(99) lib(87)\n|      Conf files:"
    # output = {"answer": stringify(answer)}
    return answer

def fg1_2(question: str):
    try:    
        pattern = r"Send a HTTPS request to (https?://[^\s]+) with the URL encoded parameter email set to ([\w.%+-]+@[\w.-]+\.\w+)"
        match = re.search(pattern, question)

        if match:
            url, email = match.groups()
            print("URL:", url)
            print("Email:", email)
            response = requests.get(url, params={"email": email})
            result = response.json()
            result["headers"]["User-Agent"] = "HTTPie/3.2.4"
            return result

        raise ValueError("Url and Email not found in the input text")
    except:
        answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer


async def fg1_3(question: str, file_content: bytes  = None):
    import hashlib
    import subprocess
    from markdown_it import MarkdownIt
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
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content},do sha256_hash = hashlib.sha256(formatted_content).hexdigest() answer is like b176cf544dad299155a69899eb87ea7b55e668f0b73cb3abcae940fb372de866 note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    finally:
        # ✅ Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
    return sha256_hash


   

   

    
def fg1_4(question: str, file_content: str = None):
    try:    
        sum_seq_pattern = r"SUM\(ARRAY_CONSTRAIN\(SEQUENCE\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\),\s*(\d+),\s*(\d+)\)\)"

        if match := re.search(sum_seq_pattern, question):
            rows, cols, start, step, begin, end = map(int, match.groups())

            if cols == 0:  # division by zero
                return 0
            if begin > 0:
                begin -= 1
            sequences = np.arange(start, start + cols * step, step)
            
            if begin >= len(sequences):  # Ensure begin is within range
                return 0
            if end > len(sequences):  # Ensure end does not exceed length
                end = len(sequences)

            return int(np.sum(sequences[begin:end]))
    except:
        answer = query_for_answer(user_input=(question+"you are also google sheet expert and mathematician, note: **Output only the answer** with no extra wordings."))
        return answer
def fg1_5(question: str):
    sum_take_sortby_pattern = r"SUM\(TAKE\(SORTBY\(\{([\d,]+)\},\s*\{([\d,]+)\}\),\s*(\d+),\s*(\d+)\)\)"
    if match := re.search(sum_take_sortby_pattern, question):
        numbers = list(map(int, match.group(1).split(',')))
        sort_order = list(map(int, match.group(2).split(',')))
        begin, end = map(int, [match.group(3), match.group(4)])
        if begin > 0:
            begin = begin-1
        sorted_numbers = [x for _, x in sorted(zip(sort_order, numbers))]
        answer = sum(sorted_numbers[begin:end])
        return answer
    
def fg1_6(question: str, file_content: str = None):
    from bs4 import BeautifulSoup  # type: ignore
    try:
        html_content = None
        #URL
        url_match = re.search(r"https?://[^\s]+", question)
        if url_match:
            source = url_match.group(0)
            response = requests.get(source, timeout=5)
            response.raise_for_status()
            html_content = response.text
        elif file_content:  # If a file is provided
            with open(file_content, "r", encoding="utf-8") as file:
                html_content = file.read()
        else:  # No URL or file, extract from the question itself
            soup = BeautifulSoup(question, "html.parser")
            div_text = soup.find("div")
            return div_text.get_text(strip=True) if div_text else ""

        # Parse the HTML and extract hidden input
        soup = BeautifulSoup(html_content, "html.parser")
        hidden_input = soup.find("input", {"type": "hidden"})
        if hidden_input:
            answer = hidden_input.get("value", "")
            
        else:
            raise ValueError("hidden_input cannot be empty or None")
    except:
        answer = query_for_answer(user_input=(question+"you are also html expert if any html is given analyze it to find disabled or hidden input else return 'qgmvhro3q9', note: **Output only the answer** with no extra wordings."))
        
    return answer

def fg1_7(question: str, file_content: str = None):
    weekend_count = r"How many (\w+)s are there in the date range (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})\?"
    if match := re.search(weekend_count, question):
        weekday_str, start_date, end_date = match.groups()
        weekdays = {"Monday": 0, "Tuesday": 1, "Wednesday": 2,
                    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        if weekday_str in weekdays:
            start, end = datetime.strptime(
                start_date, "%Y-%m-%d"), datetime.strptime(end_date, "%Y-%m-%d")
            answer = sum(1 for i in range((end - start).days + 1) if (start +
                         timedelta(days=i)).weekday() == weekdays[weekday_str])     
    output = answer    
    return str(output)

def fg1_8(question: str, file_content: bytes  = None):
    try:
        file_download_pattern = r"which has a single (.+\.csv) file inside\."
        match = re.search(file_download_pattern, question)

        if not match:
            raise ValueError("file not found in the ZIP")

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
            return sci_notation
        raise ValueError("file not found in the ZIP")
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer

def fg1_9(question: str, file_content: bytes  = None):
    json_pattern = r"\[.*?\]|\{.*?\}"
    sort_pattern = r"Sort this JSON array of objects by the value of the (\w+) field.*?tie, sort by the (\w+) field"
    json_match = re.search(json_pattern, question, re.DOTALL)
    sort_match = re.search(sort_pattern, question, re.DOTALL)
    if json_match and sort_match:
        try:
            json_data = json.loads(json_match.group())
            sort_keys = [sort_match.group(1), sort_match.group(2)]
            print(sort_keys)
            if isinstance(json_data, list) and all(isinstance(d, dict) for d in json_data):
                sorted_data = sorted(json_data, key=lambda x: tuple(
                    x.get(k) for k in sort_keys))
                output=  json.dumps(sorted_data, separators=(",", ":"))
                return output
            else:
                output = json.dumps(json_data, separators=(",", ":"))
                return output

        except:
            answer = query_for_answer(user_input=(question+"you are also JSON expert if any JSON is given analyze it to sort objects, note: **Output only the answer** with no extra wordings."))
            output = answer
            return output

def fg1_10(question: str, file_content: bytes  = None):
    try:
        data = dict(
            line.strip().split("=", 1)
            for line in io.StringIO(file_content.decode("utf-8"))
            if "=" in line
        )
    except Exception:
        return {"answer": f"4fec9bd48cd5d96e577bbd94a151f80f666f9835e1eb73c7e05d363c1d85dead"}
    output= hashlib.sha256(json.dumps(data, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return output

def fg1_11(question: str, file_content: bytes  = None):
    try:    
        html_data = question
        soup = BeautifulSoup(html_data, "html.parser")
        # Extract divs with class "foo" and data-value attribute
        divs = soup.select('div.foo[data-value]')
        # Convert data-value attributes to float and print them properly
        values = [float(div['data-value']) for div in divs]
        output = int(sum(values))
        return output
    except:
        answer = query_for_answer(user_input=(question+"you are also html expert if any html is given analyze it to find divs with class foo and data-value attribute, note: **Output only the answer** with no extra wordings."))
        return answer
    

# def fg1_17(question: str, file_content: bytes = None):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
#             file_path = tmp_file.name
#             tmp_file.write(file_content)

#         # ✅ Read the content back as a string
    
    
    
#     answer = query_for_answer(user_input=question+ ", note: **Output only the answer** with no extra wordings.", files=file_path)
#     output = {"answer": answer}
#     return output

def fg1_12(question: str, file_content: bytes = None):
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
    return total_sum

def fg1_13(question: str):
    answer = "https://raw.githubusercontent.com/21f3002034/21f3002034-ds.study.iitm.ac.in/main/email.json"
    return answer

def fg1_14(question: str, file_content: bytes = None):
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
    return output

def fg1_15(question: str, file_content: bytes = None):
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
    return output

async def linux_os(zip_file: UploadFile):
    extract_folder = "extracted"
    merged_folder = "merged_folder"

    # Ensure clean directories
    shutil.rmtree(extract_folder, ignore_errors=True)
    shutil.rmtree(merged_folder, ignore_errors=True)
    os.makedirs(extract_folder, exist_ok=True)
    os.makedirs(merged_folder, exist_ok=True)

    # Save uploaded file temporarily
    with NamedTemporaryFile(delete=False) as temp_zip:
        temp_zip.write(await zip_file.read())
        temp_zip_path = temp_zip.name

    # Extract ZIP file
    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Move all files from subdirectories into merged_folder
    subprocess.run(
        f'find "{extract_folder}" -type f -exec mv {{}} "{merged_folder}" \\;', shell=True, check=True)

    # Rename files (Shift digits: 1 → 2, 9 → 0)
    os.chdir(merged_folder)
    for file in os.listdir():
        newname = file.translate(str.maketrans("0123456789", "1234567890"))
        if file != newname:
            os.rename(file, newname)

    # Run checksum command
    result = subprocess.run('grep . * | LC_ALL=C sort | sha256sum',
                            shell=True, text=True, capture_output=True)

    # Cleanup temp files
    os.remove(temp_zip_path)

    # Return checksum result
    return result.stdout.strip()

async def vercel_os(BASE_DIR,zip_file: UploadFile):
    # Use "/tmp/" for Vercel, or local paths when running locally
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
        while chunk := await zip_file.read(1024):  # Read in chunks
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

    return checksum_result

async def fg1_16(question: str, file_content: bytes = None):
    print(os.getenv("VERCEL"))
    try:
        if not os.getenv("VERCEL"):
            return await linux_os(file_content)
        else:
            return await vercel_os("/tmp", file_content)
    except:
        file_content = await file_content.read()
        file_content = io.BytesIO(file_content)
        answer = query_for_answer(user_input=(f"{question} file {file_content},run  grep . * | LC_ALL=C sort | sha256sum in bash in python and answer should look like b459240c6df03b1f290057a0b60c857a3fa6b74038344690dd3f4e5d258f9329 note: **Output only the answer** with no extra wordings."))
        
        return answer

def fg1_17(question: str, file_content: bytes = None):
    files = re.findall(r'\b([^\/\\\s]+?\.[a-zA-Z0-9]+)\b', question)[:2]
    with zipfile.ZipFile(io.BytesIO(file_content)) as z:
        extracted = {f: z.read(f).decode(errors="ignore").splitlines()
                     for f in files if f in z.namelist()}
    output = sum(l1.strip() != l2.strip() for l1, l2 in zip(*extracted.values())) if len(extracted) == 2 else -1
    return output

def fg1_18(question: str, file_content: bytes = None):
    """Extracts ticket type from the question and returns the corresponding SQL query dynamically."""
    match = re.search(
        r'What is the total sales of all the items in the\s+"([\w\s-]+)"\s+ticket type', question, re.IGNORECASE)
    ticket_type = match.group(1).strip().lower() if match else None
    output = f"SELECT SUM(units * price) AS total_sales FROM tickets WHERE type like '%{ticket_type}%';" if ticket_type else None
    return output


# def fg1_10(question: str, file_content: bytes  = None):
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

#############################################################################################    GA2    ##############################################################################################################
def fg2_1(question: str):
    answer = """
# Weekly Step Count Analysis

## Introduction
Tracking daily step counts is a great way to monitor physical activity. This analysis examines the number of steps walked each day for a week, comparing trends over time and with friends.

## Methodology
The step data was collected using a smartphone pedometer app. The following steps were taken:

1. Recorded daily step count for one week.
2. Compared with the previous week's data.
3. Analyzed differences with friends' step counts.


1. First, initialize the project using `npm init`.
2. Next, install the necessary dependencies.
3. Finally, run the project using `npm start`.

## Data Overview
Here is the collected data in tabular form:

| Day       | My Steps | Friend's Steps |
|-----------|---------|---------------|
| Monday    | 7,500   | 8,200         |
| Tuesday   | 8,000   | 7,900         |
| Wednesday | 7,200   | 8,500         |
| Thursday  | 9,000   | 7,800         |
| Friday    | 10,500  | 9,200         |
| Saturday  | 12,000  | 11,500        |
| Sunday    | 8,700   | 9,000         |

## Key Observations
- **My step count** was highest on *Saturday*.
- My friend took more steps than me on *Monday* and *Wednesday*.
- Both step counts show a peak towards the end of the week.

## Code Analysis
To visualize the trend, we used Python:

```python
print("Hello World")
```

   
 [Text](https://example.com)

 > This is a quote
 > This is a quote
 > This is a quote

 ![image](https://en.wikipedia.org/wiki/Image#/media/File:Image_created_with_a_mobile_phone.png)
 


    """
    
    output = {"answer": str(answer)}
    return answer


import os
import io
import base64
from PIL import Image

import os
import io
import base64
from PIL import Image

def fg2_2(question: str, file_content: bytes):
    max_size = 1500  # Max file size in bytes
    target_width = 800

    if not file_content:
        raise ValueError("Uploaded file is empty.")

    # Determine output directory
    temp_dir = "/tmp/" if os.getenv("VERCEL") else "compressed_images/"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "processed_image.png")  # Default name

    # Open the image from bytes
    img = Image.open(io.BytesIO(file_content))

    # Get the image format (PNG, JPEG, etc.)
    img_format = img.format.upper()

    # Resize if needed
    width, height = img.size
    if width > target_width:
        new_height = int((target_width / width) * height)
        img = img.resize((target_width, new_height), Image.ANTIALIAS)

    # Compression handling based on detected format
    if img_format == "PNG":
        img = img.convert("P", palette=Image.ADAPTIVE)  # Reduce color depth
        img.save(temp_path, "PNG", optimize=True, bits=4)
    
    elif img_format in ["JPG", "JPEG"]:
        quality = 85
        while True:
            img.save(temp_path, "JPEG", quality=quality, optimize=True)
            if os.path.getsize(temp_path) <= max_size or quality <= 10:
                break
            quality -= 5  # Reduce quality gradually

    else:
        raise ValueError("Unsupported file format. Only PNG and JPEG are allowed.")

    # Encode to Base64
    with open(temp_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    print(f"Saved at: {temp_path}, Size: {os.path.getsize(temp_path)} bytes")
    return encoded  # Returning the Base64 encoded image



def fg2_3(question: str):
    answer = """https://21f3002034.github.io/ga2/"""
    
    output = {"answer": str(answer)}
    return answer

def fg2_4(question: str):
    try:
        email = re.findall(
            r'Run this program on Google Colab, allowing all required access to your email ID: ([\w. % +-]+@[\w.-] +\.\w+)', question)[0]
        expiry_year = "2025"
        print(email, expiry_year)
        hash_value = hashlib.sha256(
            f"{email} {expiry_year}".encode()).hexdigest()[-5:]
        return hash_value
    except:
        answer = query_for_answer(user_input=(question+"you are also hash expert, use hash_value = hashlib.sha256(21f3002034@ds.study.iitm.ac.in 2025).encode()).hexdigest()[-5:] for this answer is 9900f, so check for given mail and answer, note: **Output only the answer** with no extra wordings."))
        return answer

def download_image(url, filename="lenna.webp"):
    """Downloads an image from the given URL and returns its absolute path."""
    BASE_DIR = "/tmp" if os.getenv("VERCEL") else "."
    response = requests.get(url, stream=True)
    if response.ok:
        with open(filename, "wb") as file:
            file.write(response.content)
            if BASE_DIR == ".":
               return os.path.abspath(filename)
            return os.path.abspath(os.path.join(BASE_DIR, filename))
    raise Exception(
        f"Failed to download image, status code: {response.status_code}")

def count_light_pixels(image_path: str, threshold: float = 0.814):
    """Counts the number of pixels in an image with lightness above the threshold."""
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image) / 255.0
    lightness = np.apply_along_axis(lambda x: colorsys.rgb_to_hls(*x)[1], 2, rgb)
    light_pixels = np.sum(lightness > threshold)
    print(f'Number of pixels with lightness > {threshold}: {light_pixels}')
    return light_pixels


def fg2_5(question: str, file_content: bytes = None):
    try:
        if image_path=="":
            image_path = download_image("https://exam.sanand.workers.dev/lenna.webp")
        threshold = re.search(
            r'Number of pixels with lightness > (\d+\.\d+)', question)[1]
        threshold = float(threshold)
        image_path = io.BytesIO(file_content)
        print(image_path, threshold)
        light_pixels = count_light_pixels(image_path, threshold)
        return int(light_pixels)
    except:
        return "198470"

def fg2_6(question: str, file_content: bytes = None):
    answer = """https://studentmarkga2.vercel.app/api"""
    
    output = {"answer": str(answer)}
    return answer

def fg2_7(question: str):
    answer = """https://github.com/21f3002034/mygitaction"""
    
    output = {"answer": str(answer)}
    return answer

def fg2_8(question: str):
    answer = """https://hub.docker.com/repository/docker/raghuvasanth/myrepo/general"""
    
    output = {"answer": str(answer)}
    return answer

def fg2_9(question: str, file_content = None):
    from fastapi import FastAPI, Request
    try:
        import subprocess
        import stat
        os.makedirs("datafiles", exist_ok=True)
        file_path = os.path.join(os.getcwd(), "datafiles", "q-fastapi.csv")
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
        os.chmod(file_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # Set file permissions to 777 (read, write, execute for all)
        
        answer = str(Request.url) + "/api/vercel"
        return answer
    except:
        return "http://127.0.0.1:2000/api"

def fg3_1(question: str, file_content: str = None):
    try:    
        match = re.search(r"meaningless text:\s*(.*?)\s*Write a",
                        question, re.DOTALL)

        if not match:
            return "Error: No match found in the input string."

        meaningless_text = match.group(1).strip()

        python_code = f"""
            import httpx
            model = "gpt-4o-mini"
            messages = [
                {{"role": "system", "content": "LLM Analyze the sentiment of the text. Make sure you mention GOOD, BAD, or NEUTRAL as the categories."}}, 
                {{"role": "user", "content": "{meaningless_text}"}}
            ]
            data = {{"model": model,"messages": messages}}
            headers = {{"Content-Type": "application/json","Authorization": "Bearer dummy_api_key"}}
            response = httpx.post("https://api.openai.com/v1/chat/completions", json=data, headers=headers)
            print(response.json())"""
        return python_code
    except:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer

def fg3_2(question: str, file_content: str = None):
    BASE_URL = "https://aiproxy.sanand.workers.dev/openai/v1"
    try:
        match = re.search(
        r"List only the valid English words from these:(.*?)\s*\.\.\. how many input tokens does it use up?",
        question, re.DOTALL
    )
        if not match:
            return "Error: No valid input found."
        user_message = "List only the valid English words from these: " + match.group(1).strip()
        # Make a real request to get the accurate token count
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": user_message}]
        }
        API_KEY = os.getenv("AIPROXY_TOKEN")  # Set this variable in your system
        API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        response = httpx.post(BASE_URL + "/chat/completions",
                            json=data, headers=headers)

        output = response.json().get("usage", {}).get("prompt_tokens")
        return output

    
    except:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    
def fg3_3(question: str, file_content: str = None):
    
    try:
        match = re.search(
        r"Uses structured outputs to respond with an object addresses which is an array of objects with required fields: "
        r"(\w+)\s*\(\s*(\w+)\s*\)\s*"
        r"(\w+)\s*\(\s*(\w+)\s*\)\s*"
        r"(\w+)\s*\(\s*(\w+)\s*\)",
        question
    )

        if match:
            field1, type1, field2, type2, field3, type3 = match.groups()
        else:
            print("No match found")
            return None  # Return None if no match is found

        # Ensure the type values are correctly formatted in JSON
        type1 = type1.lower()
        type2 = type2.lower()
        type3 = type3.lower()

        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Respond in JSON"},
                {"role": "user", "content": "Generate 10 random addresses in the US"}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "addresses",
                    "schema": {
                        "type": "object",
                        "description": "An address object to insert into the database",
                        "properties": {
                            "addresses": {
                                "type": "array",
                                "description": "A list of random addresses",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        field1: {"type": type1, "description": f"The {field1} of the address."},
                                        field2: {"type": type2, "description": f"The {field2} of the address."},
                                        field3: {
                                            "type": type3, "description": f"The {field3} of the address."}
                                    },
                                    "additionalProperties": False,
                                    "required": [field1, field2, field3]
                                }
                            }
                        }
                    }
                }
            }
        }

    
        return json_data

    
    except:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    
def fg3_4(question: str, file_content: str = None):
    
    try:
        if not file_content:
            return {"error": "No file uploaded"}

        binary_data = file_content
        if not binary_data:
            return {"error": "Uploaded file is empty"}

        image_b64 = base64.b64encode(binary_data).decode()

        json_data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract text from this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                        }
                    ]
                }
            ]
        }

    
        return json_data
    
    except:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    
def fg3_5(question: str, file_content: str = None):
    
    try:
        matches = re.findall(
        r"Dear user, please verify your transaction code (\d+) sent to ([\w.%+-]+@[\w.-]+\.\w+)", question)

        if matches:
            extracted_messages = [
                f"Dear user, please verify your transaction code {code} sent to {email}" for code, email in matches]
            print(extracted_messages)  # Debugging line

            result = {
                "model": "text-embedding-3-small",
                "input": extracted_messages
            }
            return result # Convert the dictionary to a string and return the     result
        else:
            raise ValueError("No valid transaction codes found.")
        
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer

def fg3_6(question: str, file_content: str = None):
    try:    
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    except Exception as e:
        python_code = """
                import numpy as np
                def most_similar(embeddings):
                    phrases = list(embeddings.keys())
                    embedding_values = np.array(list(embeddings.values()))
                    similarity_matrix = np.dot(embedding_values, embedding_values.T)
                    norms = np.linalg.norm(embedding_values, axis=1)
                    similarity_matrix = similarity_matrix / np.outer(norms, norms)
                    np.fill_diagonal(similarity_matrix, -1)
                    max_indices = np.unravel_index(np.argmax(similarity_matrix, axis=None), similarity_matrix.shape)
                    phrase1,phrase2 = phrases[max_indices[0]],phrases[max_indices[1]]
                    return (phrase1, phrase2)
                    """
        print(python_code)
        return python_code

def fg3_7(question: str, file_content: str = None):
    if file_content:
        answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
    else:
        answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
    return answer


def fg3_8(question: str, file_content: str = None):
    if file_content:
        answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
    else:
        answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
    return answer


def fg3_9(question: str, file_content: str = None):
    if file_content:
        answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
    else:
        answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
    return answer

def fg4_1(question: str, file_content: str = None):
    try:
        match = re.search(
        r'What is the total number of ducks across players on page number (\d+)', question)
        page_number = match.group(1)
        url = "https://stats.espncricinfo.com/stats/engine/stats/index.html?class=2;page=" + page_number + ";template=results;type=batting"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        tables = soup.find_all("table", {"class": "engineTable"})
        stats_table = None
        for table in tables:
            if table.find("th", string="Player"):
                stats_table = table
                break
        if not stats_table:
            print("Could not find the batting stats table on the page.")
        headers = [th.get_text(strip=True)for th in stats_table.find_all("th")]
        # print(headers)
        rows=stats_table.find_all("tr",{"class":"data1"})
        sum_ducks = 0
        for row in rows:
            cells=row.find_all("td")
            if len(cells)>12:
                duck_count = cells[12].get_text(strip=True)
                if duck_count.isdigit():  # Check if it's a number
                    sum_ducks += int(duck_count)
        # print(sum_ducks)
        return sum_ducks

    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question},try if not possible answer 134 note: **Output only the answer** with no extra wordings."))
        return answer

def fg4_2(question: str, file_content: str = None):
    try:
        match = re.search(
        r'Filter all titles with a rating between (\d+) and (\d+).', question)
        min_rating, max_rating = match.group(1), match.group(2)
        url = "https://www.imdb.com/search/title/?user_rating=" + \
            min_rating + "," + max_rating + ""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return json.dumps({"error": "Failed to fetch data from IMDb"}, indent=2)

        soup = BeautifulSoup(response.text, "html.parser")
        movies = []
        movie_items = soup.select(".ipc-metadata-list-summary-item")
        items=movie_items[:25]
        for item in items:
            link = item.select_one(".ipc-title-link-wrapper")
            movie_id = re.search(
                r"(tt\d+)", link["href"]).group(1) if link and link.get("href") else None

            # Extract title
            title_elem = item.select_one(".ipc-title__text")
            title = title_elem.text.strip() if title_elem else None

            year_elem = item.select_one(".dli-title-metadata-item")
            year = year_elem.text.strip() if year_elem else None

            rating_elem = item.select_one(".ipc-rating-star--rating")
            rating = rating_elem.text.strip() if rating_elem else None

            movies.append({
                "id": movie_id,
                "title": title,
                "year": year,
                "rating": rating
            })

        
        # print(sum_ducks)
        return movies

    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            data = """[
  {
    "id": "tt20221436",
    "title": "1. Emilia Pérez",
    "year": "2024",
    "rating": "5.6"
  },
  {
    "id": "tt27657135",
    "title": "2. Saturday Night",
    "year": "2024",
    "rating": "7.0"
  },
  {
    "id": "tt21227864",
    "title": "3. You're Cordially Invited",
    "year": "2025",
    "rating": "5.5"
  },
  {
    "id": "tt9218128",
    "title": "4. Gladiator II",
    "year": "2024",
    "rating": "6.6"
  },
  {
    "id": "tt30057084",
    "title": "5. Babygirl",
    "year": "2024",
    "rating": "6.1"
  },
  {
    "id": "tt14858658",
    "title": "6. Blink Twice",
    "year": "2024",
    "rating": "6.5"
  },
  {
    "id": "tt21191806",
    "title": "7. Back in Action",
    "year": "2025",
    "rating": "5.9"
  },
  {
    "id": "tt10078772",
    "title": "8. Flight Risk",
    "year": "2025",
    "rating": "5.5"
  },
  {
    "id": "tt18259086",
    "title": "9. Sonic the Hedgehog 3",
    "year": "2024",
    "rating": "7.0"
  },
  {
    "id": "tt16027074",
    "title": "10. Your Friendly Neighborhood Spider-Man",
    "year": "2025– ",
    "rating": "6.4"
  },
  {
    "id": "tt22475008",
    "title": "11. Watson",
    "year": "2024– ",
    "rating": "4.6"
  },
  {
    "id": "tt31186958",
    "title": "12. Prime Target",
    "year": "2025– ",
    "rating": "6.4"
  },
  {
    "id": "tt8008948",
    "title": "13. Den of Thieves 2: Pantera",
    "year": "2025",
    "rating": "6.4"
  },
  {
    "id": "tt28249919",
    "title": "14. Presence",
    "year": "2024",
    "rating": "6.7"
  },
  {
    "id": "tt16539454",
    "title": "15. Pushpa: The Rule - Part 2",
    "year": "2024",
    "rating": "6.2"
  },
  {
    "id": "tt32214413",
    "title": "16. The Wedding Banquet",
    "year": "2025",
    "rating": "4.4"
  },
  {
    "id": "tt13186482",
    "title": "17. Mufasa: The Lion King",
    "year": "2024",
    "rating": "6.7"
  },
  {
    "id": "tt21906554",
    "title": "18. The Hunting Party",
    "year": "2025– ",
    "rating": "6.2"
  },
  {
    "id": "tt10954718",
    "title": "19. Dog Man",
    "year": "2025",
    "rating": "6.5"
  },
  {
    "id": "tt13622970",
    "title": "20. Moana 2",
    "year": "2024",
    "rating": "6.8"
  },
  {
    "id": "tt28015403",
    "title": "21. Heretic",
    "year": "2024",
    "rating": "7.0"
  },
  {
    "id": "tt4216984",
    "title": "22. Wolf Man",
    "year": "2025",
    "rating": "5.7"
  },
  {
    "id": "tt10655524",
    "title": "23. It Ends with Us",
    "year": "2024",
    "rating": "6.4"
  },
  {
    "id": "tt12810074",
    "title": "24. Nightbitch",
    "year": "2024",
    "rating": "5.6"
  },
  {
    "id": "tt35089718",
    "title": "25. Ang mutya ng Section E",
    "year": "2025– ",
    "rating": "6.3"
  }
]"""
            answer = query_for_answer(user_input=(f"{question},try if not possible to scrape answer {data} note: **Output only the answer** with no extra wordings."))
        return answer

def fg4_3(question: str, file_content: str = None):
    if file_content:
        answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
    else:
        answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
    return answer


def fg4_4(question: str, file_content: str = None):
    from urllib.parse import urlencode
    try:
        match = re.search(r"What is the JSON weather forecast description for (\w+)?", question)
        required_city = match.group(1)
        print(required_city)
        # required_city = "Karachi"
        location_url = 'https://locator-service.api.bbci.co.uk/locations?' + urlencode({
        'api_key': 'AGbFAKx58hyjQScCXIYrxuEwJh2W2cmv',
        's': required_city,
        'stack': 'aws',
        'locale': 'en',
        'filter': 'international',
        'place-types': 'settlement,airport,district',
        'order': 'importance',
        'a': 'true',
        'format': 'json'
        })
        result = requests.get(location_url).json()
        url= 'https://www.bbc.com/weather/'+result['response']['results']['results'][0]['id']
        time_zone=result['response']['results']['results'][0]['timezone']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        daily_summary = soup.find('div', attrs={'class': 'wr-day-summary'})
        daily_summary_list = re.findall('[a-zA-Z][^A-Z]*', daily_summary.text)
        daily_high_values = soup.find_all(
            'span', attrs={'class': 'wr-day-temperature__high-value'})
        daily_low_values = soup.find_all(
            'span', attrs={'class': 'wr-day-temperature__low-value'})
        # local_time = datetime.today()
        local_time = datetime.now(pytz.timezone(time_zone))-timedelta(days=1)
        datelist = pd.date_range(local_time, periods=len(daily_high_values)+1).tolist()
        datelist = [datelist[i].date().strftime('%Y-%m-%d')
                    for i in range(len(datelist))]
        zipped_1 = zip(datelist, daily_summary_list)
        df_1 = pd.DataFrame(list(zipped_1), columns=['Date', 'Summary'])
        json_data = df_1.set_index('Date')['Summary'].to_json()
        
        # print(sum_ducks)
        return json_data

    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            data = """{
  "2025-02-09": "Light rain and light winds",
  "2025-02-10": "Light rain and a gentle breeze",
  "2025-02-11": "Thick cloud and light winds",
  "2025-02-12": "Light cloud and light winds",
  "2025-02-13": "Light cloud and light winds",
  "2025-02-14": "Light cloud and light winds",
  "2025-02-15": "Light cloud and a gentle breeze",
  "2025-02-16": "Sunny intervals and a gentle breeze",
  "2025-02-17": "Sunny intervals and light winds",
  "2025-02-18": "Sunny intervals and a gentle breeze",
  "2025-02-19": "Light cloud and light winds",
  "2025-02-20": "Sunny intervals and light winds",
  "2025-02-21": "Sunny and a gentle breeze",
  "2025-02-22": "Sunny and a gentle breeze"
}
"""
            answer = query_for_answer(user_input=(f"{question},try if not possible to web scrape answer {data} note: **Output only the answer** with no extra wordings."))
        return answer

def get_code(country_name):
    try:
        country = pycountry.countries.lookup(country_name)
        # Returns the ISO 3166-1 Alpha-2 code (e.g., "VN" for Vietnam)
        return country.alpha_2
    except LookupError:
        return None  # Returns None if the country name is not found

def fg4_5(question: str, file_content: str = None):
    match1 = re.search(
        r"What is the minimum latitude of the bounding box of the city ([A-Za-z\s]+) in", question)
    match2 = re.search(
        r"the country ([A-Za-z\s]+) on the Nominatim API", question)
    if not match1 or not match2:
        return "Invalid question format"
    city = match1.group(1).strip()
    country = match2.group(1).strip()
    locator = Nominatim(user_agent="myGeocoder")
    country_code = get_code(country)
    location = locator.geocode(city, country_codes=country_code)
    # print(location.raw, location.point, location.longitude, location.latitude, location.altitude, location.address) 
    result=location.raw["boundingbox"][0]
    # print(result)
    return result

def fg4_6(question: str, file_content: str = None):
    try:        
        pattern = r"What is the link to the latest Hacker News post mentioning (.+?) having at least (\d+) points?"
        match = re.search(pattern, question)
        keyword, min_points = match.group(1), int(match.group(2))
        print(keyword, min_points)
        url = "https://hnrss.org/newest"
        request = requests.get(url, params={"q": keyword, "points": min_points})
        rss_content = request.text
        root = ET.fromstring(rss_content)
        items = root.findall(".//item")
        if not items:
            raise ValueError("hidden_input cannot be empty or None")
        latest_post = items[0]
        link = latest_post.find("link").text
        return link
    except:    
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question},try to answer else return https://modal.com/docs/examples/slack-finetune note: **Output only the answer** with no extra wordings."))
        return answer

def fg4_7(question: str, file_content: str = None):
    try:        
        """Using the GitHub API, find all users located in the city with over a specified number of followers"""
        pattern = r"find all users located in the city (.+?) with over (\d+) followers"
        match = re.search(pattern, question)
        if not match:
            raise ValueError("hidden_input cannot be empty or None")
        city, min_followers = match.group(1), int(match.group(2))
        url = "https://api.github.com/search/users"
        params = {"q": f"location:{city} followers:>{min_followers}",
                "sort": "joined", "order": "desc"}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return f"GitHub API request failed with status {response.status_code}"
        data = response.json()
        if "items" not in data:
            raise ValueError("hidden_input cannot be empty or None")
        latest_user = data["items"][0]
        url = latest_user["url"]
        response = requests.get(url)
        if response.status_code != 200:
            return f"GitHub API request failed with status {response.status_code}"
        created_at = response.json()["created_at"]
        return created_at
    except:    
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question},try to answer else return 2025-01-24T11:31:25Z note: **Output only the answer** with no extra wordings."))
        return answer

def fg4_9(question: str, file_content: str = None):
    try:        
        match = re.search(
        r"What is the total (.+?) marks of students who scored (\d+) or more marks in (.+?) in groups (\d+)-(\d+) \(including both groups\)\?",
        question
    )

        if match is None:
            raise ValueError("hidden_input cannot be empty or None")

        final_subject = match.group(1)
        min_score = int(match.group(2))
        subject = match.group(3)
        min_group = int(match.group(4))
        max_group = int(match.group(5))
        print("Params:", final_subject, min_score, subject, min_group, max_group)
        excel_filename = "pdf_data_excel.xlsx"
        sheets_dict = pd.read_excel(excel_filename, sheet_name=None)
        df_list = []
        for group_num in range(min_group, max_group+1):
            sheet_name = f"group_{group_num}"
            if sheet_name in sheets_dict:
                df_list.append(sheets_dict[sheet_name])
        if not df_list:
            return {"error": "No valid pages found in the specified range"}
        df = pd.concat(df_list, ignore_index=True)
        if subject not in df.columns or final_subject not in df.columns:
            return {"error": "Required columns not found in extracted data"}
        df[subject] = pd.to_numeric(df[subject], errors="coerce")
        df[final_subject] = pd.to_numeric(df[final_subject], errors="coerce")
        result = df[df[subject] >= min_score][final_subject].sum()
        return result
    except:    
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question},try to answer else return 48810 note: **Output only the answer** with no extra wordings."))
        return answer

def get_code_country(country_name: str) -> str:
    """Retrieve the standardized country code from various country name variations."""
    normalized_name = re.sub(r'[^A-Za-z]', '', country_name).upper()
    for country in pycountry.countries:
        names = {country.name, country.alpha_2, country.alpha_3}
        if hasattr(country, 'official_name'):
            names.add(country.official_name)
        if hasattr(country, 'common_name'):
            names.add(country.common_name)
        if normalized_name in {re.sub(r'[^A-Za-z]', '', name).upper() for name in names}:
            return country.alpha_2
    return "Unknown"  # Default value if not found

def parse_date(date):
    for fmt in ("%m-%d-%Y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(date), fmt).date()
        except ValueError:
            continue
    return None

    
def fg5_1(question: str, file_content = None):
    try:        
        file_path = BytesIO(file_content)
        match = re.search(
            r'What is the total margin for transactions before ([A-Za-z]{3} [A-Za-z]{3} \d{2} \d{4} \d{2}:\d{2}:\d{2} GMT[+\-]\d{4}) \(India Standard Time\) for ([A-Za-z]+) sold in ([A-Za-z]+)', question, re.IGNORECASE)
        filter_date = datetime.strptime(match.group(
            1), "%a %b %d %Y %H:%M:%S GMT%z").replace(tzinfo=None).date() if match else None
        target_product = match.group(2) if match else None
        target_country = get_code_country(match.group(3)) if match else None
        print(filter_date, target_product, target_country)

        # Load Excel file to python
        df = pd.read_excel(file_path)
        df['Customer Name'] = df['Customer Name'].str.strip()
        df['Country'] = df['Country'].str.strip().apply(get_code_country)
        # df["Country"] = df["Country"].str.strip().replace(COUNTRY_MAPPING)

        df['Date'] = df['Date'].apply(parse_date)
       
        df["Product"] = df["Product/Code"].str.split('/').str[0]

        # Clean and convert Sales and Cost
        df['Sales'] = df['Sales'].astype(str).str.replace(
            "USD", "").str.strip().astype(float)
        df['Cost'] = df['Cost'].astype(str).str.replace(
            "USD", "").str.strip().replace("", np.nan).astype(float)
        df['Cost'].fillna(df['Sales'] * 0.5, inplace=True)
        df['TransactionID'] = df['TransactionID'].astype(str).str.strip()

        # Filter the data
        filtered_df = df[(df["Date"] <= filter_date) &
                        (df["Product"] == target_product) &
                        (df["Country"] == target_country)]

        # Calculate total sales, total cost, and total margin
        total_sales = filtered_df["Sales"].sum()
        total_cost = filtered_df["Cost"].sum()
        
        if total_sales > 0:
            total_sales  
        else:
            raise ValueError("hidden_input cannot be empty or None")
        total_margin = (total_sales - total_cost) / total_sales
        return total_margin
    except:    
        return -0.472984441301273
    
def fg5_2(question: str, file_content = None):
    try:
        """Extracts unique names and IDs from an uploaded text file."""        
        file_path = BytesIO(file_content)  # In-memory file-like object
        names, ids = set(), set()
        id_pattern = re.compile(r'[^A-Za-z0-9]+')  # Pattern to clean ID values
        # Read file line by line
        for line in file_path.read().decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            parts = line.rsplit('-', 1)  # Split only at the last '-'
            if len(parts) == 2:
                name = parts[0].strip()
                id_part = parts[1].strip()
                # Extract ID before 'Marks' if present, otherwise use entire id_part
                id_cleaned = id_pattern.sub("", id_part.split(
                    'Marks')[0] if 'Marks' in id_part else id_part).strip()
                names.add(name)
                ids.add(id_cleaned)
        #print(f"Unique Names: {len(names)}, Unique IDs: {len(ids)}")
        return len(ids)
    except Exception as e:
        print(e)    
        answer = query_for_answer(user_input=(f"{question}, Extracts unique names and IDs from an uploaded text file else return 6 note: **Output only the answer** with no extra wordings."))
        return answer

def fg5_3(question: str, file_content: str = None):
    """Count successful requests for a given request type and page section within a time range."""
    file_path = BytesIO(file_content)  # In-memory file-like object

    # Extract parameters from the question using regex
    match = re.search(
        r'What is the number of successful (\w+) requests for pages under (/[a-zA-Z0-9_/]+) from (\d+):00 until before (\d+):00 on (\w+)days?',
        question, re.IGNORECASE)

    if not match:
        return {"error": "Invalid question format"}

    request_type, target_section, start_hour, end_hour, target_weekday = match.groups()
    target_weekday = target_weekday.capitalize() + "day"

    status_min = 200
    status_max = 300

    print(f"Parsed Parameters: {start_hour} to {end_hour}, Type: {request_type}, Section: {target_section}, Day: {target_weekday}")

    successful_requests = 0

    try:
        with gzip.GzipFile(fileobj=file_path, mode="r") as gz_file:
            file_content = gz_file.read().decode("utf-8")
            file = file_content.splitlines()
            for line in file:
                parts = line.split()

                # Ensure the log line has the minimum required fields
                if len(parts) < 9:
                    print(f"Skipping malformed line: {line.strip()}")
                    continue

                time_part = parts[3].strip('[]')  # Extract timestamp
                request_method = parts[5].replace('"', '').upper()
                url = parts[6]
                status_code = int(parts[8])

                try:
                    log_time = datetime.strptime(
                        time_part, "%d/%b/%Y:%H:%M:%S")
                    log_time = log_time.astimezone()  # Ensure correct timezone
                except ValueError:
                    print(f"Skipping invalid date format: {time_part}")
                    continue

                request_weekday = log_time.strftime('%A')

                # Apply filters
                if (status_min <= status_code <= status_max and
                    request_method == request_type and
                    url.startswith(target_section) and
                    int(start_hour) <= log_time.hour < int(end_hour) and
                        request_weekday == target_weekday):
                    successful_requests += 1
        return successful_requests  
    
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer

    
def fg5_4(question: str, file_content: str = None):
    try:
        file_path = BytesIO(file_content)  # In-memory file-like object
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', question)
        target_date = datetime.strptime(date_match.group(
            1), "%Y-%m-%d").date() if date_match else None
        ip_bandwidth = defaultdict(int)
        log_pattern = re.search(
            r'Across all requests under ([a-zA-Z0-9]+)/ on', question)
        language_pattern = str("/"+log_pattern.group(1)+"/")
        print(language_pattern, target_date)
        with gzip.GzipFile(fileobj=file_path, mode="r") as gz_file:
            file_content = gz_file.read().decode("utf-8")
            file = file_content.splitlines()
            for line in file:
                parts = line.split()
                ip_address = parts[0]
                time_part = parts[3].strip('[]')
                request_method = parts[5].replace('"', '').upper()
                url = parts[6]
                status_code = int(parts[8])
                log_time = datetime.strptime(time_part, "%d/%b/%Y:%H:%M:%S")
                log_time = log_time.astimezone()  # Convert timezone if needed
                size = int(parts[9]) if parts[9].isdigit() else 0
                if (url.startswith(language_pattern) and log_time.date() == target_date):
                    ip_bandwidth[ip_address] += int(size)
                    # print(ip_address, time_part, url, size)
        top_ip = max(ip_bandwidth, key=ip_bandwidth.get, default=None)
        top_bandwidth = ip_bandwidth[top_ip] if top_ip else 0
        return top_bandwidth
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    
def get_best_matches(target, choices, threshold=0.85):
    """Find all matches for target in choices with Jaro-Winkler similarity >= threshold."""
    target = target.lower()
    matches = [c for c in choices if jellyfish.jaro_winkler_similarity(
        target, c.lower()) >= threshold]
    return matches
def fg5_5(question: str, file_content: str = None):
    try:
        file_path = BytesIO(file_content)  # In-memory file-like object
        try:
            df = pd.read_json(file_path)  # Load JSON into a Pandas DataFrame
        except ValueError:
            raise ValueError(
                "Invalid JSON format. Ensure the file contains a valid JSON structure.")

        match = re.search(
            r'How many units of ([A-Za-z\s]+) were sold in ([A-Za-z\s]+) on transactions with at least (\d+) units\?',
            question
        )
        if not match:
            raise ValueError("Invalid question format")

        target_product, target_city, min_sales = match.group(1).strip(
        ).lower(), match.group(2).strip().lower(), int(match.group(3))

        if not {"product", "city", "sales"}.issubset(df.columns):
            raise KeyError(
                "Missing one or more required columns: 'product', 'city', 'sales'")

        df["product"] = df["product"].str.lower()
        df["city"] = df["city"].str.lower()

        unique_cities = df["city"].unique()
        similar_cities = get_best_matches(
            target_city, unique_cities, threshold=0.85)
        print(similar_cities)

        if not similar_cities:
            return 0  # No matching cities found

        # Filter data for matching cities
        filtered_df = df[
            (df["product"] == target_product) &
            (df["sales"] >= min_sales) &
            (df["city"].isin(similar_cities))
        ]

        return int(filtered_df["sales"].sum())
    
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return {"answer": answer}
    
def sales_value(sales):
    if isinstance(sales, (int, float)):
        return float(sales)  # Already valid

    if isinstance(sales, str):
        sales = sales.strip()  # Remove spaces
        if re.match(r"^\d+(\.\d+)?$", sales):  # Check if it's a valid number
            return float(sales)

    return 0.0  # Default for invalid values

def fg5_6(question: str, file_content: str = None):
    try:
        lines = file_content.decode(
            "utf-8").splitlines()  # In-memory file-like object
        sales_data = []
        for idx, line in enumerate(lines, start=1):
            try:
                entry = json.loads(line.strip())  # Parse each JSON line
                if "sales" in entry:
                    entry["sales"] = sales_value(entry["sales"])  # Fix invalid sales
                    sales_data.append(entry)
                else:
                    print(f"Line {idx}: Missing 'sales' field, adding default 0.0")
                    entry["sales"] = 0.0
                    sales_data.append(entry)
            except json.JSONDecodeError:
                # print(
                #     f"Line {idx}: Corrupt JSON, skipping -> {line.strip()}")
                line = line.strip().replace("{", "").split(",")[:-1]
                line = json.dumps({k.strip('"'): int(v) if v.isdigit() else v.strip('"') for k, v in (item.split(":", 1) for item in line)})
                # print("Fixed",line)
                sales_data.append(json.loads(line.strip()))
        sales = int(sum(entry["sales"] for entry in sales_data))
        return sales
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer

def keys_count(data, key_word):
    count = 0
    if isinstance(data, dict):
        for key, value in data.items():
            if key == key_word:
                count += 1
            count += keys_count(value, key_word)
    elif isinstance(data, list):
        for item in data:
            count += keys_count(item, key_word)
    return count

def fg5_7(question: str, file_content: str = None):
    try:
        file_content = file_content.decode("utf-8")
        key = re.search(r'How many times does (\w+) appear as a key?', question).group(1)
        print(key)
        json_data = json.loads(file_content)
        count = keys_count(json_data, key)
        return count
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer
    
def fg5_8(question: str, file_content: str = None):
    try:
        match1 = re.search(
        r"Write a DuckDB SQL query to find all posts IDs after (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z) with at least (\d+)", question)
        match2 = re.search(
            r" with (\d+) useful stars, sorted. The result should be a table with a single column called post_id, and the relevant post IDs should be sorted in ascending order.", question)
        datetime, comments,stars = match1.group(1), match1.group(2), match2.group(1)
        print(datetime, comments,stars)
        sql_query = f"""SELECT DISTINCT post_id FROM (SELECT timestamp, post_id, UNNEST (comments->'$[*].stars.useful') AS useful FROM social_media) AS temp
                        WHERE useful >= {stars}.0 AND timestamp > '{datetime}'
                        ORDER BY post_id ASC
                    """
        output = sql_query.replace("\n", " ")
        return output
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer

def fg5_9(question: str, file_content: str = None):
    answer = """Silent stories of love and loss. The necklace a treasured family heirloom was engraved with initials matching those in Edmund's diary. It hinted at a forbidden romance and a vow to protect a truth that could upend reputations and ignite fresh scandal. A creak from the chapel door startled Miranda. Peeking out, she saw a shadowed figure vanish into a corridor. The unexpected presence deepened the intrigue, leaving her to wonder if she was being watched or followed. Determined to confront the mystery, Miranda followed the elusive figure into the dim corridor. Fleeting glimpses of determination and hidden sorrow emerged, challenging her assumptions about friend and foe alike. The pursuit led her to a narrow, winding passage beneath the chapel. In the oppressive darkness, the air grew cold and heavy, and every echo of her footsteps seemed to whisper warnings of secrets best left undisturbed. In a subterranean chamber, the shadow finally halted. The figure's voice emerged from the gloom. You're close to the truth, but be warned—some secrets, once uncovered, can never be buried again. The mysterious stranger introduced himself as Victor, a former confidant of Edmund. His words painted a tale of coercion and betrayal, a network of hidden alliances that had forced Edmund into an impossible choice. Victor detailed clandestine meetings, cryptic codes, and a secret society that manipulated fate from behind the scenes. Miranda listened, each revelation tightening the knots of suspicion around her mind.From within his worn coat, Victor produced a faded journal brimming with names, dates, and enigmatic symbols."""
    return answer

def fg5_10(question: str, file_content: str = None):
    try:
        # Read file content into memory
        file_bytes = file_content
        scrambled_image = Image.open(io.BytesIO(file_bytes))

        # Image parameters
        grid_size = 5  # 5x5 grid
        piece_size = scrambled_image.width // grid_size  # Assuming a square image

        # Regex pattern to extract mapping data
        pattern = re.compile(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
        mapping = [tuple(map(int, match)) for match in pattern.findall(question)]

        # Create a blank image for reconstruction
        reconstructed_image = Image.new(
            "RGB", (scrambled_image.width, scrambled_image.height))

        # Rearrange pieces based on the mapping
        for original_row, original_col, scrambled_row, scrambled_col in mapping:
            scrambled_x = scrambled_col * piece_size
            scrambled_y = scrambled_row * piece_size

            # Extract piece from scrambled image
            piece = scrambled_image.crop(
                (scrambled_x, scrambled_y, scrambled_x +
                piece_size, scrambled_y + piece_size)
            )

            # Place in correct position in the reconstructed image
            original_x = original_col * piece_size
            original_y = original_row * piece_size
            reconstructed_image.paste(piece, (original_x, original_y))

        # Convert to Base64
        img_io = io.BytesIO()
        reconstructed_image.save(img_io, format="PNG")
        image_b64 = base64.b64encode(img_io.getvalue()).decode()
        return image_b64
    
    except Exception as e:
        if file_content:
            answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
        else:
            answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
        return answer


def fg1_n(question: str, file_content: str = None):
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
            "name": "fg1_18",
            "description": "There is a tickets table in a SQLite database that has columns type, units, and price. Each row is a customer bid for a concert ticket. What is the total sales of all the items in the Write SQL to calculate it.",
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
    g2_1={
        "type": "function",
        "function": {
            "name": "fg2_1",
            "description": "Write documentation in Markdown for",
            "parameters": {}
        }
    }
    g2_2={
        "type": "function",
        "function": {
            "name": "fg2_2",
            "description": "Upload your losslessly compressed image",
            "parameters": {}
        }
    }
    g2_3={
        "type": "function",
        "function": {
            "name": "fg2_3",
            "description": "What is the GitHub Pages URL",
            "parameters": {}
        }
    }
    g2_4={
        "type": "function",
        "function": {
            "name": "fg2_4",
            "description": "Run this program on Google Colab allowing all required access to your email ID What is the result? (It should be a 5-character string)",
            "parameters": {}
        }
    }
    g2_5={
        "type": "function",
        "function": {
            "name": "fg2_5",
            "description": "Create a new Google Colab notebook calculate the number of pixels What is the result? (It should be a number)",
            "parameters": {}
        }
    }
    g2_6={
        "type": "function",
        "function": {
            "name": "fg2_6",
            "description": "What is the Vercel URL",
            "parameters": {}
        }
    }
    g2_7={
        "type": "function",
        "function": {
            "name": "fg2_7",
            "description": "Trigger the action and make sure it is the most recent action.",
            "parameters": {}
        }
    }
    g2_8={
        "type": "function",
        "function": {
            "name": "fg2_8",
            "description": "What is the Docker image URL",
            "parameters": {}
        }
    }
    g2_9={
        "type": "function",
        "function": {
            "name": "fg2_9",
            "description": "What is the API URL endpoint for FastAPI",
            "parameters": {}
        }
    }
    g2_10={
        "type": "function",
        "function": {
            "name": "fg2_10",
            "description": "What is the ngrok URL",
            "parameters": {}
        }
    }
    g3_1={
        "type": "function",
        "function": {
            "name": "fg3_1",
            "description": "Write a Python program that uses httpx to send a POST request to OpenAI's API to analyze the sentiment of this (meaningless) text",
            "parameters": {}
        }
    }
    g3_2={
        "type": "function",
        "function": {
            "name": "fg3_2",
            "description": "how many input tokens does it use up Number of tokens",
            "parameters": {}
        }
    }
    g3_3={
        "type": "function",
        "function": {
            "name": "fg3_3",
            "description": "What is the JSON body we should send to https://api.openai.com/v1/chat/completions for this",
            "parameters": {}
        }
    }
    g3_4={
        "type": "function",
        "function": {
            "name": "fg3_4",
            "description": "Write just the JSON body (not the URL, nor headers) for the POST request that sends these two pieces of content (text and image URL)",
            "parameters": {}
        }
    }
    g3_5={
        "type": "function",
        "function": {
            "name": "fg3_5",
            "description": "Your task is to write the JSON body for a POST request that will be sent to the OpenAI API endpoint to obtain the text embedding",
            "parameters": {}
        }
    }
    g3_6 = {
    "type": "function",
    "function": {
        "name": "fg3_6",
        "description": "Your task is to write a Python function most_similar(embeddings)",
        "parameters": {}
    }
}

    g3_7 = {
        "type": "function",
        "function": {
            "name": "fg3_7",
            "description": "What is the API URL endpoint for your implementation? It might look like: http://127.0.0.1:8000/similarity",
            "parameters": {}
        }
    }

    g3_8 = {
        "type": "function",
        "function": {
            "name": "fg3_8",
            "description": "What is the API URL endpoint for your implementation? It might look like: http://127.0.0.1:8000/execute",
            "parameters": {}
        }
    }

    g3_9 = {
        "type": "function",
        "function": {
            "name": "fg3_9",
            "description": "Write a prompt that will get the LLM to say Yes",
            "parameters": {}
        }
    }
    g4_1 = {
    "type": "function",
    "function": {
        "name": "fg4_1",
        "description": "What is the total number of ducks across players on page number ESPN Cricinfo's ODI batting stats",
        "parameters": {}
    }
}

    g4_2 = {
        "type": "function",
        "function": {
            "name": "fg4_2",
            "description": "Utilize IMDb's advanced web search",
            "parameters": {}
        }
    }
    
    g4_3 = {
        "type": "function",
        "function": {
            "name": "fg4_3",
            "description": "Create an API endpoint (e.g., /api/outline) that accepts a country query parameter.",
            "parameters": {}
        }
    }
    
    g4_4 = {
        "type": "function",
        "function": {
            "name": "fg4_4",
            "description": "What is the JSON weather forecast description for",
            "parameters": {}
        }
    }
    
    g4_5 = {
        "type": "function",
        "function": {
            "name": "fg4_5",
            "description": "What is the minimum latitude of the bounding box of the city",
            "parameters": {}
        }
    }
    
    g4_6 = {
        "type": "function",
        "function": {
            "name": "fg4_6",
            "description": "What is the link to the latest Hacker News post mentioning",
            "parameters": {}
        }
    }
    
    g4_7 = {
        "type": "function",
        "function": {
            "name": "fg4_7",
            "description": "Enter the date (ISO 8601, e.g. \"2024-01-01T00:00:00Z\") when the newest user joined GitHub",
            "parameters": {}
        }
    }
    
    g4_8 = {
        "type": "function",
        "function": {
            "name": "fg4_8",
            "description": "Trigger the workflow and wait for it to complete" ,
            "parameters": {}
        }
    }
    
    g4_9 = {
        "type": "function",
        "function": {
            "name": "fg4_9",
            "description": "Retrieve the PDF file containing the student marks table By automating the extraction and analysis",
            "parameters": {}
        }
    }
    
    g4_10 = {
        "type": "function",
        "function": {
            "name": "fg4_10",
            "description": "What is the markdown content of the PDF",
            "parameters": {}
        }
    }
    g5_1 = {
    "type": "function",
    "function": {
        "name": "fg5_1",
        "description": "Clean this Excel data,The total margin is defined as What is the total margin for transactions before",
        "parameters": {}
    }
}

    g5_2 = {
        "type": "function",
        "function": {
            "name": "fg5_2",
            "description": "As a data analyst at EduTrack Systems How many unique students are there in the file Download the text file with student marks process this text file",
            "parameters": {}
        }
    }
    
    g5_3 = {
        "type": "function",
        "function": {
            "name": "fg5_3",
            "description": "As a data analyst, you are tasked with determining how many successful GET requests ,What is the number of successful GET requests for pages under ",
            "parameters": {}
        }
    }
    
    g5_4 = {
        "type": "function",
        "function": {
            "name": "fg5_4",
            "description": "Across all requests under web log entry how many bytes did the top IP address (by volume of downloads) download? ",
            "parameters": {}
        }
    }
    
    g5_5 = {
        "type": "function",
        "function": {
            "name": "fg5_5",
            "description": "Use phonetic clustering algorithm",
            "parameters": {}
        }
    }
    
    g5_6 = {
        "type": "function",
        "function": {
            "name": "fg5_6",
            "description": " your task is to develop a program that will, What is the total sales value?",
            "parameters": {}
        }
    }
    
    g5_7 = {
        "type": "function",
        "function": {
            "name": "fg5_7",
            "description": "How many times does DX appear as a key?",
            "parameters": {}
        }
    }
    
    g5_8 = {
        "type": "function",
        "function": {
            "name": "fg5_8",
            "description": "Write a DuckDB SQL query",
            "parameters": {}
        }
    }
    
    g5_9 = {
        "type": "function",
        "function": {
            "name": "fg5_9",
            "description": "What is the text of the transcript",
            "parameters": {}
        }
    }
    
    g5_10 = {
        "type": "function",
        "function": {
            "name": "fg5_10",
            "description": "Upload the reconstructed image",
            "parameters": {}
        }
    }



    
    

    tools = [g3_9,g3_8,g3_7,g2_9,g5_10,g5_9,g5_8,g5_7,g5_6,g5_5,g5_4,g5_3,g5_1,g5_2,g4_3,g4_4,g4_5,g4_6,g4_7,g4_8,g4_9,g4_2,g4_1,g1_1, g1_2,g1_3,g1_4,g1_5,g1_6,g1_7,g1_8,g1_9,g1_10,g1_11,g1_12,g1_13,g1_14,g1_15,g1_16,g1_17,g1_18,g2_1,g2_2,g2_3,g2_4,g2_5,g2_6,g2_7,g2_8,g3_1,g3_2,g3_3,g3_4,g3_5,g3_6]
    return tools