from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import pandas as pd
from io import BytesIO
import os
import requests
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
from fastapi import UploadFile

app = FastAPI()

# OpenAI API Config
EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"

templates = Jinja2Templates(directory="templates")
def read_file(uploaded_file: UploadFile):
    file_ext = uploaded_file.filename.split(".")[-1].lower()
    """Reads different file types and returns their content as a string."""
    if file_ext == "txt":
        return uploaded_file.file.read().decode("utf-8")  # Read and decode text file
    
    elif file_ext == "json":
        return json.dumps(json.load(uploaded_file.file), indent=2)  # Load and format JSON
    
    elif file_ext == "csv":
        df = pd.read_csv(uploaded_file.file)
        return df.to_csv(index=False)  # Convert CSV to a string
    
    elif file_ext == "zip":
        extracted_data = []
        with zipfile.ZipFile(uploaded_file.file, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                with zip_ref.open(file_name) as file:
                    extracted_data.append(file.read().decode("utf-8", errors="ignore"))
        return "\n\n".join(extracted_data)  # Combine extracted file contents
    
    else:
        raise ValueError("Unsupported file format")
@app.get("/api/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/api/")
async def handle_request(
    question: str = Form(...),   # Expecting FormData
    file: UploadFile = File(None)
):
    extracted_value = None

    # Handle ZIP file extraction
    if file:
        file_content = read_file(file)
    
    # Prepare LLM prompt
    
    prompt = f"Question: {question}"
    print("file:",file)
        

    # LLM query
    try:
        headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
        if file:
            # OpenAI expects `messages` field, not `input`
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "user", "content": file_content},
                    {"role": "system", "content": "give only the solution no explanation, no extra information, no extra wordings other than answer"},
                    {"role": "system", "content": "if the file is present then please use it to answer the question"}
                    ]
            }
        else:
            # OpenAI expects `messages` field, not `input`
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": "give only the solution no explanation, no extra information, no extra wordings other than answer"}                    
                    ]
            }
        # Send the POST request
        response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
        response_json = response.json()
        if response.status_code == 200:
             answer_content = response_json["choices"][0]["message"]["content"]  # Extract content
             return JSONResponse(content={"answer": answer_content})
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    finally:
        # Cleanup
        if os.path.exists("temp"):
            for f in os.listdir("temp"):
                os.remove(os.path.join("temp", f))
            os.rmdir("temp")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)