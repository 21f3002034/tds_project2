from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import pandas as pd
from io import BytesIO
import os
import requests

app = FastAPI()

# OpenAI API Config
EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"

@app.post("/api/")
async def handle_request(
    question: str = Form(...),   # Expecting FormData
    file: UploadFile = File(None)
):
    extracted_value = None

    # Handle ZIP file extraction
    if file:
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
        
        try:
            with zipfile.ZipFile(BytesIO(await file.read())) as zip_ref:
                zip_ref.extractall("temp")

                for extracted_file in zip_ref.namelist():
                    if extracted_file.endswith('.csv'):
                        df = pd.read_csv(os.path.join("temp", extracted_file))
                        if "answer" in df.columns:
                            extracted_value = str(df['answer'].iloc[0])
                        break
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting ZIP: {str(e)}")

    # Prepare LLM prompt
    prompt = f"Question: {question}\n"
    if extracted_value:
        prompt += f"CSV Answer: {extracted_value}\n"

    # LLM query
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
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

        if response.status_code == 200:
            return JSONResponse(content={"answer": response.json()})
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
