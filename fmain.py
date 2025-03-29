from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request,Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import zipfile
import pandas as pd
from io import BytesIO
import os
import requests
import json
import base64
# Absolute import
from ai_query import query_gpt
from function_ai import fg2_9,fg5_9,fg5_10,fg5_8,fg5_7,fg5_6,fg5_5,fg5_4,fg5_3,fg5_2,fg5_1,fg4_9,fg4_6,fg4_7,fg4_5,fg4_3,fg4_4,fg4_2,fg4_1,fg3_6,fg3_5,fg3_4,fg3_3,fg3_2,fg1_1, fg1_2, fg1_3, fg1_4, fg1_5, fg1_6,fg1_7,fg1_10,fg1_17,fg1_8,fg1_9,fg1_11,fg1_12,fg1_13,fg1_14,fg1_15,fg1_16,fg1_18,fg2_1,fg2_2,fg2_3,fg2_4,fg2_5,fg2_6,fg2_7,fg2_8,fg3_1
from typing import Dict, Any, List
import mimetypes
from fastapi.middleware.cors import CORSMiddleware

def to_string(value):
    """Converts any type of value to a string representation."""
    if value is None:
        return "None"
    if not isinstance(value, str):
        try:
            # Converts lists, dicts, and serializable objects
            return json.dumps(value)
        except (TypeError, ValueError):
            return str(value)  # Fallback for other types
    return value

def read_file(uploaded_file: UploadFile):
    file_ext = uploaded_file.filename.split(".")[-1].lower()

    try:
        if file_ext == "txt":
            return uploaded_file.file.read().decode("utf-8")

        elif file_ext == "json":
            return json.dumps(json.load(uploaded_file.file), indent=2)

        elif file_ext == "csv":
            df = pd.read_csv(uploaded_file.file)
            return df.to_csv(index=False)

        elif file_ext == "zip":
            extracted_data = []
            with zipfile.ZipFile(uploaded_file.file, "r") as zip_ref:
                for file_name in zip_ref.namelist():
                    with zip_ref.open(file_name) as file:
                        extracted_data.append(file.read().decode("utf-8", errors="ignore"))
            return "\n\n".join(extracted_data)

        else:
            file_content = file.read()  # Read bytes
              # Convert bytes to string
            return file_content
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# OpenAI API Config
EMBEDDING_API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6Imxpc2EubWlyYW5kYUBncmFtZW5lci5jb20ifQ.nvcT6zt6b65Hf-rJE1Q0bwn4KrAeGzGZ6lCi5RP3IhY"

# Templates directory 
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
@app.get("/api/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the form template."""
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/api/",name="get_data")
async def handle_request(
    request: Request, 
    question: str = Form(...),
    file: UploadFile = File(None)
) -> Dict[str, str]:
    """Handles API requests with optional file uploads."""
    query=await query_gpt(question)
    try:
        
        tool_call = query["choices"][0]["message"]["tool_calls"][0]
        func_name = tool_call["function"]["name"]
        print(func_name)
        if file:
            file_content = file.file.read()
            file_type, _ = mimetypes.guess_type(file.filename)
            print(f"File type detected: {file_type}")
        else:
            file_content = None
            
        # Handle cases with no arguments
        args = {"question": question} 
        if "arguments" in tool_call["function"] and tool_call["function"]["arguments"]:
            loaded_args = json.loads(tool_call["function"]["arguments"])
            if "file_content" in loaded_args:  # Preserve only file_content if it exists
                args["file_content"] = loaded_args["file_content"]
        
        if file:
            args["file_content"] = file_content
        # Dynamically call the function
        
        if func_name in globals():
            if func_name in ["fg1_16"]:
                if file:
                    args["file_content"] = file
                    print("file content:", args["file_content"])
                output =await globals()[func_name](**args)  # Pass args only if present
                output = to_string(output)
                answer = {"answer": output}
                print("response from here:", answer)
                return answer
            elif func_name in ["fg2_9"]:
                responce = globals()[func_name](**args)
                base_url = str(request.base_url)+"api/fromvercel"
                return {"answer": base_url}
            else:
                responce = globals()[func_name](**args) # Pass args only if present
                output = to_string(responce)
                answer = {"answer": output}
                print("no json output :", answer)
                try:
                    print("json output :", json.loads(output))
                except json.JSONDecodeError:
                    pass
                return answer
        else:
            raise ValueError(f"Function '{func_name}' not found")
    
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Error processing GPT response: {str(e)}")

@app.get("/api/fromvercel")
async def get_students(class_: list[str] = Query(default=None, alias="class")):
    from ass2of9 import read_student_data
    students_data = read_student_data(os.path.join(os.getcwd(), "datafiles", "q-fastapi.csv"))
    if class_:
        filtered_students = [
            student for student in students_data if student["class"] in class_]
        print(filtered_students)
        return JSONResponse(content={"students": filtered_students})
    return JSONResponse(content={"students": students_data})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    