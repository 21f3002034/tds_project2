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
from function_ai import fg3_9,fg3_7,fg2_9,fg5_9,fg5_10,fg5_8,fg5_7,fg5_6,fg5_5,fg5_4,fg5_3,fg5_2,fg5_1,fg4_9,fg4_6,fg4_7,fg4_5,fg4_3,fg4_4,fg4_2,fg4_1,fg3_6,fg3_5,fg3_4,fg3_3,fg3_2,fg1_1, fg1_2, fg1_3, fg1_4, fg1_5, fg1_6,fg1_7,fg1_10,fg1_17,fg1_8,fg1_9,fg1_11,fg1_12,fg1_13,fg1_14,fg1_15,fg1_16,fg1_18,fg2_1,fg2_2,fg2_3,fg2_4,fg2_5,fg2_6,fg2_7,fg2_8,fg3_1
from typing import Dict, Any, List
import mimetypes
from fastapi.middleware.cors import CORSMiddleware
from ass3of7 import get_embeddings, cosine_similarity, SearchRequest, SearchResponse
from function_ai import query_for_answer

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
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIxZjMwMDIwMzRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.KEQjxQbjAIHY8_0l-WpiOL_KrBslnPTFZnexib9N6qc"

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
            elif func_name in ["fg3_7"]:
                base_url = str(request.base_url)+"similarity"
                return {"answer": base_url}
            elif func_name in ["fg3_8"]:
                base_url = str(request.base_url)+"execute"
                return {"answer": base_url}
            elif func_name in ["fg4_3"]:
                base_url = str(request.base_url)+"api/outline"
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
            if file_content:
                answer = query_for_answer(user_input=(f"{question} file {file_content}, note: **Output only the answer** with no extra wordings."))
            else:
                answer = query_for_answer(user_input=(f"{question}, note: **Output only the answer** with no extra wordings."))
            return {"answer": to_string(answer)}
    
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

@app.post("/similarity")
async def get_similar_docs(request: SearchRequest) -> SearchResponse:
    import logging   
    try:
        logging.debug(f"Received request: {request}")
        # Get embeddings for all texts
        texts = request.docs + [request.query]
        embeddings = get_embeddings(texts)
        # Separate document embeddings and query embedding
        document_emb = embeddings[:-1]
        q_emb = embeddings[-1]
        # Calculate similarities
        similarities = [
            (i, cosine_similarity(doc_emb, q_emb))
            for i, doc_emb in enumerate(document_emb)
        ]
        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        # Get top 3 most similar documents
        top_matches = [request.docs[idx] for idx, _ in similarities[:3]]
        return SearchResponse(matches=top_matches)

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/execute")
async def execute_query(q: str):
    import re
    import json
    from ass3of8 import get_embeddings, cosine_similarity, SearchRequest, SearchResponse
    try:
        query = q.lower()
        pattern_debug_info = {}       
        if re.search(r"ticket.*?\d+", query):
            ticket_id = int(re.search(r"ticket.*?(\d+)", query).group(1))
            return {"name": "get_ticket_status", "arguments": json.dumps({"ticket_id": ticket_id})}
        pattern_debug_info["ticket_status"] = re.search(
            r"ticket.*?\d+", query) is not None        
        if re.search(r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE):
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", query)
            time_match = re.search(r"(\d{2}:\d{2})", query)
            room_match = re.search(
                r"room\s*([A-Za-z0-9]+)", query, re.IGNORECASE)
            if date_match and time_match and room_match:
                return {"name": "schedule_meeting", "arguments": json.dumps({
                    "date": date_match.group(1),
                    "time": time_match.group(1),
                    "meeting_room": f"Room {room_match.group(1).capitalize()}"
                })}
        pattern_debug_info["meeting_scheduling"] = re.search(
            r"schedule.?\d{4}-\d{2}-\d{2}.?\d{2}:\d{2}.*?room", query, re.IGNORECASE) is not None
       
        if re.search(r"expense", query):
            emp_match = re.search(r"employee\s*(\d+)", query, re.IGNORECASE)
            if emp_match:
                return {"name": "get_expense_balance", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1))
                })}
        pattern_debug_info["expense_balance"] = re.search(
            r"expense", query) is not None
       
        if re.search(r"bonus", query, re.IGNORECASE):
            emp_match = re.search(
                r"emp(?:loyee)?\s*(\d+)", query, re.IGNORECASE)
            year_match = re.search(r"\b(2024|2025)\b", query)
            if emp_match and year_match:
                return {"name": "calculate_performance_bonus", "arguments": json.dumps({
                    "employee_id": int(emp_match.group(1)),
                    "current_year": int(year_match.group(1))
                })}
        pattern_debug_info["performance_bonus"] = re.search(
            r"bonus", query, re.IGNORECASE) is not None

        if re.search(r"(office issue|report issue)", query, re.IGNORECASE):
            code_match = re.search(
                r"(issue|number|code)\s*(\d+)", query, re.IGNORECASE)
            dept_match = re.search(
                r"(in|for the)\s+(\w+)(\s+department)?", query, re.IGNORECASE)
            if code_match and dept_match:
                return {"name": "report_office_issue", "arguments": json.dumps({
                    "issue_code": int(code_match.group(2)),
                    "department": dept_match.group(2).capitalize()
                })}
        pattern_debug_info["office_issue"] = re.search(
            r"(office issue|report issue)", query, re.IGNORECASE) is not None

        raise HTTPException(
            status_code=400, detail=f"Could not parse query: {q}")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse query: {q}. Error: {str(e)}. Pattern matches: {pattern_debug_info}"
        )
import httpx
from bs4 import BeautifulSoup
WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"
@app.get("/api/outline")
async def get_country_outline(
    country: str = Query(..., title="Country Name", description="Name of the country")
):
    url = WIKIPEDIA_BASE_URL + country.replace(" ", "_")

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            return {"error": "Could not fetch Wikipedia page"}

        html_content = response.text  # No need for `await` in httpx

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract headings H1 to H6
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])

    # Build Markdown-style outline
    markdown_outline = "## Contents\n\n"
    for heading in headings:
        level = int(heading.name[1])  # Extract level from tag (h1-h6)
        markdown_outline += f"{'#' * level} {heading.get_text(strip=True)}\n\n"

    return {"country": country, "outline": markdown_outline}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    