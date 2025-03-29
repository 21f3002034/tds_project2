import uvicorn  
from fastapi.middleware.cors import CORSMiddleware  
from fastapi.responses import JSONResponse  
from fastapi import FastAPI, Query, HTTPException  
import csv
import os

app = FastAPI()

app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],  
     allow_credentials=True,
     allow_methods=["*"], 
     allow_headers=["*"],  
     )

def read_student_data(file_path: str):
    students_data = []
    print(file_path)
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            students_data.append(
            {"studentId": int(row["studentId"]), "class": row["class"]})
    return students_data


