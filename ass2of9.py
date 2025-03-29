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

@app.get("/api")
async def get_students(class_: list[str] = Query(default=None, alias="class")):
    students_data = read_student_data(os.path.join(os.getcwd(), "datafiles", "q-fastapi.csv"))
    if class_:
        filtered_students = [
            student for student in students_data if student["class"] in class_]
        print(filtered_students)
        return JSONResponse(content={"students": filtered_students})
    return JSONResponse(content={"students": students_data})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2020, reload=True)