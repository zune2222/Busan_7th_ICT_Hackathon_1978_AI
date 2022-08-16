import datetime
import os
import secrets

from fastapi import FastAPI, UploadFile, File
from human_detection import detection as HumanDetection

app = FastAPI()

@app.post("/human-detection")
async def create_upload_files(file: UploadFile = File(...)):
    UPLOAD_DIRECTORY = "./image"
    contents = await file.read()
    currentTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    saved_file_name = ''.join([currentTime, secrets.token_hex(16), file.filename])
    with open(os.path.join(UPLOAD_DIRECTORY, saved_file_name), "wb") as fp:
        fp.write(contents)
    result = await HumanDetection.detectImage(os.path.join(UPLOAD_DIRECTORY,saved_file_name))
    os.remove(os.path.join(UPLOAD_DIRECTORY,saved_file_name))
    print(result)
    return result

@app.get("/")
async def helloWorld():
    return "hi, I'm AI API"