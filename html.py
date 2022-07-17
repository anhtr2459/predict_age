import time
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
from starlette.responses import StreamingResponse, HTMLResponse
import io
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles

import asyncio
import use_model_class as model
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates/")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def welcome(request: Request):
    return templates.TemplateResponse('appear.html', context={'request': request})


@app.get("/image")
async def image(request: Request):
    return templates.TemplateResponse('image.html', context={'request': request})

@app.post("/upload")
async def upload_and_predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    global im_jpg
    im_jpg = model.response(frame)
    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpg")

        
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 1920)  #
        self.video.set(4, 1080) 
    def __del__(self):
        self.video.release()

 
async def gen(camera):
    c = 1
    start = time.time()
    while True:
        start_1 = time.time()
        if c % 20 == 0:
            end = time.time()
            FPS = 20/(end-start)
            print("FPS_avg : {:.6f} ".format(FPS))
            start = time.time()
        success,frame = camera.read()
        frame = cv2.resize(frame,(1080,1080))
        if not success:
            break
        else:
            buff = model.response(frame)
            frame = buff.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        end_1 = time.time()
        FPS = 1/(end_1-start_1)
        print("FPS : {:.6f} ".format(FPS))
        c +=1
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

@app.get("/camera") 
async def request_cam():
    camera = cv2.VideoCapture(0)
    cv2.release()
    return StreamingResponse(gen(camera), media_type="multipart/x-mixed-replace;boundary=frame" )

