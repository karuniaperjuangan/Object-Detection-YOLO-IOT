from ultralytics import YOLO
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Union
from shapely.geometry import Polygon
import websockets
import asyncio
from websockets.server import serve
import time
import threading
import os
"""Perlu diingat bahwa koordinat XY ditinjau dari sudut kiri atas gambar, dengan sumbu X mengarah ke kanan dan sumbu Y mengarah ke bawah."""


#start websocket server without interrupting the main program

# create handler for each connection

async def handler(websocket, path):
    while True:
        #send hello message
        await websocket.send("Hello")
        asyncio.sleep(1)
 
async def start_server():
    async with websockets.serve(handler, "localhost", 12302):
        await asyncio.Future()

#start loop in a new thread
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
t = threading.Thread(target=loop.run_until_complete, daemon=True, args=(start_server(),))
t.start()
print("Server started")

#load model
model = YOLO("yolov8n.pt")

#configuration for text
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 2

#set camera id
camera_id = 0

#capture video from camera
cap = cv2.VideoCapture(camera_id)
cap.set(3, 1280)
cap.set(4, 720)

#class id that we want to detect
class_id = 0

#definisikan daerah yang berbahaya jika terdapat orang yang berada di daerah tersebut
danger_zone_polygon = np.array([
    [640, 180],
    [480, 540],
    [800, 540],
]) 

def check_intersection(polygon_1: np.ndarray, polygon_2: np.ndarray) -> bool:
    """Check if there are any intersection area between two polygons."""
    polygon_1 = Polygon(polygon_1)
    polygon_2 = Polygon(polygon_2)
    return polygon_1.intersects(polygon_2)

#definisikan fungsi untuk mendeteksi objek pada gambar
def detect_image():
    # read frame from webcam
    success, img = cap.read()
    status = "Aman"
    results = model.predict(img, verbose=False)

    # draw danger zone
    cv2.polylines(img, [danger_zone_polygon], True, (0, 0, 255), 1)
    cv2.putText(img, "Danger Zone", danger_zone_polygon[0], font, fontScale, color, thickness)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            if int(box.cls) != class_id:
                continue
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # check if box in danger zone by checking if there any intersection between box and danger zone
            if check_intersection(np.array([[x1, y1], [x2, y2], [x2, y1], [x1, y2]]), danger_zone_polygon):
                status = "Bahaya"

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            
            # confidence
            confidence = np.ceil((box.conf[0]*100))/100
            #print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            #print("Class name -->", r.names[cls])

            # object details
            org = [x1, y1]
            
            # show class
            cv2.putText(img, r.names[cls], org, font, fontScale, color, thickness)
        #show status
        cv2.putText(img, status, (10, 50), font, fontScale, color, thickness)
    cv2.imshow('Webcam', img)

    return status
    #if cv2.waitKey(1) == ord('q'):
    #    break

while True:
    status = detect_image()
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()