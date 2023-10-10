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
from websockets.sync.client import connect
"""Perlu diingat bahwa koordinat XY ditinjau dari sudut kiri atas gambar, dengan sumbu X mengarah ke kanan dan sumbu Y mengarah ke bawah."""


#start websocket server without interrupting the main program

# create handler for each connection

clients = []
async def handler(websocket, path):
    try:
        # register new client
        clients.append(websocket)
        print(clients)
        while True:
            # receive data from client
            data = await websocket.recv()
            #print(data)
            # send data to all connected clients
            websockets.broadcast(clients, data, raise_exceptions=False)
            #await asyncio.sleep(0.5)
    except websockets.exceptions.ConnectionClosed:
        print("Connection closed")
        clients.remove(websocket)
        print(clients)
 
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

#id untuk orang adalah 0, lainnya bisa cek di https://gist.github.com/tersekmatija/9d00c4683d52d94cf348acae29e8db1a
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
    # baca gambar dari kamera
    success, img = cap.read()
    status = "Aman"
    results = model.predict(img, verbose=False, stream=True)

    # gambarkan daerah berbahaya dalam kamera
    cv2.polylines(img, [danger_zone_polygon], True, (0, 0, 255), 1)
    cv2.putText(img, "Danger Zone", danger_zone_polygon[0], font, fontScale, color, thickness)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            if int(box.cls) != class_id:
                continue

            # ambil koordinat box dari hasil deteksi
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # jika ada orang yang berada di daerah berbahaya, maka status akan berubah menjadi bahaya
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

last_status = "Aman"
while True:
    try:
        with connect("ws://localhost:12302") as websocket:
            while True:
                status = detect_image()
                if status != last_status:
                    last_status = status
                    websocket.send(status)
                #keluar jika pengguna menekan tombol q
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except websockets.exceptions.ConnectionClosed:
        continue
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        break 