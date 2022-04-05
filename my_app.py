# Miniconda : all_in_all environement
# imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import pyzbar.pyzbar as pyzbar
from base45 import b45decode
from zlib import decompress
from cose.messages import CoseMessage
import cbor2
from scipy.spatial import distance
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import time


# load our serialized face detector model
faceNet = cv2.dnn.readNetFromCaffe('src/facemask/deploy.prototxt.txt', 'src/facemask/res10_300x300_ssd_iter_140000.caffemodel')
# load the facemask classifier model
maskNet = load_model("src/facemask/mask_detector.model")

# Load Yolo : For social distance app
net = cv2.dnn.readNet("src/social_dst/yolov3-tiny.weights", "src/social_dst/yolov3-tiny.cfg")
classes = []
with open("src/social_dst/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class MainWindow():
    cnt = 0
    detection_id = 0
    frame_id = 0

    def __init__(self, window, cap):
        self.interval = 20  # Interval in ms to get the latest frame
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        _, self.frame = self.cap.read()
        self.start_time = time.time()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.window = tab2

        self.chosen_fct = ""

        # Message to display after the image
        self.result = tk.Label(self.window, text="Scanning ... ", font=("Gabriola", 12), fg="blue")
        self.result.grid(row=12, column=1)
        self.names = []

        # fps
        self.fps = tk.Label(self.window, text="...", font=("Gabriola", 12),)
        self.fps.grid(row=13, column=1)

        # combobox pour le choix d'application
        ttk.Label(window, text="Select an option", font=("Gabriola", 12)).grid(column=0, row=0,  padx=10, pady=30)
        n = tk.StringVar()
        global apps_list

        apps_list = ttk.Combobox(self.window,  textvariable=n)
        apps_list['values'] = (' Live', ' Facemask', ' Social Distance', ' Qr code')
        apps_list.grid(column=1, row=0)
        apps_list.current(0) #default value

        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=2, column=1)

        # Update image on canvas
        self.update_image()

    def update_image(self):
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB)  # to RGB
        self.chosen_fct = apps_list.get()

        self.frame_id += 1

        if self.chosen_fct == " Facemask":
            self.image = self.facemask()
        elif self.chosen_fct == " Social Distance":
            self.image = self.social_distance()
        elif self.chosen_fct == " Qr code":
            self.check_data_qr_code(self.image)


        # calcule du fps
        value_fps = self.frame_id / float((time.time() - self.start_time))
        value_fps = int(value_fps)
        self.fps['text'] = "FPS = " +  str(value_fps)

        self.image = Image.fromarray(self.image)  # to PIL format
        self.image = ImageTk.PhotoImage(self.image)  # to ImageTk format

        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Repeat every 'interval' ms defined previously
        self.window.after(self.interval, self.update_image)


    def check_data_qr_code(self, x):
        decodedObjects = pyzbar.decode(x)
        for obj in decodedObjects:
            data = str(obj.data)
            qrstring = data[2:-1]
            try:
                compressedBinaryData = b45decode(qrstring[4:])
                uncompressedBinaryData = decompress(compressedBinaryData)
                cose = CoseMessage.decode(uncompressedBinaryData)
                cbor2Object = cbor2.loads(cose.payload)
                full_name = cbor2Object[-260][1]['nam']['fn'] + ' ' + cbor2Object[-260][1]['nam']['gn']
                if full_name in self.names:
                    self.result['text'] = full_name # + ".. Présent .."
                else:
                    self.names.append(full_name)
                    self.result['text'] = full_name
            except:
                msg = "Invalid QR code !"
                self.result['text'] = msg

    def social_distance(self):
        _, self.frame = self.cap.read()
        height, width, channels = self.frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        midpoints = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    self.cnt = self.cnt + 1

                    midp = (center_x, center_y)
                    midpoints.append([midp, self.cnt])
                    num = len(midpoints)
                    self.result['text'] = "Detected " + str(num)
                    # Compute distance between objects
                    for m in range(num):
                        for n in range(m + 1, num):
                            if m != n:
                                dst = distance.euclidean(midpoints[m][0], midpoints[n][0])
                                p1 = midpoints[m][1]
                                p2 = midpoints[n][1]
                                if (dst <= 300):
                                    self.result['text'] = "Distance = " + str(dst) + " , ALERT"
                                else:
                                    self.result['text'] = "Distance = " + str(dst) + " , Normal"
                                self.detection_id = self.detection_id + 1
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                color = colors[class_ids[i]]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(self.frame, (x, y), (x + w, y + 30), color, -1)

        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # to RGB
        return frame


    def facemask(self):
        _, self.frame = self.cap.read()
        (h, w) = self.frame.shape[:2]

        blob = cv2.dnn.blobFromImage( self.frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()

        faces = []
        locs = []
        preds = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                face = self.frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))
                facesAsNumpy = np.array(faces, dtype="float32")
                preds = maskNet.predict(facesAsNumpy, batch_size=32)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            label = ""
            if (mask > withoutMask):
                label = "Mask"
                self.result['text'] = "Mask"
            else:
                label = "No Mask"
                self.result['text'] = "No mask, ALERT !!"

            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            cv2.rectangle(self.frame, (startX, startY), (endX, endY), color, 2)

        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)  # to RGB
        return frame


def login():
    # known codes
    code_list = [1234, 5555, 1590]
    # code entrer par l'utilisateur
    input_code = int(code.get())

    if input_code in code_list:
        tabControl.hide(tab1)
        tab2.pack()
        tabControl.add(tab2, text='Smart Surveillance Application')
        tabControl.pack(expand=1, fill="both")
        tabControl.place(x=250, y=100)
        MainWindow(tab2, cv2.VideoCapture(0))
    else:
        tk.Label(tab1, text="Wrong code !", font=('Gabriola 13'), fg='red').grid(column=1, row=2)

root = tk.Tk()
root.title("SMART SURVEILLANCE APPLICATION - UNIVERSITY ABDELMALEK ESSAADI")
root.geometry("800x600")
root.iconbitmap('my_icon.ico')
root['background']='#F0F0F0'

# afficher logo application
img_1 = Image.open("logo.png")
img_1 = img_1.resize((200, 200), Image.ANTIALIAS)
img_1 = ImageTk.PhotoImage(img_1)
panel = ttk.Label(root, image=img_1)
panel.grid(row=1, column=0)
panel.place(x=10, y=10)

Title_app = ttk.Label(root, text="Welcome to Smart Surveillance Application", font=('Gabriola 20 bold'))
Title_app.grid(row=0, column=1)
Title_app.place(x=250, y=20)

tabControl = ttk.Notebook(root)
tabControl.place(x=250, y=100)

tab1 = tk.Frame(tabControl, width=900, height=600)
tab1.grid(row=2, column=1)
tab1.pack()
tab2 = tk.Frame(tabControl, width=900, height=600)
tabControl.add(tab1, text="Login Page")

# authentification par code de confidentialité
global code
code = tk.StringVar()

click_btn = Image.open("login_btn_2.png")
click_btn= click_btn.resize((40, 40), Image.ANTIALIAS)
click_btn= ImageTk.PhotoImage(click_btn)

ttk.Label(tab1, text="Password", font=('Gabriola 15')).grid(column=0, row=1, padx=30, pady=30)
ttk.Entry(tab1, textvariable=code, show="*", width=20).grid(column=1, row=1, padx=30, pady=30)
tk.Button(tab1, image=click_btn, text="OK", command=login, borderwidth=0).grid(column=2, row=1, padx=30, pady=30)

# affiche informative
img_2 = Image.open("affiche_info.png")
img_2 = img_2.resize((280, 280), Image.ANTIALIAS)
img_2 = ImageTk.PhotoImage(img_2)
panel_2 = ttk.Label(tab1, image=img_2)
panel_2.grid(row=3, column=1)


root.mainloop()