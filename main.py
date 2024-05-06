import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
from ultralytics import YOLO
import supervision as sv

model_yolo = YOLO("C:/Users/sohai/Downloads/Traffic Sign Detection and Recognition and Voice alert system/traffic_det.pt")
box_annotator = sv.BoxAnnotator(thickness=2,text_scale=0)

model = load_model('C:/Users/sohai/Downloads/Traffic Sign Detection and Recognition and Voice alert system/traffic_classifier.h5')

classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }
                 

engine = pyttsx3.init()

live_recognition_flag = False

def speak(text):
    engine.say(text)
    engine.runAndWait()

def classify_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    sign_class = np.argmax(pred) + 1
    sign_label = classes.get(sign_class, "Unknown")
    label.configure(text="Image Recognition Result: " + sign_label)
    
    label.pack()
    sign_image.pack()
    speak(sign_label)
    

    uploaded_image = Image.open(file_path)
    uploaded_image = uploaded_image.resize((200, 200), Image.ANTIALIAS)
    uploaded_image = ImageTk.PhotoImage(uploaded_image)
    sign_image.configure(image=uploaded_image)
    sign_image.image = uploaded_image

    button_frame.pack_forget()

def show_image_recognition_page():
    button_frame.pack_forget()
    upload_button.pack(side=BOTTOM, pady=10)
    live_recognition_button.pack_forget()
    image_recognition_button.pack_forget()

def show_live_recognition_page():
    button_frame.pack_forget()
    upload_button.pack_forget()
    live_recognition_button.pack_forget()
    image_recognition_button.pack_forget()
    live_recognition()

    
    button_frame.place(relx=0.5, rely=0.95, anchor='s')
    
    image_recognition_button.pack(side=LEFT)
    live_recognition_button.pack(side=RIGHT)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        classify_image(file_path)

    except Exception as e:
        print(e)

def live_recognition():
    vid = cv2.VideoCapture(0) 

    while(True): 
        ret, frame = vid.read() 

        
        res = model_yolo(frame)[0]

        detection = sv.Detections.from_ultralytics(res)
        idx = len(detection)

        if (idx>0):
            image = box_annotator.annotate(scene=frame, detections=detection)
            x1, y1, x2, y2 = detection.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
            cls_ = model_yolo.names[detection.class_id[0]]
            cv2.putText(frame,cls_ , (x1+10, y1 - 10) , cv2.FONT_HERSHEY_SIMPLEX,fontScale=1, color=(0,255, 0), thickness=2)
            speak(cls_)
        
        cv2.imshow('frame', frame) 

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    vid.release() 
    cv2.destroyAllWindows() 


def stop_live_recognition():
    global live_recognition_flag
    live_recognition_flag = False

def show_button_frame():
    button_frame.pack_forget()
    upload_button.pack_forget()
    live_recognition_button.pack_forget()
    image_recognition_button.pack_forget()

    label.configure(text="")
    sign_image.configure(image="")

    stop_live_recognition()

    button_frame.place(relx=0.5, rely=0.95, anchor='s')
    image_recognition_button.pack(side=LEFT)
    live_recognition_button.pack(side=RIGHT)


def reset_gui():
    button_frame.pack_forget()
    upload_button.pack_forget()
    live_recognition_button.pack_forget()
    image_recognition_button.pack_forget()

    label.configure(text="")
    sign_image.configure(image="")

    stop_live_recognition()

    button_frame.place(relx=0.5, rely=0.95, anchor='s')
    image_recognition_button.pack(side=LEFT)
    live_recognition_button.pack(side=RIGHT)


top = tk.Tk()
top.geometry('900x750')
top.title('Traffic Sign Classification')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 10, 'bold'))
sign_image = Label(top)

button_frame = Frame(top, background='#CDCDCD')

image_recognition_button = Button(button_frame, text="Image Recognition", command=show_image_recognition_page,
                                  padx=10, pady=5)
image_recognition_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

live_recognition_button = Button(button_frame, text="Live Recognition", command=show_live_recognition_page,
                                 padx=10, pady=5)
live_recognition_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

button_frame.place(relx=0.5, rely=0.65, anchor='center')
image_recognition_button.pack(side=LEFT)
live_recognition_button.pack(side=RIGHT)

upload_button = Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

heading = Label(top, text="Know Your Traffic Sign", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top, background='#CDCDCD')

home_button = Button(top, text="Home", command=reset_gui, padx=10, pady=5)
home_button.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
home_button.place(relx=0.95, rely=0.05, anchor='ne')

top.mainloop()
