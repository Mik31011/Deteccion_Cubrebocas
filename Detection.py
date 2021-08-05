from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from urllib.request import urlopen
from imutils.video import VideoStream #Importar librería para usar la camara nativa de la computadora
import numpy as np
import argparse
import imutils
import requests
import time
import cv2
import os
#import pyttsx3


#Variables de conversión de texto a voz por TTS
#Engine = pyttsx3.init('sapi5')
#Voices = Engine.getProperty('voices')
#Engine.setProperty('voice', Voices[0].id)
#Función para leer un texto y reproducirlo con TTS
#def Speak(audio):
    #Engine.say(audio)
    #Engine.runAndWait()

def Detect_Mask_Predictions(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    Detections = faceNet.forward()
    
    Faces = []
    Locs = []
    Preds = []

    for i in range(0, Detections.shape[2]):
        confidence = Detections[0, 0, i, 2]
        if confidence > args["confidence"]:
            box = Detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            Face = frame[startX:endX, startY:endY]
            Face = cv2.cvtColor(Face, cv2.COLOR_BGR2RGB)
            Face = cv2.resize(Face, (224, 224))
            Face = img_to_array(Face)
            Face = preprocess_input(Face)
            Face = np.expand_dims(Face, axis= 0)
            Faces.append(Face)
            Locs.append((startX, startY, endX, endY))

    if len(Faces) > 0:
        Preds = maskNet.predict(Faces)

    return (Locs, Preds)

#Creación de argumentos requeridos para tiempo de ejecución
Argument_Parser = argparse.ArgumentParser()
Argument_Parser.add_argument("-f","--face", type = str, default = "face_detector", help = "directorio del modelo de reconocimiento de rostros")
Argument_Parser.add_argument("-m", "--model", type = str, default = "mask_detector.model", help = "directorio del entrenamiento para el modelo de reconocimeinto de objetos")
Argument_Parser.add_argument("-c", "--confidence", type = float, default = 0.5, help = "probabilidad mínima para filtrar detecciones débiles")
args = vars(Argument_Parser.parse_args())
print(args)

#Carga de modelo para la detección de caras .caffemodel
print("*[INFO] Cargando modelo de detección de caras...*")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, modelPath)
print("*[INFO] Modelo de deteccion de caras cargado correctamente...*")

#Creación de modelo de detección de cubrebocas .model
print("*[INFO] Cargando modelo de detección de cubrebocas...*")
maskNet = load_model(args["model"])
print("*[INFO] Modelo de detección de cubrebocas cargado correctamente...*")

print("*[INFO] Inicializando la captura de vídeo...*")
#Si quiere usar un dispositivo USB externo de captura de vídeo requerira saber el index de la camara habilitada puede ejecutar Detectar_Camaras_Fisicas.py que le retornara los distintos index de camaras útiles
#Si se quiere usar la camara nativa de la computadora deje lo siguiente:
vs = VideoStream(src = 0).start()
time.sleep(2) 
#Si se quiere usar una camara externa por IP, los ajustes de URL como el puerto, usuario, y contraseña van a cambiar de acuerdo a la camara:
#Url_Camera = 'http://192.168.2.20:8080/video'
#vs = VideoStream(Url_Camera).start()
#time.sleep(2)

bot_token = '1634706110:AAENkGoKMX0iXyJ8YNyglUajcfrs7fSUoOA'
bot_chatID = '1094906652'

def telegram_bot_sendtxt(bot_message):
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
    response = requests.get(send_text)
    return response.json()

While_Alert = 0
while True:
    try:
        Frame = vs.read()
        Frame = imutils.resize(Frame, width = 400)
        (Locs, Preds) = Detect_Mask_Predictions(Frame, faceNet, maskNet) 
        for (box, pred) in zip(Locs, Preds):
            (startX, startY, endX, endY) = box
            (mask, whitoutmask) = pred
            Label = "USANDO CUBREBOCAS" if mask > whitoutmask else "SIN CUBREBOCAS"
            color = (0, 255, 0) if Label == "USANDO CUBREBOCAS" else (0,0,255)
            Label = "{}: {:.2f}%".format(Label, max(mask, whitoutmask) * 100)
            cv2.putText(Frame, Label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(Frame, (startX, startY), (endX, endY), color, 2)
        
        cv2.imshow("Detección de cubrebocas", Frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    except Exception as e:
        pass

cv2.destroyAllWindows()
vs.stop()