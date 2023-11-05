from fastapi import APIRouter, File, UploadFile
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

import numpy as np
import cv2
from services.cnn_service import preprocessing_algorithm
from onnxruntime import InferenceSession
from ultralytics import YOLO
import os
from utils.preprocessing import preprocessing_algorithm2
from utils.orb_functions import findRelevantKeypoints
from utils.orb_functions import findMax
from utils.orb_functions import findDescription
import re
from fastapi.responses import JSONResponse
prediction = APIRouter()

#cargar modelo pt
model_yolo_2021 = YOLO("services/2021.pt")
model_yolo_2009 = YOLO("services/2009.pt")


# Cargar modelo
soles_model = load_model("services/model_ResNet50_v2.h5")
dollars_model = load_model("services/model_ResNet50_v2.h5")

@prediction.post("/predictions/soles", tags=["Prediction"])
async def predict_soles_veracity(file: UploadFile = File(...)):
    labels = ['100 soles', '100 soles falsos', '20 soles', '20 soles falsos']

    contents = await file.read()
    imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    
    # Para la V2 no olvidar entrenar el modelo con imagenes en escala de grises => image_input = Input(shape=(width_shape, height_shape, 1))
    # El modelo tiene transeferencia de aprendizaje con Imagenet, el preentranamiento fue hecho en RGB por lo q las imgs del datatset deben tener los mismos canales
    preprocess_image = preprocessing_algorithm(imaget)
    image_resized = cv2.resize(preprocess_image, (224, 224), interpolation=cv2.INTER_AREA)

    xt = np.asarray(image_resized) #preprocess_image
    xt = preprocess_input(xt)
    xt = np.expand_dims(xt,axis=0)

    preds = soles_model.predict(xt)
    predict = labels[np.argmax(preds)]
    
    result = {
        'prediction': predict,
        'percentage': float(np.max(preds))
    }
    return result

@prediction.post("/predictions/dollars", tags=["Prediction"])
async def predict_dollars_veracity(file: UploadFile = File(...)):
    labels = ['100 soles', '100 soles falsos', '20 soles', '20 soles falsos']

    contents = await file.read()
    imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    
    preprocess_image = preprocessing_algorithm(imaget)
    image_resized = cv2.resize(preprocess_image, (224, 224), interpolation=cv2.INTER_AREA)

    xt = np.asarray(image_resized) #preprocess_image
    xt = preprocess_input(xt)
    xt = np.expand_dims(xt,axis=0)

    preds = dollars_model.predict(xt)
    predict = labels[np.argmax(preds)]
    
    result = {
        'prediction': predict,
        'percentage': float(np.max(preds))
    }
    return result

@prediction.post("/predictions/yolo/soles", tags=["Prediction"])
async def predict_soles_veracity(file: UploadFile = File(...)):

    #imaget
    contents = await file.read()
    imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)


    #orb
    path="services/dataset"
    images=[]
    classNames=[]
    myList=os.listdir(path)
    print(myList)
    for img in myList:
     if img.endswith(('.png', '.jpg', '.jpeg')):
        imgCur = preprocessing_algorithm2(f'{path}/{img}')
        images.append(imgCur)
        classNames.append(os.path.splitext(img)[0])
    print(classNames)

    img2=preprocessing_algorithm(imaget)
    desList = findDescription(images)
    matchList = findRelevantKeypoints(img2, desList)
    id, accuracy = findMax(matchList)
    print(accuracy)
    print(classNames[id])
    #concatenar
    string = classNames[id]
    matches = re.match(r"(\d+)([^\d]+)_(\d+)", string)
    if matches:
     value_response = matches.group(1) +" " +matches.group(2)
     year_response = matches.group(3)

     print(f"Value: {value_response}")
     print(f"Year: {year_response}")
    else:
     print("No se encontraron coincidencias.")  
  
    
    # Realizar la predicción
    if year_response=="2021":
       results = model_yolo_2021.predict(imaget, imgsz=640, conf=0.0)
       labels = ["hilo_de_seguridad", "marca_de_agua", "microimpresiones", "figuras_en_movimiento"] 
    if year_response=="2009":
       results = model_yolo_2009.predict(imaget, imgsz=640, conf=0.0)
       labels = ["marca_de_aguas", "microimpresioness", "numero_ocultos", "hilo_de_seguridads"]

    #print(results[0].boxes.conf,"ver resultado")

   #details
    
    result_confidences = results[0].boxes.conf.tolist() 
    promedio_confianzas_response = sum(result_confidences) / len(labels)
    if promedio_confianzas_response > 0.75:
     prediction_response = "verdadero"
    else:
     prediction_response = "falso"
    print(promedio_confianzas_response,"porcentaje")

    #print(result_confidences,"confidences")
    details_dict_response = {label: confidence for label, confidence in zip(labels, result_confidences)}

    print(details_dict_response,"lista")
    
    


    #JSON
    response_data={
    "value":value_response,  
    "edition":year_response,
    "percentage":promedio_confianzas_response,
    "prediction":prediction_response,
    "details":details_dict_response
    }
    


    return JSONResponse(content=response_data)

@prediction.post("/predictions/orb/soles", tags=["Prediction"])
async def predict_soles_veracity(file: UploadFile = File(...)):
    contents = await file.read()
    imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    path="services/dataset"
    images=[]
    classNames=[]

    myList=os.listdir(path)
    print(myList)

    for img in myList:
     if img.endswith(('.png', '.jpg', '.jpeg')):
        imgCur = preprocessing_algorithm2(f'{path}/{img}')
        images.append(imgCur)
        classNames.append(os.path.splitext(img)[0])
    print(classNames)

    img2=preprocessing_algorithm(imaget)
    desList = findDescription(images)
    matchList = findRelevantKeypoints(img2, desList)
    id, accuracy = findMax(matchList)
    print(accuracy)
    print(classNames[id])
    #concatenar
    string = classNames[id]
    matches = re.match(r"(\d+)([^\d]+)_(\d+)", string)
    if matches:
     value = matches.group(1) +" " +matches.group(2)
     year = matches.group(3)

     print(f"Value: {value}")
     print(f"Year: {year}")
    else:
     print("No se encontraron coincidencias.")  

    #json
    response_data={
    "edition":year,
    "value":value,
    "percentage": accuracy
    }
    return JSONResponse(content=response_data)


# Similarmente, para la ruta de predicción de dólares...



# @prediction.get("/predictions", tags=["Prediction"])
# def get_prediction():
#     predict = "Verdadero"

#     labels = ['100 soles', '100 soles falsos', '20 soles', '20 soles falsos']
    
#     imaget_path = "images/20soles_4.jpg"
#     imaget=cv2.resize(cv2.imread(imaget_path), (224, 224), interpolation = cv2.INTER_AREA)


#     xt = np.asarray(imaget)
#     xt = preprocess_input(xt)
#     xt = np.expand_dims(xt,axis=0)
#     preds = model.predict(xt)

#     return labels[np.argmax(preds)]