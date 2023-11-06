from fastapi import APIRouter, File, UploadFile
import numpy as np
import cv2
from services.cnn_service import preprocessing_algorithm
from ultralytics import YOLO
import os
from utils.preprocessing import preprocessing_algorithm2
from utils.orb_functions import findRelevantKeypoints
from utils.orb_functions import findMax
from utils.orb_functions import findDescription
import re
from fastapi.responses import JSONResponse

prediction = APIRouter()

# Cargar modelo PT
model_yolo_2021 = YOLO("services/2021.pt")
model_yolo_2009 = YOLO("services/2009.pt")
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

    for img in myList:
     if img.endswith(('.png', '.jpg', '.jpeg')):
        imgCur = preprocessing_algorithm2(f'{path}/{img}')
        images.append(imgCur)
        classNames.append(os.path.splitext(img)[0])

    img2=preprocessing_algorithm(imaget)
    desList = findDescription(images)
    matchList = findRelevantKeypoints(img2, desList)
    id, accuracy = findMax(matchList)

    #concatenar
    string = classNames[id]
    matches = re.match(r"(\d+)([^\d]+)_(\d+)", string)
    if matches:
     value_response = matches.group(1) +" " +matches.group(2)
     year_response = matches.group(3)
    else:
     print("No se encontraron coincidencias.")  
  
    
    # Realizar la predicción
    results_2021 = model_yolo_2021.predict(imaget, imgsz=640, conf=0.1)
    results_2009 = model_yolo_2009.predict(imaget, imgsz=640, conf=0.1)


   #details
    names_2021 = list(results_2021[0].names.values())
    names_2009 = list(results_2009[0].names.values())


    probs_2021 = [i.item() for i in results_2021[0].boxes.conf]
    probs_2009 = [i.item() for i in results_2009[0].boxes.conf]

    promedio_confianzas_response_2021 = sum(probs_2021) / len(names_2021)
    promedio_confianzas_response_2009 = sum(probs_2009) / len(names_2009)

    if(promedio_confianzas_response_2021 > promedio_confianzas_response_2009):
       promedio_confianzas_response = promedio_confianzas_response_2021
       classes = [i.item() for i in results_2021[0].boxes.cls]
       
       response = {}
       for i, n in enumerate(names_2021):
        if i in classes:
            response[n] = probs_2021[classes.index(i)]
        else:
            response[n] = 0.00

            
        if promedio_confianzas_response_2021 > 0.75:
            prediction_response = "verdadero"
        else:
            prediction_response = "falso"

    else:
       promedio_confianzas_response = promedio_confianzas_response_2009
       classes = [i.item() for i in results_2009[0].boxes.cls]
       
       response = {}
       for i, n in enumerate(names_2009):
        if i in classes:
            response[n] = probs_2009[classes.index(i)]
        else:
            response[n] = 0.00

            
        if promedio_confianzas_response_2009 > 0.75:
            prediction_response = "verdadero"
        else:
            prediction_response = "falso"

     
    #JSON
    response_data = {
        "value":value_response,  
        "edition":year_response,
        "percentage":promedio_confianzas_response,
        "prediction":prediction_response,
        "details":response
    }

    return JSONResponse(content=response_data)

@prediction.post("/predictions/orb/soles", tags=["Prediction"])
async def predict_soles_veracity_orb(file: UploadFile = File(...)):
    contents = await file.read()
    imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    path="services/dataset"
    images=[]
    classNames=[]

    myList=os.listdir(path)
    #print(myList)

    for img in myList:
     if img.endswith(('.png', '.jpg', '.jpeg')):
        imgCur = preprocessing_algorithm2(f'{path}/{img}')
        images.append(imgCur)
        classNames.append(os.path.splitext(img)[0])
    #print(classNames)

    img2=preprocessing_algorithm(imaget)
    desList = findDescription(images)
    matchList = findRelevantKeypoints(img2, desList)
    id, accuracy = findMax(matchList)
    print(accuracy)
    #print(classNames[id])
    #concatenar
    string = classNames[id]
    matches = re.match(r"(\d+)([^\d]+)_(\d+)", string)
    if matches:
     value = matches.group(1) +" " +matches.group(2)
     year = matches.group(3)

    #  print(f"Value: {value}")
    #  print(f"Year: {year}")
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

# async def predict_soles_veracity(file: UploadFile = File(...)):

#     #imaget
#     contents = await file.read()
#     imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)


#     #orb
#     path="services/dataset"
#     images=[]
#     classNames=[]
#     myList=os.listdir(path)
#     #print(myList)
#     for img in myList:
#      if img.endswith(('.png', '.jpg', '.jpeg')):
#         imgCur = preprocessing_algorithm2(f'{path}/{img}')
#         images.append(imgCur)
#         classNames.append(os.path.splitext(img)[0])
#     #print(classNames)

#     img2=preprocessing_algorithm(imaget)
#     desList = findDescription(images)
#     matchList = findRelevantKeypoints(img2, desList)
#     id, accuracy = findMax(matchList)
#     #print(accuracy)
#     #print(classNames[id])
#     #concatenar
#     string = classNames[id]
#     matches = re.match(r"(\d+)([^\d]+)_(\d+)", string)
#     if matches:
#      value_response = matches.group(1) +" " +matches.group(2)
#      year_response = matches.group(3)

#      #print(f"Value: {value_response}")
#      #print(f"Year: {year_response}")
#     else:
#      print("No se encontraron coincidencias.")  
  
    
#     # Realizar la predicción
#     if year_response=="2021":
#        results = model_yolo_2021.predict(imaget, imgsz=640, conf=0.0)
#        labels = ["hilo_de_seguridad", "marca_de_agua", "microimpresiones", "figuras_en_movimiento"] 
#     if year_response=="2009":
#        results = model_yolo_2009.predict(imaget, imgsz=640, conf=0.0)
#        labels = ["marca_de_agua", "microimpresiones", "numero_oculto", "hilo_de_seguridad"]

#     #print(results[0].boxes.conf,"ver resultado")

#    #details
    
#     result_confidences = results[0].boxes.conf.tolist() 
#     promedio_confianzas_response = sum(result_confidences) / len(labels)
#     if promedio_confianzas_response > 0.75:
#      prediction_response = "verdadero"
#     else:
#      prediction_response = "falso"
#     #print(promedio_confianzas_response,"porcentaje")

#     #print(result_confidences,"confidences")
#     details_dict_response = {label: confidence for label, confidence in zip(labels, result_confidences)}

#     #print(details_dict_response,"lista")
     
#     #JSON
#     response_data = {
#         "value":value_response,  
#         "edition":year_response,
#         "percentage":promedio_confianzas_response,
#         "prediction":prediction_response,
#         "details":details_dict_response
#     }

#     return JSONResponse(content=response_data)