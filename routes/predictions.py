from fastapi import APIRouter, File, UploadFile
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model

import numpy as np
import cv2
#from services.cnn_service import preprocessing_algorithm

prediction = APIRouter()

# Cargar modelo
model = load_model("services/model_ResNet50_v2.h5")

@prediction.get("/predictions", tags=["Prediction"])
def get_prediction():
    predict = "Verdadero"

    labels = ['100 soles', '100 soles falsos', '20 soles', '20 soles falsos']
    
    imaget_path = "images/20soles_4.jpg"
    imaget=cv2.resize(cv2.imread(imaget_path), (224, 224), interpolation = cv2.INTER_AREA)


    xt = np.asarray(imaget)
    xt = preprocess_input(xt)
    xt = np.expand_dims(xt,axis=0)
    preds = model.predict(xt)

    print(labels[np.argmax(preds)])
    print(preds)

    return labels[np.argmax(preds)]


@prediction.post("/predictions/images", tags=["Prediction"])
async def predict_veracity(file: UploadFile = File(...)):
    labels = ['100 soles', '100 soles falsos', '20 soles', '20 soles falsos']

    contents = await file.read()
    imaget = cv2.imdecode(np.fromstring(contents, np.uint8), cv2.IMREAD_COLOR)
    image_resized = cv2.resize(imaget, (224, 224), interpolation=cv2.INTER_AREA)

    #TODO: Para la V2 no olvidar entrenar el modelo con imagenes en escala de grises => image_input = Input(shape=(width_shape, height_shape, 1))
    #preprocess_image = preprocessing_algorithm(image_resized)

    xt = np.asarray(image_resized) #preprocess_image
    xt = preprocess_input(xt)
    xt = np.expand_dims(xt,axis=0)

    preds = model.predict(xt)
    predict = labels[np.argmax(preds)]
    
    result = {
        'prediction': predict,
        'percetange': float(np.max(preds))
    }
    return result