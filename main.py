from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.currency import currency
from routes.detection import detection
from routes.predictions import prediction

tags_metadata = [
    {
        "name": "Currency",
        "description": "Operaciones para listar y guardar los tipos de monedas permitidos.",
    },
    {
        "name": "Detection",
        "description": "Operaciones para guardar las detecciones hechas por la clientes tras recibir los resultados de la predicción.",
    },
    {
        "name": "Prediction",
        "description": "Métodos para la predicción de billetes auténticos basada en el modelo entrenado de red neuronal convolucional.",
    },
]

app = FastAPI(
    title = "Counterfeit Detection API",
    version = "1.0",
    summary = "API para la detección de billetes falsos (Soles peruanos y dolares USD) basada en procesamiento de imágenes y redes neuronales convolucionales.",
    openapi_tags = tags_metadata)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(currency)
app.include_router(detection)
app.include_router(prediction)

if __name__ == "__main__":
     import uvicorn
     uvicorn.run(app, host="0.0.0.0", port=8000)

# source dev-env/bin/activate
# uvicorn main:app --reload
# deactivate