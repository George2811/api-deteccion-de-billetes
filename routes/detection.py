from fastapi import APIRouter
from config.db import conn
from model.detection import detections
from schemas.detection import Detection

detection = APIRouter()

@detection.get("/detections/{id}", tags=["Detection"])
async def get_detection_by_id(id: int):
    result = conn.execute(detections.select().where(detections.c.id == id)).first()
    return result

@detection.get("/detections/user/{user_id}", tags=["Detection"])
async def get_detections_by_user(user_id: int):
    result = conn.execute(detections.select().where(detections.c.user_id == user_id)).fetchall()
    return result

@detection.post("/detections", tags=["Detection"])
async def save_detection(register: Detection):
    new_register = {'user_id': register.user_id,
                    'detection_date': register.detection_date,
                    'currency_id': register.currency_id,
                    'classification': register.classification,
                    'percentage': register.percentage,
                    'image_url': register.image_url}
    
    add_register = conn.execute(detections.insert().values(new_register))
    result = conn.execute(detections.select().where(detections.c.user_id == register.user_id, detections.c.id == add_register.lastrowid)).first()
    return result

@detection.delete("/detections/user/{user_id}/{id}", tags=["Detection"])
async def delete_detection(user_id: int, id:int):
    #Mejorar el delete detection
    remove = conn.execute(detections.delete().where(detections.c.user_id == user_id, detections.c.id == id))
    return f"Se eliminó el registro del usuario con id {user_id}"