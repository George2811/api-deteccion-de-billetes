from fastapi import APIRouter
from config.db import conn
from model.currency import currencies
from schemas.currency import Currency

currency = APIRouter()

@currency.get("/currencies/{id}", tags=["Currency"])
async def get_currenciey_by_id(id:int):
    result = conn.execute(currencies.select().where(currencies.c.id == id)).first()
    return result or "Id no encontrado"

@currency.get("/currencies", tags=["Currency"])
async def get_currencies():
    result = conn.execute(currencies.select()).fetchall()
    return result

@currency.post("/currencies", tags=["Currency"])
async def save_currency(cur: Currency):
    new_currency = {'name': cur.name}
    result = conn.execute(currencies.insert().values(new_currency))
    return "Se guardó"