from fastapi import APIRouter
from config.db import conn
from model.currency import currencies
from schemas.currency import Currency

currency = APIRouter()

@currency.get("/currencies", tags=["Currency"])
def get_currencies():
    result = conn.execute(currencies.select()).fetchall()
    return result

@currency.post("/currencies", tags=["Currency"])
def save_currency(cur: Currency):
    new_currency = {'name': cur.name}
    result = conn.execute(currencies.insert().values(new_currency))
    return "Se guadó"