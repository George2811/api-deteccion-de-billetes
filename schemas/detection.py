from typing import Optional
from pydantic import BaseModel
from datetime import date

class Detection(BaseModel):
    id: Optional[int] = None
    user_id: int
    detection_date: date
    currency_id: int
    classification: str
    percentage: float
    image_id: int