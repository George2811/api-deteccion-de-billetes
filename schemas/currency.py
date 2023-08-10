from typing import Optional
from pydantic import BaseModel

class Currency(BaseModel):
    id: Optional[int] = None
    name: str