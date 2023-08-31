from typing import Union

from fastapi import FastAPI
from service.api.api import main_router
app = FastAPI()

app.include_router(main_router)
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: Union[str, None] = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}
