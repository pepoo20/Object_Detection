from fastapi import APIRouter
from service.api.endpoints.Detect import detect_router


main_router = APIRouter()
main_router.include_router(detect_router)
