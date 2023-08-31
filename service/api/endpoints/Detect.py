from fastapi import APIRouter , UploadFile 
from service.core.logic.object_boxes import model_predict
from service.api.endpoints.Request import take_request
from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2
import json
detect_router = APIRouter()

@detect_router.post("/detect/")
def detect_endpoints(im: UploadFile):
    img_array = take_request(im)
    img_boxed, labels = model_predict(img_array)
    
    # Convert the image array to bytes
    img_bytes = cv2.imencode(".jpg", img_boxed)[1].tobytes()
    
    # Create a BytesIO stream to hold the image bytes
    img_stream = BytesIO(img_bytes)
    
    # Convert labels to JSON string
    labels_json = json.dumps(labels.tolist())
    
    # Generate the response
    response = StreamingResponse(img_stream, media_type="image/jpeg")
    
    # Add labels as headers to the response
    response.headers["labels"] = labels_json
    
    return response