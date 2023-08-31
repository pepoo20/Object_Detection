from fastapi import     HTTPException
from PIL import Image
from io import BytesIO
import numpy as np


def take_request(im):
    if im.filename.split(".")[-1] in ("jpg" , "png" , "jpeg"):
        pass
    else:
        raise HTTPException(
            status_code=415 , detail= "not an image"
        )
    image = Image.open(BytesIO(im.file.read()))
    image_arr = np.array(image)
    return image_arr