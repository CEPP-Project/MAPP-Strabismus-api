from fastapi import APIRouter, UploadFile
from app.utils.validate import validate_file_size_type
import random

uploads_router = APIRouter()

@uploads_router.post("/upload-images")
async def create_upload_images(files: list[UploadFile]):
    for file in files:
        validate_file_size_type(file)

    return {"result": f"{random.randint(0, 100)}", "filenames": [file.filename for file in files]}