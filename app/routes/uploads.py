from fastapi import APIRouter, UploadFile
from app.utils.validate import validate_file_size_type
from app.utils.file_handler import save_upload_file, remove_upload_file
from app.utils.image_utils import crop_and_save_upload_file
from pathlib import Path
# import shutil
import random
# from os import path as p

current_dir = Path(__file__).parent
UPLOAD_DIR = current_dir.parent / 'uploads'
# UPLOAD_PATH = p.join(p.dirname(p.realpath(__file__)), "uploads/")
# save_to = UPLOAD_DIR / 'sadasdsd'
# print(save_to)

uploads_router = APIRouter()

@uploads_router.post("/upload-images")
async def create_upload_images(files: list[UploadFile]):
    for index, file in enumerate(files):
        validate_file_size_type(file)
        file_location = UPLOAD_DIR / f'{index}_{file.filename}'
        # file_location = f"uploads/{file.filename}"
        # file_location = p.join(UPLOAD_PATH, file.filename)
        # print(file_location)
        # file.file.seek(0) # Seek to the beginning of the file object --> seek after validate
        # with open(file_location, "wb+") as file_object:
        #     shutil.copyfileobj(file.file, file_object)
        save_upload_file(file, file_location)
        crop_and_save_upload_file(file_location)

    return {"result": f"{random.randint(0, 100)}", "filenames": [file.filename for file in files]}