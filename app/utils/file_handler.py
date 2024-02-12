from fastapi import UploadFile
from pathlib import Path
import shutil

def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as error:
        print("Error occured when saving file :", error)
    finally:
        upload_file.file.close()

def remove_upload_file(file_location: Path) -> None:
    try:
        file_location.unlink()
    except Exception as error:
        print("Error occured when removing file :", error)
        
