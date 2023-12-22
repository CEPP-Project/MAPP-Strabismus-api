from fastapi import HTTPException, status
from typing import IO

def validate_file_size_type(file: IO):
    FILE_SIZE = 2097152 # 2MB

    if file.filename.split('.')[-1] not in ["png", "jpeg", "jpg"] or file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type",
        )
    
    real_file_size = 0
    for chunk in file.file:
        real_file_size += len(chunk)
        if real_file_size > FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
                detail="File is too large"
            )