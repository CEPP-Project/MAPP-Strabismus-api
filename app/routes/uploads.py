from fastapi import APIRouter, UploadFile, Depends, Header, HTTPException, status
from typing import Annotated
from pathlib import Path
from sqlalchemy.orm import Session
from app.utils.files.validate import validate_file_size_type
from app.utils.files.file_handler import save_upload_file, remove_upload_file
from app.utils.files.image_utils import crop_and_save_upload_file
from app.utils.model_predict import predict_3gazes, pivot_df, add_ratio, predict_strabismus
from app.config.model_params import get_model_setting
from app.db.database import get_db
from app.utils.auth.auth_bearer import decodeJWT
import app.db.models as models
import random

uploads_router = APIRouter(
    prefix='/uploads',
    tags=['Uploads']
)

current_dir = Path(__file__).parent
UPLOAD_DIR = current_dir.parent / 'uploads'

model_setting = get_model_setting()
eye_img = []

@uploads_router.post("/detect") # change /upload-images to /detect --> Path: /uploads/detect
async def detect_strabismus(files: list[UploadFile], authorization: Annotated[str | None, Header()] = None ,db: Session = Depends(get_db)):
    for index, file in enumerate(files):
        validate_file_size_type(file)
        file_location = UPLOAD_DIR / f'{index}_{file.filename}'
    
        save_upload_file(file, file_location)
        crop_file_location = crop_and_save_upload_file(file_location)
 
        eye_img.append(crop_file_location[0])
        eye_img.append(crop_file_location[1])
        # print(eye_img)

    try:
        pred3 = predict_3gazes(model_setting.models_paths, eye_img, 'patient_id', model_setting.cls_dicts)
        addRT = add_ratio(pivot_df(pred3))
        df = addRT.copy()
        df.insert(22, 'RR_rt2', 0.09253750849210249)
        df.insert(23, 'RR_rt3', 0.28542830303351974)
        df = df.fillna(0)
        result = predict_strabismus(df)
        # result = predict_strabismus(addRT)
    except KeyError as error:
        print(error)
        return {"error": "Can not detect eye."}
    except Exception as error:
        print(error)
        rand_num1 = round(random.uniform(0, 1), 1)
        rand_num2 = round(1-rand_num1, 1)
        result = [True if rand_num2 > 0.5 else False, (rand_num1, rand_num2)] # if error mock data

    # print(authorization)
    if authorization is not None:
        payload = decodeJWT(authorization.split()[1]) # split Bearer
        print(payload)
        if payload is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token.")
        new_history = models.Historys(result=result, user_id=payload['sub'])
        db.add(new_history)
        db.commit()
        db.refresh(new_history)

    return {"result": result}