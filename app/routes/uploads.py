from fastapi import APIRouter, UploadFile, Depends, Header, HTTPException, status
from typing import Annotated
import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from app.utils.files.validate import validate_file_size_type
from app.utils.files.file_handler import save_upload_file, remove_upload_file
from app.utils.files.image_utils import crop_and_save_upload_file
from app.utils.model_predict import RTDETR_prediction, prediction_preprocess, create_empty_df, final_preprocess, add_eye_ratio_com, read_latest_file, predict_strabismus, procrustes_1d_nan
from app.config.config import get_settings
from app.config.model_params import get_model_settings
from app.db.database import get_db
from app.utils.auth.auth_bearer import decodeJWT
import app.db.models as models

uploads_router = APIRouter(
    prefix='/uploads',
    tags=['Uploads']
)

current_dir = Path(__file__).parent
UPLOAD_DIR = current_dir.parent / 'uploads'

model_settings = get_model_settings()
settings = get_settings()

@uploads_router.post('/detect') # change /upload-images to /detect --> Path: /uploads/detect
async def detect_strabismus(files: list[UploadFile], authorization: Annotated[str | None, Header()] = None ,db: Session = Depends(get_db)):

    # eye images need to reset everytime
    eye_img = []

    for index, file in enumerate(files):
        validate_file_size_type(file)
        file_location = UPLOAD_DIR / f'{index}_{file.filename}'   

        save_upload_file(file, file_location)
        eye_img.append(str(file_location))

    try:
        print('eye_img', eye_img)

        # load images from mobile application into patches
        pathes_L = [[eye_img[0],'App-L-LE'],[eye_img[1],'App-L-RE']]
        pathes_M = [[eye_img[2],'App-M-LE'],[eye_img[3],'App-M-RE']]
        pathes_R = [[eye_img[4],'App-R-LE'],[eye_img[5],'App-R-RE']]

        # predicting 9 landmarks
        pred_L = RTDETR_prediction(settings.MODEL_L,pathes_L,model_settings.cls_dicts['left'])
        pred_M = RTDETR_prediction(settings.MODEL_M,pathes_M,model_settings.cls_dicts['mid'])
        pred_R = RTDETR_prediction(settings.MODEL_R,pathes_R,model_settings.cls_dicts['right'])  

        # preprocessing data I
        prepro1_L = prediction_preprocess(pred_L.copy())
        prepro1_M = prediction_preprocess(pred_M.copy())
        prepro1_R = prediction_preprocess(pred_R.copy())

        # create df for second preprocessing
        pivot_L = create_empty_df(prepro1_L,'L')
        pivot_M = create_empty_df(prepro1_M,'M')
        pivot_R = create_empty_df(prepro1_R,'R')

        # preprocessing data II
        save_name = 'newRTDETR_2_PREPROCESS'
        prepro2_L = final_preprocess(prepro1_L.copy(),pivot_L.copy())
        prepro2_M = final_preprocess(prepro1_M.copy(),pivot_M.copy())
        prepro2_R = final_preprocess(prepro1_R.copy(),pivot_R.copy())
        
        # preprocessing data III
        ### Geometric Priors (mean eye shape) to fill in missing landmarks ###

        dataframe_trimmed_avg = pd.read_pickle(read_latest_file(settings.ML_PATH,'dataframe_trimmed_avg'))
        df_new_data_L = prepro2_L.iloc[:,2:].T
        df_new_data_M = prepro2_M.iloc[:,2:].T
        df_new_data_R = prepro2_R.iloc[:,2:].T
        df_new_data = pd.concat([df_new_data_L,df_new_data_M,df_new_data_R])
        dataframe_trimmed_avg['detect_eye'] = df_new_data

        df1 = dataframe_trimmed_avg.iloc[:, 0:1] 
        df2 = dataframe_trimmed_avg.iloc[:, 1:2] 
        df2['detect_eye'] = pd.to_numeric(df2['detect_eye'], errors='coerce')

        mean_eye_shape = df1.to_numpy()
        detected_landmarks = df2.to_numpy()
        
        filled_landmarks =  procrustes_1d_nan(detected_landmarks, mean_eye_shape)

        combine_df = pd.DataFrame(filled_landmarks[:,0],index = df1.index).T
        combine_df.insert(0,'image_name','App')
        combine_df.insert(1,'strabismus',prepro2_M['strabismus'])

        # preprocessing data IV
        final_combine = combine_df.copy()
        final_combine = add_eye_ratio_com(final_combine,settings.ML_PATH)
        final_combine = final_combine[model_settings.ratio_com]
        final_combine.dropna(inplace=True)

        # predict strabismus with top MLs
        preidct_image = final_combine.head(1).drop(columns=['image_name','strabismus']).copy()
        topml = pd.read_pickle(read_latest_file(settings.ML_PATH,'topml[DT-RF-KNN]'))
        result = predict_strabismus(preidct_image,topml) #result = [True/False, %AvgConfidence]
        if result[0] == 'Undetectable':
            return {'error': 'Undetectable'}
        
        print("strabismus result =", result)
        result = [bool(result[0]), [float(x) for x in result[1]]]

    # Error logging
    except KeyError as error:
        print('--- Error ---')
        print('Error name:', error.__class__.__name__)
        print(error)
        print('-------------')
        return {'error': 'Can not detect eye.'}
    except Exception as error:
        print('--- Error ---')
        print('Error name:', error.__class__.__name__)
        print(error)
        print('-------------')
        return {'error': 'Cannot Detect Landmarks'}

    if authorization is not None:
        payload = decodeJWT(authorization.split()[1]) # split Bearer
        print(payload)
        if payload is None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Invalid token.')
        new_history = models.Historys(result=result, user_id=payload['sub'])
        db.add(new_history)
        db.commit()
        db.refresh(new_history)

    return {'result': result}