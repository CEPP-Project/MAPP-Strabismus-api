from functools import lru_cache
from pydantic_settings import BaseSettings
from app.config.config import get_settings

setting = get_settings()
cls_L_dict: dict = {0:'L1',1:'L2',2:'L3',3:'L4',4:'L5',5:'L6',6:'L7',7:'L8',8:'L9'}
cls_M_dict: dict = {0:'M1',1:'M2',2:'M3',3:'M4',4:'M5',5:'M6',6:'M7',7:'M8',8:'M9'}   
cls_R_dict: dict = {0:'R1',1:'R2',2:'R3',3:'R4',4:'R5',5:'R6',6:'R7',7:'R8',8:'R9'}

class Settings(BaseSettings):
    models_paths: dict = {'left': setting.MODEL_L_PATH, 
                    'mid': setting.MODEL_M_PATH, 
                    'right': setting.MODEL_R_PATH}

    cls_dicts: dict = {'left':cls_L_dict,
                 'mid':cls_M_dict, 
                 'right':cls_R_dict}

# @lru_cache
def get_model_setting():
    return Settings()

def reload_model_settings():
    get_model_setting.cache_clear()