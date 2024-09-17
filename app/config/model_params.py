from functools import lru_cache
from pydantic_settings import BaseSettings

# class dictionaries
cls_L_dict: dict = {0:'L1',1:'L2',2:'L3',3:'L4',4:'L5',5:'L6',6:'L7',7:'L8',8:'L9'}
cls_M_dict: dict = {0:'M1',1:'M2',2:'M3',3:'M4',4:'M5',5:'M6',6:'M7',7:'M8',8:'M9'}   
cls_R_dict: dict = {0:'R1',1:'R2',2:'R3',3:'R4',4:'R5',5:'R6',6:'R7',7:'R8',8:'R9'}

#new ratio
ratio_com: list = ['image_name','strabismus',
             'MR_rt1','MR_rt2','MR_rt3','MR_rt4','MR_rt5',
             'ML_rt1','ML_rt2','ML_rt3','ML_rt4','ML_rt5',
             'LR_rt1','LR_rt2','LR_rt3','LR_rt4','LR_rt5',
             'LL_rt1','LL_rt2','LL_rt3','LL_rt4','LL_rt5',
             'RR_rt1','RR_rt2','RR_rt3','RR_rt4','RR_rt5', 
             'RL_rt1','RL_rt2','RL_rt3','RL_rt4','RL_rt5',] 

#final columns name
col_names: list = ['image_name','strabismus',
'L1-L','L2-L','L3-L','L4-L','L5-L','L6-L','L7-L','L8-L','LC-L',
'M1-L','M2-L','M3-L','M4-L','M5-L','M6-L','M7-L','M8-L','MC-L',
'R1-L','R2-L','R3-L','R4-L','R5-L','R6-L','R7-L','R8-L','RC-L',
'L1-R','L2-R','L3-R','L4-R','L5-R','L6-R','L7-R','L8-R','LC-R',
'M1-R','M2-R','M3-R','M4-R','M5-R','M6-R','M7-R','M8-R','MC-R',
'R1-R','R2-R','R3-R','R4-R','R5-R','R6-R','R7-R','R8-R','RC-R']

class Settings(BaseSettings):
    cls_dicts: dict = {'left':cls_L_dict,
                 'mid':cls_M_dict, 
                 'right':cls_R_dict}
    ratio_com: list = ratio_com
    col_names: list = col_names

# @lru_cache
def get_model_settings():
    return Settings()

# def reload_model_settings():
#     get_model_settings.cache_clear()