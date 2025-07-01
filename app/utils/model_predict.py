import os
import pickle
import pandas as pd
import numpy as np
from ultralytics import RTDETR

# Gaze : L M R 
# Eye : LE RE

def RTDETR_prediction(model_path,images_url,cls_dict):
    model = RTDETR(model_path)
    append_df = pd.DataFrame(columns=['image_name','xmin', 'ymin','xmax', 'ymax','confidence','class','cls_name'])
    for url,name in images_url:
# detect eye landmarks
        detect_im = model.predict(url)
        crash = None
        j = 0
        while crash == None:
            try:
# extract each landmarks
                boxj = detect_im[0].boxes[j].xyxy[0].cpu().detach().numpy()
                clsj = detect_im[0].boxes[j].cls.cpu().detach().numpy()[0]
                confj = detect_im[0].boxes[j].conf.cpu().detach().numpy()[0]
# update & save
                append_df = update_df(append_df,cls_dict,name,boxj,clsj,confj)

                j += 1
            except:
                crash = 1
    return append_df

# find the most recent version of model
def read_latest_file(file_path,file_name):
    i = 0
    existing_file = os.path.join(file_path,file_name)
    while os.path.exists(existing_file+'-v%s.pkl'%i):
        i+=1
    file_name = os.path.join(file_path,existing_file+'-v%s.pkl'%(i-1))
    return file_name

def update_df(df,cls_dict,name,box,cls,conf):
    # create temp dataframe
    new_df = pd.Series({'image_name':name,
                        'xmin' : box[0],
                        'ymin' : box[1],
                        'xmax' : box[2],
                        'ymax' : box[3],
                        'confidence' : conf,
                        'class' : int(cls),
                        'cls_name' :cls_dict[int(cls)] })
    
    # append temp df to main df
    df = pd.concat([df, pd.DataFrame([new_df], columns=new_df.index)]).reset_index(drop=True)
    return df

def prediction_preprocess(prediction,file_name=None):
    df = prediction.copy()
    df['xmid'] = round((df['xmax']+df['xmin'])/2)
    df['ymid'] = round((df['ymax']+df['ymin'])/2)
    df['mid_xy'] = list(zip(df['xmid'],df['ymid']))
    df.reset_index(inplace=True)
    df= df.drop(columns=['index'])
    df = df.drop(columns=['xmin','ymin','xmax','ymax','xmid','ymid','class'])
    df.rename(columns = {'name':'cls_name'}, inplace = True)
    df.rename(columns = {'image_no':'image_name'}, inplace = True)
    df['strabismus'] = df['image_name'].str.len()>10
    return df

def create_empty_df(df,gaze):
    # remove left & right eyes suffixes
    replace_name = df['image_name'].str.replace('-'+gaze+'-RE','')
    replace_name = replace_name.str.replace('-'+gaze+'-LE','')
    
    # create empty df matching preprocess df
    empty_df = pd.DataFrame(columns=['image_name','strabismus',
                                     gaze+'1-L',gaze+'2-L',gaze+'3-L',gaze+'4-L',gaze+'5-L',gaze+'6-L',gaze+'7-L',gaze+'8-L',gaze+'C-L',
                                     gaze+'1-R',gaze+'2-R',gaze+'3-R',gaze+'4-R',gaze+'5-R',gaze+'6-R',gaze+'7-R',gaze+'8-R',gaze+'C-R'         
                                    ])
    empty_df['image_name'] = replace_name
    empty_df['strabismus'] = df['strabismus']
    empty_df = empty_df.drop_duplicates()
    empty_df.reset_index(inplace=True)
    empty_df = empty_df.drop(columns=['index'])
    return empty_df

def add_eye_ratio_com(df,save_path=None):
    gazes = ['M','L','R']
    ratioL = [                 
                 ['L_rt1','7-L','1-L'],
                 ['L_rt2','8-L','7-L'],
                 ['L_rt3','4-L','8-L'],
                 ['L_rt4','C-L','7-L'],
                 ['L_rt5','8-L','C-L'],    
              ]
    ratioR = [              
                 ['R_rt1','7-R','1-R'],
                 ['R_rt2','8-R','7-R'],
                 ['R_rt3','4-R','8-R'],
                 ['R_rt4','C-R','7-R'],
                 ['R_rt5','8-R','C-R'],
            ]
    for g in gazes:
        df[str(g+'WID-L')] = df[g+'4-L'] - df[g+'1-L']
        df[str(g+'WID-L')] = df[str(g+'WID-L')].fillna(0).replace(0,None)
        df[str(g+'WID-R')] = df[g+'4-R'] - df[g+'1-R']
        df[str(g+'WID-R')] = df[str(g+'WID-R')].fillna(0).replace(0,None)        
        for ratio, upper, lower in ratioL:
            try:
                df[str(g+ratio)]= (df[str(g+upper)]-df[str(g+lower)])/df['MWID-L']
            except:
                pass
        for ratio, upper, lower in ratioR:
            try:
                df[str(g+ratio)]= (df[str(g+upper)]-df[str(g+lower)])/df['MWID-R']
            except:
                pass  
    return df

def final_preprocess(old_df,new_df,file_name=None):
    for i,row in old_df.iterrows():
        # get index that have matching image_name between old_df and new_df
        index = new_df[new_df['image_name'] == row['image_name'][:-5]].index.values.astype(int)[0]    
        image_name = row['image_name']
        strabismus = row['strabismus']
        cls_name = row['cls_name']
        mid_xy = row['mid_xy']
        x = mid_xy[0]
        y = mid_xy[1]

        if '-RE' in image_name:
            if cls_name == 'L1':
                new_df.loc[index, 'L1-L'] = x
            elif cls_name == 'L2':
                new_df.loc[index, 'L2-L'] = x
            elif cls_name == 'L3':
                new_df.loc[index, 'L3-L'] = x
            elif cls_name == 'L4':
                new_df.loc[index, 'L4-L'] = x
            elif cls_name == 'L5':
                new_df.loc[index, 'L5-L'] = x
            elif cls_name == 'L6':
                new_df.loc[index, 'L6-L'] = x
            elif cls_name == 'L7':
                new_df.loc[index, 'L7-L'] = x
            elif cls_name == 'L8':
                new_df.loc[index, 'L8-L'] = x 
            elif cls_name == 'L9':
                new_df.loc[index, 'LC-L'] = x      
            elif cls_name == 'LC':
                new_df.loc[index, 'LC-L'] = x   
            elif cls_name == 'M1':
                new_df.loc[index, 'M1-L'] = x
            elif cls_name == 'M2':
                new_df.loc[index, 'M2-L'] = x
            elif cls_name == 'M3':
                new_df.loc[index, 'M3-L'] = x
            elif cls_name == 'M4':
                new_df.loc[index, 'M4-L'] = x
            elif cls_name == 'M5':
                new_df.loc[index, 'M5-L'] = x
            elif cls_name == 'M6':
                new_df.loc[index, 'M6-L'] = x
            elif cls_name == 'M7':
                new_df.loc[index, 'M7-L'] = x
            elif cls_name == 'M8':
                new_df.loc[index, 'M8-L'] = x
            elif cls_name == 'M9':
                new_df.loc[index, 'MC-L'] = x
            elif cls_name == 'MC':
                new_df.loc[index, 'MC-L'] = x
            elif cls_name == 'R1':
                new_df.loc[index, 'R1-L'] = x
            elif cls_name == 'R2':
                new_df.loc[index, 'R2-L'] = x
            elif cls_name == 'R3':
                new_df.loc[index, 'R3-L'] = x
            elif cls_name == 'R4':
                new_df.loc[index, 'R4-L'] = x
            elif cls_name == 'R5':
                new_df.loc[index, 'R5-L'] = x
            elif cls_name == 'R6':
                new_df.loc[index, 'R6-L'] = x
            elif cls_name == 'R7':
                new_df.loc[index, 'R7-L'] = x
            elif cls_name == 'R8':
                new_df.loc[index, 'R8-L'] = x
            elif cls_name == 'R9':
                new_df.loc[index, 'RC-L'] = x
            elif cls_name == 'RC':
                new_df.loc[index, 'RC-L'] = x
            else:
                print('fail at',index,image_name,cls_name)
            # print(image_name,'& RE')
            # print(index,'&',cls_name)
            # print()
        elif '-LE' in image_name:
            if cls_name == 'L1':
                new_df.loc[index, 'L1-R'] = x
            elif cls_name == 'L2':
                new_df.loc[index, 'L2-R'] = x
            elif cls_name == 'L3':
                new_df.loc[index, 'L3-R'] = x
            elif cls_name == 'L4':
                new_df.loc[index, 'L4-R'] = x
            elif cls_name == 'L5':
                new_df.loc[index, 'L5-R'] = x
            elif cls_name == 'L6':
                new_df.loc[index, 'L6-R'] = x
            elif cls_name == 'L7':
                new_df.loc[index, 'L7-R'] = x
            elif cls_name == 'L8':
                new_df.loc[index, 'L8-R'] = x 
            elif cls_name == 'L9':
                new_df.loc[index, 'LC-R'] = x     
            elif cls_name == 'LC':
                new_df.loc[index, 'LC-R'] = x     
            elif cls_name == 'M1':
                new_df.loc[index, 'M1-R'] = x
            elif cls_name == 'M2':
                new_df.loc[index, 'M2-R'] = x
            elif cls_name == 'M3':
                new_df.loc[index, 'M3-R'] = x
            elif cls_name == 'M4':
                new_df.loc[index, 'M4-R'] = x
            elif cls_name == 'M5':
                new_df.loc[index, 'M5-R'] = x
            elif cls_name == 'M6':
                new_df.loc[index, 'M6-R'] = x
            elif cls_name == 'M7':
                new_df.loc[index, 'M7-R'] = x
            elif cls_name == 'M8':
                new_df.loc[index, 'M8-R'] = x
            elif cls_name == 'M9':
                new_df.loc[index, 'MC-R'] = x
            elif cls_name == 'MC':
                new_df.loc[index, 'MC-R'] = x
            elif cls_name == 'R1':
                new_df.loc[index, 'R1-R'] = x
            elif cls_name == 'R2':
                new_df.loc[index, 'R2-R'] = x
            elif cls_name == 'R3':
                new_df.loc[index, 'R3-R'] = x
            elif cls_name == 'R4':
                new_df.loc[index, 'R4-R'] = x
            elif cls_name == 'R5':
                new_df.loc[index, 'R5-R'] = x
            elif cls_name == 'R6':
                new_df.loc[index, 'R6-R'] = x
            elif cls_name == 'R7':
                new_df.loc[index, 'R7-R'] = x
            elif cls_name == 'R8':
                new_df.loc[index, 'R8-R'] = x
            elif cls_name == 'R9':
                new_df.loc[index, 'RC-R'] = x
            elif cls_name == 'RC':
                new_df.loc[index, 'RC-R'] = x
            else:
                print('fail at',index,image_name,cls_name)
            # print(image_name,'& LE')
            # print(index,'&',cls_name)
            # print()
    
    return new_df

def predict_strabismus(pred_input,models):
    if len(pred_input.columns) < 30:
        print('invalid number of input X_test, need 30 got %s'%len(pred_input.columns))
        strabismus = 'Undetectable'
        return [strabismus, [0,0]]
    
    all_predict_results = []
    all_predict_proba = []

    for i in models:
        try:
            model = i['model']

            # Predict result and probabilities
            predict_result = model.predict(pred_input)
            predict_proba = model.predict_proba(pred_input)
            print(predict_proba)
            # Store all predict result and probabilities

            if 0.2 <= abs(predict_proba[0][0]-predict_proba[0][1]):
                all_predict_results.append(predict_result)
                all_predict_proba.append(predict_proba)
                # print(i['model_name'],'predict_result:',predict_result)
                # print(i['model_name'],'predict_proba:', predict_proba[0])

        except:
            pass
    # calculate average probabilities 
    all_predict_proba = np.array(all_predict_proba)
    average_proba = np.mean(all_predict_proba, axis=0)
    
    strabismus = average_proba[0][1]>average_proba[0][0]
    return [strabismus, average_proba.tolist()[0]]

# Geometric Priors (mean eye shape) to fill in missing landmarks
# Geometric Priors (mean eye shape) to fill in missing landmarks
def procrustes_1d_nan(target, source):
    """
    Aligns 1D array 'source' to 1D array 'target' (with possible NaNs) using
    optimal scale + translation. Returns aligned source and disparity.
    Only positions where both arrays are valid (non-NaN) are used.
    """
    target = np.asarray(target).flatten()
    source = np.asarray(source).flatten()

    if target.shape != source.shape:
        raise ValueError("Arrays must have the same shape.")

    # Create mask where both arrays have valid numbers
    valid_mask = ~np.isnan(target) & ~np.isnan(source)

    if valid_mask.sum() == 0:
        raise ValueError("No overlapping valid data points (both arrays have NaNs everywhere).")

    t_valid = target[valid_mask]
    s_valid = source[valid_mask]

    # Center both valid parts
    t_mean = t_valid.mean()
    s_mean = s_valid.mean()
    t_centered = t_valid - t_mean
    s_centered = s_valid - s_mean

    # Compute optimal scaling factor
    numerator = np.sum(t_centered * s_centered)
    denominator = np.sum(s_centered ** 2)
    scale = numerator / denominator if denominator != 0 else 0

    # Align entire source array (including NaNs) using global scale & translation
    aligned_source = (source - s_mean) * scale + t_mean
    aligned_source = aligned_source.reshape(-1, 1)
    
    # Compute disparity only on valid points
    disparity = np.sum((t_valid - aligned_source[valid_mask]) ** 2)
    
    # detect and replace missing landmarks
    detected = target.copy()
    present_mask = ~np.isnan(detected[:])
    detected = detected.reshape(-1, 1)
    # If too many points missing, estimation might not be reliable
    if np.sum(present_mask) < 3:
        print("Too few landmarks to estimate reliably.")
        return detected

    # Replace missing points in detected landmarks with corresponding transformed mean
    detected[~present_mask] = aligned_source[~present_mask]

    return detected