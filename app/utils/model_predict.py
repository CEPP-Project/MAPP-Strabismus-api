import pickle
import pandas as pd
import numpy as np
from ultralytics import RTDETR
from app.config.config import get_settings

# Gaze : L M R 
# Eye : L R
# ML = M:มองตรง, L:ตาซ้าย
setting = get_settings()

def predict(model_path, image_path, landmark_id, cls_dict):
    temp_df = pd.DataFrame(columns=['image_name','xmin', 'ymin','xmax', 'ymax','confidence','class','cls_name'])
    #load model
    model = RTDETR(model_path)
    detect_landmarks = model.predict(image_path)
    crash = None
    j = 0
    while crash == None:
        try:
            # extract each landmarks
            boxj = detect_landmarks[0].boxes[j].xyxy[0].cpu().detach().numpy()
            clsj = detect_landmarks[0].boxes[j].cls.cpu().detach().numpy()[0]
            confj = detect_landmarks[0].boxes[j].conf.cpu().detach().numpy()[0]  
            # update & save
            new_df = pd.Series({'image_name':landmark_id,
                                'xmin' : boxj[0],
                                'ymin' : boxj[1],
                                'xmax' : boxj[2],
                                'ymax' : boxj[3],
                                'confidence' : confj,
                                'class' : int(clsj),
                                'cls_name' :cls_dict[int(clsj)] })
            # append temp df to main df
            temp_df = pd.concat([temp_df, pd.DataFrame([new_df], columns=new_df.index)]).reset_index(drop=True)     
            j += 1
        except:
            crash = True
    return temp_df

# eye_img = [LL, LR, ML, MR, RL, RR]
def predict_3gazes(models_paths, eye_img, landmark_id, cls_dicts):
    
    landmarks_df = pd.DataFrame(columns=['image_name','xmin', 'ymin','xmax', 'ymax','confidence','class','cls_name'])
    
    predict_df = predict(models_paths['left'], eye_img[0], landmark_id+'LL', cls_dicts['left'])
    landmarks_df = pd.concat([predict_df, landmarks_df]).reset_index(drop=True)

    predict_df = predict(models_paths['left'], eye_img[1], landmark_id+'LR', cls_dicts['left'])
    landmarks_df = pd.concat([predict_df, landmarks_df]).reset_index(drop=True)                

    predict_df = predict(models_paths['mid'], eye_img[2], landmark_id+'ML', cls_dicts['mid'])
    landmarks_df = pd.concat([predict_df, landmarks_df]).reset_index(drop=True)

    predict_df = predict(models_paths['mid'], eye_img[3], landmark_id+'MR', cls_dicts['mid'])
    landmarks_df = pd.concat([predict_df, landmarks_df]).reset_index(drop=True)           

    predict_df = predict(models_paths['right'], eye_img[4], landmark_id+'RL', cls_dicts['right'])
    landmarks_df = pd.concat([predict_df, landmarks_df]).reset_index(drop=True)

    predict_df = predict(models_paths['right'], eye_img[5], landmark_id+'RR', cls_dicts['right'])
    landmarks_df = pd.concat([predict_df, landmarks_df]).reset_index(drop=True)

    print("landmarks_df")
    print(landmarks_df)

    return landmarks_df
        
def prediction_preprocess(prediction,file_name):
    prepro_df = prediction.copy()
    prepro_df['xmid'] = round((prepro_df['xmax']+prepro_df['xmin'])/2)
    prepro_df['ymid'] = round((prepro_df['ymax']+prepro_df['ymin'])/2)
    prepro_df['mid_xy'] = list(zip(prepro_df['xmid'],prepro_df['ymid']))
    prepro_df.reset_index(inplace=True)
    prepro_df= prepro_df.drop(columns=['index'])
    prepro_df = prepro_df.drop(columns=['xmin','ymin','xmax','ymax','xmid','ymid','class'])
    prepro_df.rename(columns = {'name':'cls_name'}, inplace = True)
    prepro_df.rename(columns = {'image_no':'image_name'}, inplace = True)
    prepro_df['strabismus'] = prepro_df['image_name'].str.len()>10
    prepro_df.to_pickle(file_name)
    return prepro_df    
    
def pivot_df(df):
    old_df = df.copy() 
    columns_ = [ 'image_name',
                 'MR-1','MR-2','MR-3','MR-4','MR-5','MR-6','MR-7','MR-8','MR-9',
                 'ML-1','ML-2','ML-3','ML-4','ML-5','ML-6','ML-7','ML-8','ML-9',
                 'LR-1','LR-2','LR-3','LR-4','LR-5','LR-6','LR-7', 'LR-8','LR-9',
                 'LL-1', 'LL-2','LL-3','LL-4','LL-5','LL-6','LL-7','LL-8','LL-9',
                 'RR-1','RR-2','RR-3','RR-4','RR-5','RR-6','RR-7','RR-8','RR-9'
                 'RL-1','RL-2','RL-3','RL-4','RL-5','RL-6','RL-7','RL-8','RL-9',]
    update_df = pd.DataFrame(None, columns = columns_)
    if '-' in old_df['image_name'][0]:
        image_name = old_df['image_name'][0].split('-')[0]
    else:
        image_name = old_df['image_name'][0][:-2]    
    update_df = pd.concat([update_df,pd.DataFrame([{'image_name':image_name}])], ignore_index=True)
    for i,row in old_df.iterrows():
        eye_cls = row['image_name'][-2:]+'-'+str(row['class']+1)
        mid_x = (row['xmax']+row['xmin'])/2
        mid_y = (row['ymax']+row['ymin'])/2
        conf = row['confidence']
        cls_name = row['cls_name']
        
        # pivot data into given dataframe
        update_df[eye_cls] = mid_x
    return update_df


def add_ratio(df):
    df = df.copy()
    gazes = ['M','L','R']
    widthlm = [
               ['R-1','R-4'], #widthR
               ['L-1','L-4'], #widthL
              ]
    
    eyeslm = [
        
                 ['R_rt1','R-7','R-1'],
                 ['R_rt2','R-9','R-7'],
                 ['R_rt3','R-8','R-9'],
                 ['R_rt4','R-8','R-4'],
                 ['R_rt5','R-8','R-7'], #no center

                 ['L_rt1','L-7','L-1'],
                 ['L_rt2','L-9','L-7'],
                 ['L_rt3','L-8','L-9'],
                 ['L_rt4','L-8','L-4'],
                 ['L_rt5','L-8','L-7'], #no center

              ]
    for g in gazes:
        for widL, widR in widthlm:
            df[str('WID-'+g)] = df[str(g+widR)] - df[str(g+widL)]
    for g in gazes:
        for ratio, ls, rs in eyeslm:
            try:
                df[str(g+ratio)]= (df[str(g+ls)]-df[str(g+rs)])/df[str('WID-'+g)]
            except:
                pass   
            
    df = df[df.columns.drop(list(df.filter(regex='-')))] #drops unused coumns
    return df

def predict_strabismus(df,drop=False):
    print("Predict starbismus")
    df = df.copy()
    df = df.reset_index(drop=True)
    X = df.drop(columns=['image_name']).copy()
    
    print(X)
    # mean = float(X.mean(axis=1))
    # X = X.fillna(mean)
    # X = np.array(X, dtype="float32")
    y = None

    print("Loading model")
    if drop == False:
        model = pickle.load(open(setting.PREDICT_MODEL_PATH, 'rb'))
    else:
        model = pickle.load(open(setting.PREDICT_DROP_MODEL_PATH, 'rb'))
    # model = pickle.load(open(setting.PREDICT_MODEL_PATH, 'rb'))
    # model_drop = pickle.load(open(setting.PREDICT_DROP_MODEL_PATH, 'rb'))


    print("Fitting Model")
    prediction = model.predict(X) #X,y decisiontree
    predict_prob = model.predict_proba(X)

    # print("Fitting Drop Model")
    # prediction = model_drop.predict(X) #X,y decisiontree
    # predict_prob = model_drop.predict_proba(X)

    if prediction == True:
        print('Have risk of Strabismus')
    else:
        print('No risk of Strabismus')
    return [bool(prediction), predict_prob]