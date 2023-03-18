
import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
from datetime import datetime

import sys
from os import getcwd
from os.path import join
# Définition des chemins
project_name = "projet_bnp_paribas_fraudeurs_app"
execution_path = getcwd() + r'/'
print(execution_path)
execution_path = execution_path.split(project_name)[0]
execution_path = join(execution_path, project_name)
sys.path.append(execution_path)

from model.pr_auc_score_SB05ixL import complete_y_cols, evaluate
from model.bnp_paribas_util import *

# ----------------------------------------------------------------------------------
#                        MODEL FUNCTION
# ----------------------------------------------------------------------------------
from sklearn.metrics import average_precision_score

def train_model(model, model_name, 
                dataset_dict, data_set_path, 
                scores,score_path, 
                params, features="ALL", add_data=np.nan,commentaire=np.nan,target="fraud_flag", is_transpose=False, 
                verbose=0):
    short_name = "train_model"
    if verbose>1:debug(short_name, f"{model_name} model fiting...")
    model.fit(dataset_dict.get("X_train"), dataset_dict.get("y_train")[target])

    if verbose>1:debug(short_name,f"Model evaluation...")
    score_accuracy = model.score(dataset_dict.get("X_test"), dataset_dict.get("y_test")[target])
    y_pred = model.predict(dataset_dict.get("X_test"))
    pr_auc_score_, _, _ = evaluate(X_test=dataset_dict.get("X_test"), y_test=dataset_dict.get("y_test"), y_pred=y_pred, verbose=verbose)
    av_sc = None
    try:
        result = pd.DataFrame(model.predict_proba(dataset_dict.get("X_test")))
        y_pred_test = result[1]
        av_sc = average_precision_score(dataset_dict.get("y_test")[target],y_pred_test)
    except:
        pass
    if verbose>0:info(short_name,f"accuracy score : {score_accuracy}, pr_auc score : {pr_auc_score_}, average_precision_score : {av_sc}")
    try:
        _ = predict_official_testset(model=model, test_origin=dataset_dict.get("test_origin", dataset_dict.get("X_test_challenge", None)), data_set_path=data_set_path, model_name=model_name, is_transpose=is_transpose, verbose=verbose)
    except:
        pass

    if verbose>1:debug(short_name,f"Add score...")
    if params is None and isinstance(model, GridSearchCV):
        params = str(model.best_params_)
    n_scores = add_score(scores_param=scores, modele=model_name, features=features, add_data=add_data, 
            params=params, 
            accuracy_score=score_accuracy, pr_auc_score_TEST_perso=pr_auc_score_, avg_score=av_sc,
            commentaire=commentaire,
            score_path=score_path.replace(".csv", f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"),
            verbose=verbose)

    return model, n_scores


def load_test_df(file_path, train_columns, force=False, verbose=0):
    """Load the official test df and add : amount and reducre data by typing, then save the new df.
    If the df have been save, just load it.

    Args:
        file_path (str): _description_
        train_columns (list(str)): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "load_test_df"
    test_origin = None
    save_path = file_path.replace(".csv", "_rounded.csv")
    if not exists(save_path) or force:
        test_origin = pd.read_csv(file_path, sep=',',index_col="index" ,low_memory=False)
        # Ajout des colonnes manquantes
        test_origin = add_amounts(test_origin, verbose=verbose)

        test_cols = test_origin.columns
        for col in train_columns:
            if col not in test_cols:
                test_origin[col] = 0
        
        test_origin = test_origin[train_columns]
        test_origin = reduce_data_by_typing(df=test_origin, verbose=verbose)
        test_origin.to_csv(save_path, index_label="index")
        if verbose>0:
            info(short_name, f"{test_origin.shape} test données mises à jour")
    else:
        test_origin = pd.read_csv(save_path, sep=',',index_col="index" ,low_memory=False)

    if verbose>0:
        info(short_name, f"{test_origin.shape} test données chargées")
    return test_origin

def predict_official_testset(model, test_origin, data_set_path, model_name="", is_transpose=False, verbose=0):
    short_name = "predict_official_testset"
    y_pred_test_complete = None
    if test_origin is not None:
        test_origin = test_origin[get_numeric_columns_names(test_origin)]
        if verbose>0:info(short_name,f"Prediction for test origine...")
        try:
            result = pd.DataFrame(model.predict_proba(test_origin))
            y_pred_test = result[1]
            # print(average_precision_score(y_test,y_pred_test))
        except:
            y_pred_test = model.predict(test_origin)
        
        if y_pred_test is not None:

            y_pred_test_complete = complete_y_cols(X_test=test_origin, y_param=y_pred_test)
            if is_transpose:
                group_tt = y_pred_test_complete.groupby(['ID'], as_index=False).agg({'fraud_flag':['mean']})
                group_tt.columns = ['_'.join(col) for col in group_tt.columns]
                y_pred_test_complete = group_tt.rename(columns={'fraud_flag_mean': 'fraud_flag', "ID_": "ID"})
                y_pred_test_complete['fraud_flag'] = y_pred_test_complete['fraud_flag'].round(decimals=4)
            res_path = join(data_set_path, 'official_test_predictions', model_name+"_"+datetime.now().strftime('%Y-%m-%d-%H_%M')+".csv")
            y_pred_test_complete.to_csv(res_path)
    elif verbose>0:info(short_name,f"No prediction request for test orgine")
    return y_pred_test_complete


from sklearn.linear_model import LogisticRegression
def train_LogisticRegression(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            penalty="l2", fit_intercept=True,solver='liblinear',is_transpose=False, 
                            verbose=0):
    model_name="LogisticRegression"
    if verbose>1:debug(model_name, f"Model creation...")
    my_fist_model = LogisticRegression(penalty=penalty, fit_intercept=fit_intercept,solver=solver)
    params=f"penalty='{penalty}', fit_intercept={fit_intercept},solver='{solver}'"

    model, n_scores = train_model(model=my_fist_model, model_name=model_name, 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire,is_transpose=is_transpose, verbose=verbose)
    
    return model, n_scores

import lightgbm as lgb
def train_LGBMClassifier(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            boosting_type='goss', max_depth=5, learning_rate=0.1, n_estimators=1000, 
                            subsample=0.8,colsample_bytree=0.6,is_transpose=False, 
                            verbose=0):
    
    model_name="LGBMClassifier"
    if verbose>1:debug(model_name, f"Model creation...")
    lgb_classifier = lgb.LGBMClassifier(boosting_type=boosting_type,  
                                    max_depth=max_depth, 
                                    learning_rate=learning_rate,
                                    n_estimators=n_estimators, 
                                    subsample=subsample,  
                                    colsample_bytree=colsample_bytree,
                                   )
    params=f"boosting_type='{boosting_type}', max_depth={max_depth},learning_rate={learning_rate},n_estimators={n_estimators}.subsample={subsample},colsample_bytree={colsample_bytree}"
    model, n_scores = train_model(model=lgb_classifier, model_name=model_name, 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire, is_transpose=is_transpose,verbose=verbose)
    return model, n_scores

from sklearn.model_selection import GridSearchCV
def train_LGBMClassifier_GridSearch(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            grid_params=None,
                            boosting_type='goss', max_depth=5, learning_rate=0.1, n_estimators=1000, 
                            subsample=0.8,colsample_bytree=0.6,is_transpose=False, 
                            verbose=0):
    
    model_name="LGBMClassifier_GridSearch"

    if grid_params is None:
        grid_params = { 'boosting_type'      : list({'gbdt', 'dart', 'goss', boosting_type}),
                        'max_depth'          : list({max_depth}),
                        'learning_rate'      : list({learning_rate}),
                        'n_estimators'       : list({n_estimators}),
                        'subsample'          : list({subsample}),
                        'colsample_bytree'   : list({colsample_bytree})
                        }

    lgb_classifier = lgb.LGBMClassifier()
    grid = GridSearchCV(lgb_classifier,param_grid=grid_params, cv=4, verbose=verbose)
    
    if verbose>1:debug(model_name, f"Model creation...")
    # params=f"boosting_type='{boosting_type}', max_depth={max_depth},learning_rate={learning_rate},n_estimators={n_estimators}.subsample={subsample},colsample_bytree={colsample_bytree}"
    model, n_scores = train_model(model=grid, model_name=model_name, 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=None, features=features,
                                add_data=add_data,commentaire=commentaire, is_transpose=is_transpose,verbose=verbose)

    # Print the best parameters found
    if verbose>0:
        info(model_name, f"{grid.best_params_}")
        info(model_name, f"{grid.best_score_}")
        
    return model, n_scores


from sklearn.ensemble import RandomForestClassifier
def train_RandomForestClassifier(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            random_state=42,max_depth=2,n_estimators=1000, is_transpose=False, 
                            verbose=0):
    
    model_name="RandomForestClassifier"
    if verbose>1:debug(model_name, f"Model creation...")
    my_fist_model = RandomForestClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)
    params=f"random_state={random_state},max_depth={max_depth},n_estimators={n_estimators}"

    model, n_scores = train_model(model=my_fist_model, model_name=model_name, 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire, is_transpose=is_transpose,verbose=verbose)
    
    return model, n_scores

from sklearn.ensemble import AdaBoostClassifier
def train_AdaBoostClassifier(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            random_state=42,n_estimators=1000,is_transpose=False, 
                            verbose=0):
    
    model_name="AdaBoostClassifier"
    if verbose>1:debug(model_name, f"Model creation...")
    my_fist_model = AdaBoostClassifier(random_state=random_state, n_estimators=n_estimators)
    params=f"random_state={random_state},n_estimators={n_estimators}"

    model, n_scores = train_model(model=my_fist_model, model_name=model_name, 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire,is_transpose=is_transpose, verbose=verbose)
    
    return model, n_scores


from sklearn.ensemble import GradientBoostingClassifier
def train_GradientBoostingClassifier(dataset_dict, data_set_path, 
                            scores, score_path, 
                            features="ALL", add_data=np.nan,commentaire=np.nan, 
                            random_state=42,max_depth=2,n_estimators=1000,is_transpose=False, 
                            verbose=0):
    
    model_name="GradientBoostingClassifier"
    if verbose>1:debug(model_name, f"Model creation...")
    my_fist_model = GradientBoostingClassifier(max_depth=max_depth, random_state=random_state, n_estimators=n_estimators)
    params=f"random_state={random_state},max_depth={max_depth},n_estimators={n_estimators}"

    model, n_scores = train_model(model=my_fist_model, model_name=model_name, 
                                dataset_dict=dataset_dict, data_set_path=data_set_path, scores=scores, score_path=score_path, params=params, features=features,
                                add_data=add_data,commentaire=commentaire,is_transpose=is_transpose, verbose=verbose)
    
    return model, n_scores

import pickle

def save_model(model, model_name, model_path, train_file_name):
    # save model
    tp_file_name = train_file_name.replace(".csv", f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{model_name}.pickle")
    file_path = join(model_path,  tp_file_name)
    pickle.dump(model, open(file_path, "wb"))
    return file_path

def load_model(filename):
    # load model
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model


# ----------------------------------------------------------------------------------
#                        SCORE
# ----------------------------------------------------------------------------------
def save_score(scores, score_path):
    index_label ="date"
    if index_label in list(scores.columns):
        index_label ="index"
    scores.to_csv(score_path, sep='|', index_label =index_label)

def load_scores(score_path, save_it=False, verbose=0):
    scores = pd.read_csv(score_path, sep='|', index_col="date")
    # scores = scores[['date', 'Modèle', 'Features', 'Add Data', 'Accuracy Score',  'pr_auc_score TEST perso', 'pr_auc_score TEST officiel', 'Commentaire', 'Params']]
    for col in ['Accuracy Score',  'pr_auc_score TEST perso', 'pr_auc_score TEST officiel']:
        try:
            scores.loc[scores[col]=="", col] = np.nan
        except Exception as err:
            if verbose>1: print(err)
        try:
            scores[col] = scores[col].fillna(0)
        except Exception as err:
            if verbose>1: print(err)
        try:
            scores[col] = scores[col].astype(float)
        except Exception as err:
            if verbose>1: print(err)
    if save_it:
        save_score(scores=scores, score_path=score_path)
    return scores

def add_score(scores_param, modele, features, add_data, params, accuracy_score=0, pr_auc_score_TEST_perso=0, avg_score=0,pr_auc_score_TEST_officiel=0, commentaire=np.nan, score_path=None,verbose=0):
    short_name = "add_score"
    data_dict = {
        'date' : [datetime.now().strftime('%Y-%m-%d %H:%M')],
        'Modèle' : [modele],
        'Features' : [features], 
        'Add Data' : [add_data],
        'Accuracy Score':[accuracy_score],
        'pr_auc_score TEST perso':[pr_auc_score_TEST_perso],
        'avg_score':[avg_score],
        'pr_auc_score TEST officiel':[pr_auc_score_TEST_officiel], 
        'Params':[params.replace('"', "'") if isinstance(params, str) else params],
        'Commentaire':[commentaire.replace('"', "'") if isinstance(commentaire, str) else commentaire],
    }
    if verbose>1:
        debug(short_name, data_dict)
    to_add = pd.DataFrame.from_dict(data_dict)
    scores = None
    if scores_param is None:
        scores = to_add
    else:
        scores = pd.concat([scores_param.reset_index(), to_add])
        scores = scores.set_index('date')
    if score_path is not None and len(score_path)>0:
        save_score(scores=scores, score_path=score_path)
    return scores



