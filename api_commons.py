import pandas as pd
from os import getcwd, path
from model.bnp_paribas_preprocessing import pre_processing
from model.bnp_paribas_Model import load_model

def do_prediction(cart, verbose=0):
    predict = None
    if cart is not None:
        df_input = cart.to_df()

        df_to_predict = pre_processing(X=df_input, y=None, save_file_path=None, verbose=verbose)
        
        model_path = path.join(_get_execution_path(), "model", "2023-03-12-LGBMClassifier_best.pickle")
        print(f"Model path : {model_path}")

        # Chargement du model
        model = load_model(model_path)
        predict_price = model.predict(df_to_predict)

        # Attention, c'est n numpy.ndarray qui est retourné
        print(f"Prédiction : {predict_price}")
        predict = predict_price[0]
    return predict


def get_img_path():
    # Récupère le répertoire du programme
    
    model_path = _get_execution_path()    
    model_path = path.join(model_path, 'static')
    
    print(f"IMG path : {model_path}")
    return model_path

def get_python_version():
    from sys import version_info
    PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major,
                                              minor=version_info.minor,
                                              micro=version_info.micro)
    return PYTHON_VERSION

def _get_execution_path():

    # Récupère le répertoire du programme
    execution_path = getcwd()
    project_name = "projet_bnp_paribas_fraudeurs_app"
    # Permet de gérer le cas d'une exécution dans un notebook par exemple
    execution_path = execution_path.split("PROJETS")[0]
    execution_path = path.join(execution_path, "PROJETS", project_name)
    import sys
    sys.path.append(execution_path)

    return execution_path