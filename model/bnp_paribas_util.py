
import pandas as pd
from os.path import join, exists, getsize, basename, dirname
from pathlib import Path

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

from model.pr_auc_score_SB05ixL import complete_y_cols

DATASET_KEYS = ["X_train", "X_test", "y_train", "y_test", "X_test_challenge"]

SOURCES_FILE_NAME = {
    'official_X_train' : "X_train_G3tdtEn.csv",
    'official_y_train' : "Y_train_2_XPXJDyy.csv",
    'official_X_test_challenge' : "X_test_8skS2ey.csv",
}

DATASET_FILES_NAMES = {
    "dataset_preprocess"                : "dataset_preprocess.csv",
    "dataset_transpose"                 : "dataset_transpose.csv",
    "dataset_transpose_split"           : "dataset_transpose_split.csv",
    "dataset_encoded"                   : "dataset_encoded.csv",
    "dataset_encoded_rounded"           : "dataset_encoded_rounded.csv",
    "dataset_test_preprocess"           : "dataset_test_challenge_preprocess.csv",
    "dataset_test_transpose"            : "dataset_test_challenge_transpose.csv",
    "dataset_test_encoded"              : "dataset_test_challenge_encoded.csv",
    "dataset_test_encoded_rounded"      : "dataset_test_challenge_encoded_rounded.csv",
    "dataset_test_train_compatible"     : "dataset_test_challenge_train_compatible.csv",
}

NUMEROTED_COL_PREFIX = ['item', "model", 'goods_code', 'cash_price', 'make',  'Nbr_of_prod_purchas']
                      
# ----------------------------------------------------------------------------------
#                        DATA
# ----------------------------------------------------------------------------------
def col_names_without_numeroted_card_col(df, verbose=0):
    short_name = "col_names_without_numeroted_card_col"
    col_extract = []
    # On enlève toutes les colonnes liées à la place dans le panier, ce n'est pas ce qui nous intéresse
    for col_name in list(df.columns):
        if not col_name.startswith('item') and not col_name.startswith('model') and not col_name.startswith('goods_code') and not col_name.startswith('cash_price') and not col_name.startswith('make') and not col_name.startswith('Nbr_of_prod_purchas'):
            col_extract.append(col_name)
    if verbose>0:
        info(short_name, f"{col_extract}")
    return col_extract

def prefixed_cols(df, col_prefix, verbose=0):
    """Return all the column names starting with the Prefix `col_prefix`

    Args:
        df (_type_): _description_
        col_prefix (_type_): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        list(str): _description_
    """
    short_name = "prefixed_cols"
    cols = []
    for col in df.columns:
        if col.startswith(col_prefix):
            cols.append(col)
    
    if verbose > 0: print(f"[{short_name}]\tINFO: {len(cols)} columns prefixed by {col_prefix}")        
    return cols

def reduce_data_by_typing(df, verbose=0):
    """ Round float data to x.xx and convert float64 nb to int8 and float data float16

    Args:
        df (DataFrame): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    short_name = "reduce_data_by_typing"
    if verbose>0:info(short_name, f"{df.dtypes}")
    for col in df.columns:
        if df[col].dtype == 'float64':
            if col.endswith('_nb') or col in ['Nb_of_items']:
                df[col] = df[col].round(decimals=0)
                df[col] = df[col].astype("int8")
            else:
                df[col] = df[col].round(decimals=2)
                df[col] = df[col].astype("float16")
        elif df[col].dtype == 'int64' and col not in ['index', 'ID']:
            df[col] = df[col].astype("int8")
    if verbose>0:info(short_name, f"{df.dtypes}")
    return df

def add_amounts(x_df_input, verbose=0):
    """Add the amount column, sum of all item cas column

    Args:
        x_df_input (DataFrame): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: the updated df
    """
    short_name="add_amounts"
    ncol_item = "amount"
    
    x_df = x_df_input
    cols = sorted(list(x_df.columns))
    start = False
    if ncol_item not in cols:
        x_df[ncol_item] = 0
                
        for col in cols:
            if col.lower().startswith("item_"):
                start = True
                if col.lower().endswith("_cash"):
                    x_df[col] = x_df[col].fillna(0)
                    x_df[ncol_item] = x_df[ncol_item] + x_df[col]
            # On sort de la boucle dès que le nom de colonne ne commence pas par item
            elif start:
                break
    elif verbose>0:
        info(short_name, f"The column {ncol_item} ever in df.")
    return x_df


# ----------------------------------------------------------------------------------
#                        OVER DATAS FUNCTION
# ----------------------------------------------------------------------------------

start_ = "dataset_split_train_OVER_"

OVERED_DATASET_FILE_NAME = {
    "BorderlineSMOTE"            : start_+"BorderlineSMOTE.csv",
    "RandomOverSampler"          : start_+"RandomOverSampler.csv",
    "RandomOverSampler amount less" : start_+"RandomOverSampler1.csv",
    "SMOTE_minority"             : start_+"SMOTE_minority.csv",
    "SMOTE_minority amount less" : start_+"SMOTE_minority1.csv",
    "SVMSMOTE"                   : start_+"SVMSMOTE.csv",
}

def load_over_dataset(path, verbose=0):
    short_name = "load_over_dataset"
    dataset_over = None
    # Chargement des données
    save_path = path.replace(".csv", "_rounded.csv")
    if not exists(save_path):

        dataset_over = load_dump_file(data_set_path=path, verbose=verbose)
        # if verbose>0: info(short_name, f"Reduce datas {surligne_text('IN PROGRESS')}")
        # dataset_over = reduce_data_by_typing(df=dataset_over, verbose=verbose)
        try:
            to_drop = "Unnamed: 0"
            if to_drop in list(dataset_over.columns):
                if verbose>0: info(short_name, f"{to_drop} column to drop.")
                dataset_over = dataset_over.drop(labels=[to_drop], axis=1)
            else:
                if verbose>1: debug(short_name, f"{to_drop} not in dataset.")
        except:
            pass
        dump_file(data_set_path=save_path, df=dataset_over, verbose=verbose)

        if verbose>0: info(short_name, f"{dataset_over.shape} datas keep {surligne_text('SAVE')}")
    else:
        dataset_over = load_dump_file(data_set_path=save_path, verbose=verbose)
        
    if verbose>0: info(short_name, f"{dataset_over.shape} datas {surligne_text('LOAD')}")
    return dataset_over

# ----------------------------------------------------------------------------------
#                        BACKUP
# ----------------------------------------------------------------------------------
def load_dump_file(data_set_path, file_name=None, force=0, short_name = "load_dump_file", verbose=0):   
    save_file_path = data_set_path

    if save_file_path is not None:
        if file_name is not None and len(file_name)>0:
            save_file_path = join(save_file_path, file_name)
        else:
            file_name = basename(save_file_path)
        
    df_res = None
    if exists(save_file_path) and getsize(save_file_path) > 0 and not force:
        # Chargement de la DF fichier
        if verbose > 1: debug(short_name, f"{file_name} exist")
        try:
            df_res = pd.read_csv(save_file_path, sep=",", index_col="index",low_memory=False)
        except:
            df_res = pd.read_csv(save_file_path, sep=",", low_memory=False)
        if verbose > 0: info(short_name, f"{file_name} {surligne_text('LOAD')}")
        try:
            df_res = df_dense_to_sparse(df_res, verbose=verbose)
            if verbose > 0: info(short_name, f"{file_name} {surligne_text('COMPRESS')}")
        except Exception as err:
            if verbose > 0: warn(short_name, f"{file_name} : Impossible to compress data : {err}")
    return df_res


def dump_file(data_set_path,  df,file_name=None, short_name = "dump_file", verbose=0):   
    save_file_path = data_set_path

    if save_file_path is not None:
        if file_name is not None and len(file_name)>0:
            save_file_path = join(save_file_path, file_name)
        else:
            file_name = basename(save_file_path)
        
        create_parent_dir(file_path=save_file_path, verbose=verbose-1)
        # Sauvegarde du fichier
        try:
            df = df_sparse_to_dense(df=df, verbose=verbose)
        except Exception as err:
            if verbose > 0: warn(short_name, f"{file_name} : Impossible to uncompress data : {err}")
        df.to_csv(save_file_path, index_label='index')

        if verbose > 0: info(short_name, f"{file_name} => {surligne_text('SAVED')}")
    elif verbose>0:
        warn(short_name, 'No path to dump')
    return save_file_path


from sklearn.model_selection import train_test_split
from os.path import exists

def load_only_splited_data(dataset_path, over_name=None, target = 'fraud_flag', add_amount=False, verbose=0):
    short_name = "load_only_splited_data"
    dataset_dict = {}
    dataset_path_split = dataset_path

    if 'split' not in dataset_path_split:
        dataset_path_split = dataset_path_split.replace("encoded_", "")
        dataset_path_split = dataset_path_split.replace(".csv", f"_split.csv")

    for set_name in DATASET_KEYS[:-1]:
        if set_name not in list(dataset_dict.keys()):
            if over_name is not None and len(over_name)>0 and set_name == "X_train":
                if verbose > 0: info(short_name, f"over {over_name} Loading {set_name}...")
                path = dirname(dataset_path_split)
                over_path = join(path,OVERED_DATASET_FILE_NAME.get(over_name))
                dataset_over = load_over_dataset(path=over_path, verbose=verbose)
                dataset_dict["y_train"] = dataset_over[['ID',target]]
                dataset_dict[set_name] = dataset_over.drop([target], axis=1)
                continue
            
            path = dataset_path_split.replace(".csv", f"_{set_name}.csv")

            if dataset_dict.get(set_name, None) is None and exists(path):
                if verbose > 0: info(short_name, f"set {set_name} Loading {set_name}...")
                dataset_dict[set_name] = load_dump_file(data_set_path=path, file_name=None, verbose=verbose)
                if add_amount and set_name.startswith("X_"):
                    dataset_dict[set_name] = add_amounts(dataset_dict[set_name], verbose=verbose)
    return dataset_dict

def split_data_or_load_it(dataset_path, over_name=None, test_size = 0.2, random_state = 42, save_it=True, force=False, target = 'fraud_flag', add_amount=False, cart_id_attention=True, verbose=0):
    short_name="split_data_or_load_it"
    if verbose > 0: info(short_name, surligne_text('IN PROGRESS...'))

    dataset_path_split = dataset_path

    if 'split' not in dataset_path_split:
        dataset_path_split = dataset_path_split.replace("full_", "")
        dataset_path_split = dataset_path_split.replace("_encoded", "")
        dataset_path_split = dataset_path_split.replace(".csv", f"_split.csv")

    dataset_dict = {}
    if not force:
        dataset_dict =load_only_splited_data(dataset_path=dataset_path_split, over_name=over_name, target = target, add_amount=add_amount, verbose=verbose)

    # si les fichiers split n'existent pas :
    if force or len(dataset_dict) < 4:       
        
        if verbose > 1: debug(short_name, f"Chargement des données train sources...")
        train_origin = load_dump_file(data_set_path=dataset_path, file_name=None, verbose=verbose)
        dataset_dict['train_origin'] = train_origin
        if verbose > 0: info(short_name,f"Données d'origine {train_origin.shape} {surligne_text('LOAD')}")

        if verbose > 1: debug(short_name, f"Split du dataset_path...")
        dt_train, dt_test = None, None
        if cart_id_attention:
            # Ici nous sommes vigilents sur la cohérence des données, à savoir un même ID ne doit être que dans l'un des splits même s'il a plusieurs articles
            dt_id_train, dt_id_dt_test = train_test_split(train_origin['cart_ID'].unique(), test_size=test_size, random_state=random_state)
            dt_train = train_origin[train_origin['cart_ID'].isin(dt_id_train)]
            dt_test = train_origin[train_origin['cart_ID'].isin(dt_id_dt_test)]
        else:
            dt_train, dt_test = train_test_split(train_origin, test_size=test_size, random_state=random_state)
        if verbose>0: info(short_name, f"Split {dt_train.shape} / {dt_test.shape} -- Train / Test {surligne_text('DONE')}")
        
        if verbose > 1: debug(short_name, f"Séparation de X et y...")
        
        columns = list(train_origin.columns)
        if verbose > 2: debug(short_name, len(columns))
        columns.remove(target)
        if verbose > 2: debug(short_name, len(columns))
        X_train, y_train = dt_train[columns], dt_train[target]
        X_test, y_test   = dt_test[columns],  dt_test[target]
        if verbose > 1: debug(short_name, f"Y extrated")

        if add_amount:
            if verbose > 1: debug(short_name, f"Add amounts...")
            # Ajout des données manquantes
            X_train = add_amounts(X_train, verbose=verbose)
            X_test  = add_amounts(X_test, verbose=verbose)
        
        if verbose > 1: debug(short_name, f"Complete y columns...")
        y_train = complete_y_cols(X_test=X_train, y_param=y_train) 
        y_test  = complete_y_cols(X_test=X_test, y_param=y_test) 

        for dataset_name, dataset in zip(DATASET_KEYS[:-1], [X_train, X_test, y_train, y_test]):
            # dataset = reduce_data_by_typing(df=dataset, verbose=verbose)
            dataset_dict[dataset_name] = dataset
            if save_it and dataset is not None:
                dump_file(data_set_path=dataset_path_split.replace(".csv", f"_{dataset_name}.csv"), df=dataset, file_name=None, verbose=verbose)
                    
    if verbose > 0:
        info(short_name, f"{len(dataset_dict)} dataset {surligne_text('LOAD')}")
        if len(dataset_dict)>0: info(short_name, f"{dataset_dict.get(DATASET_KEYS[0]).shape} / {dataset_dict.get(DATASET_KEYS[1]).shape}")
    return dataset_dict


def load_splited_data(dataset_path, over_name=None, test_size = 0.2, random_state = 42, save_it=True, force=False, target = 'fraud_flag', add_amount=True, verbose=0):
    
    return split_data_or_load_it(dataset_path=dataset_path, over_name=over_name, 
                                test_size = test_size, random_state = random_state, 
                                save_it=save_it, force=force, target = target, add_amount=add_amount, 
                                cart_id_attention=False, verbose=0)


def catalog(save_file_path, df, verbose=0):
    """
    Load data if file exist, else extract data from df and save it.

    Args:
        save_file_path (_type_): _description_
        df (_type_): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    short_name = "catalog"
    catalog_df = None
    if save_file_path is not None and exists(save_file_path):
        catalog_df = load_dump_file(data_set_path=save_file_path, verbose=verbose-1)
        if verbose > 0: info(short_name, f"{catalog_df.shape} data from {basename(save_file_path)} {surligne_text('LOAD')}")
    elif df is not None:
        catalog_df = df[['code_produit', 'model', 'categorie', 'make', 'gamme', 'cat_embed', 'PU', 'ref_produit']]
        if verbose > 1: debug(short_name, f"{catalog_df.shape}")
        catalog_df = catalog_df.drop_duplicates()
        if verbose > 1: debug(short_name, f"{catalog_df.shape} after drop duplicated")
        catalog_df = catalog_df.sort_values('code_produit')
        catalog_df = catalog_df.reset_index(drop=True)
        dump_file(data_set_path=save_file_path, df=catalog_df, verbose=verbose-1)
        if verbose > 0: info(short_name, f"{catalog_df.shape} data in {basename(save_file_path)} {surligne_text('SAVE')}")
    return catalog_df
# ----------------------------------------------------------------------------------
#                        DF
# ----------------------------------------------------------------------------------
def diplay_target_repartition(df, target='fraud_flag', key='WARN'):
    short_name = "target_repartition"
    rate = round((df[df[target]==1].shape[0]/df.shape[0])*100, 2)
    display(short_name, f"Il y a {rate} % de fraude dans le train", key='INFO',color= COLOR_MOD.get(key, None))
    info(short_name, "The Imbalance ratio = Nb d’observations majoritaires / Nb d’observations minoritaires")
    display(short_name, f"The Imbalance ratio = {df[df[target]==0].shape[0]} / {df[df[target]==1].shape[0]}", key='INFO',color= COLOR_MOD.get(key, None))
    display(short_name, f"The Imbalance ratio = {df[df[target]==0].shape[0] / df[df[target]==1].shape[0]}", key='INFO',color= COLOR_MOD.get(key, None))


def get_numeric_columns_names(df, verbose=False):
    """Retourne les noms des colonnes numériques
    Args:
        df (DataFrame): Données
        verbose (bool, optional): Mode debug. Defaults to False.

    Returns:
        List(String): liste des noms de colonne
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    newdf = df.select_dtypes(include=numerics)
    return list(newdf.columns)

def df_memory(df, verbose=0):
    """Return the memory size

    Args:
        df (Dataframe): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        float: _description_
    """
    df_memory = df.memory_usage().sum() / 1e3
    if verbose>0:
        info('df_memory', f'{(df_memory):0,.2f} Ko (bytes) \t--> {(df_memory/1e3):0,.2f} Mo\t--> {(df_memory/1e6):0,.2f} Go')
    return df_memory

def df_dense_to_sparse(df, fill_value=0, verbose=0):
    """Convert a Dense DataFrame to Sparse DataFrame to reduce memory space.

    Args:
        df (DataFrame): DataFrame to convert
        fill_value (int, optional): The value to fill. Defaults to 0.
        verbose (int, optional): Log level. Defaults to 0.

    Returns:
        DataFrame: the updated DataFrame
    """
    short_name = "df_dense_to_sparse"
    m_before = df_memory(df=df, verbose=verbose-1)
    for col in df.columns:
        col_type = df[col].dtype
        dtype = pd.SparseDtype(col_type, fill_value=fill_value)
        df[col] = df[col].astype(dtype)

    if verbose>0:
        m_after = df_memory(df=df, verbose=verbose)
        info(short_name, f'{m_before:0,.2f} bytes before \t--> {m_after:0,.2f} Ko after --> \t {m_after-m_before} Ko')
        info(short_name, f'Sparse.density : {df.sparse.density:0,.2f}')
    return df

def df_sparse_to_dense(df, verbose=0):
    short_name = "df_sparse_to_dense"
    m_before = df_memory(df=df, verbose=verbose-1)
    res = df.sparse.to_dense()
    if verbose>0:
        m_after = df_memory(df=df, verbose=verbose)
        info(short_name, f'{m_before:0,.2f} Ko before \t--> {m_after:0,.2f} Ko after --> \t {m_after-m_before} Ko')
    return res

# ----------------------------------------------------------------------------------
#                        FILES
# ----------------------------------------------------------------------------------
def create_parent_dir(file_path, verbose=0):
    """Création du répertoire parent s'il n'existe pas

    Args:
        file_path (_type_): _description_
        verbose (int, optional): _description_. Defaults to 0.
    """
    file_name = basename(file_path)
    parent_dir = file_path[:-(len(file_name))]
    if parent_dir.endswith('\\') or  parent_dir.endswith('/'):
        parent_dir = parent_dir[:-1]
    creat_dir(dest_path=parent_dir, verbose=verbose)
        

def creat_dir(dest_path, verbose=0):
    # Création du répertoire s'il n'existe pas
    if dest_path is None or len(dest_path.strip()) > 0:   
        base = Path(dest_path)
        base.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------------
#                        PRINT
# ----------------------------------------------------------------------------------
from termcolor import colored

COLORS = {
    "default" : (6,42,30),
    "green" : (6,42,30),
    "blue" : (13,36,81),
    "red" : (74,40,34),
    "yellow" : (87,82,9),
}

COLOR_MOD = {
    'DEBUG' : 'blue',
    'WARN'  : 'yellow',
    'ERROR' : 'red',
}

def display(function_name, text, now=None, duration=None, key='INFO', color=None, log_end=""):
    log = f"[{function_name:<20}]\t{key.upper():<5} {text}"
    
    log = f"{log} {log_end:>25}"
    
    if now is not None and duration is not None:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        end_ = f"\t\t END {now} ---> in {duration}"
        log = log + f"{end_:>60}"
    elif now is not None:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        end_ = f"\t\t START {now}"
        log = log + f"{end_:>38}" 
            
    if color is not None:
        log = colored(log, color)
        
    print(log)
    return log

# %% colored
def info(function_name, text, now=None, duration=None, log_end=""):
    key='INFO'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def debug(function_name, text, now=None, duration=None, log_end=""):
    key='DEBUG'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def warn(function_name, text, now=None, duration=None, log_end=""):
    key='WARN'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)
    
def error(function_name, text, now=None, duration=None, log_end=""):
    key='ERROR'
    return display(function_name=function_name, text=text, now=now, duration=duration, key=key, color= COLOR_MOD.get(key, None), log_end=log_end)

def surligne_text(text, color="green"):
    r = COLORS.get(color,COLORS["default"])[0] 
    g = COLORS.get(color,COLORS["default"])[1] 
    b = COLORS.get(color,COLORS["default"])[2] 
    return f'\x1b[{r};{b};{g}m{text}\x1b[0m'

