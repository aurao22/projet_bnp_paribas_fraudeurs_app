"""
Module :
    1. pre_processing
    2. encode_data
    3. Dispatch UNKNOWN ?


------
Result :
    Final.1 dump
    Final.2 load
    Final.3 reduce_data_by_typing if need
"""

from datetime import datetime
import numpy as np
import pandas as pd
from os.path import join, dirname
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

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

from model.bnp_paribas_util import *

UNKNOWN_KEY = "UNKNOWN"

SUPER_CATEGORIES = {
        "COMPUTER"              :{'APPLE S','2HP ELITEBOOK 850V6','COMPUTERS','HP ELITEBOOK 850V6',},
        "COMPUTER ACCESSORIES"  :{'TARGUS GEOLITE ESSENTIAL CASE','COMPUTER PERIPHERALSACCESSORIES','BLANK MEDIAMEDIA STORAGE', 'PRINTERSSCANNERS','TOSHIBA PORTABLE HARD DRIVE', 'PRINTERS & SCANNERS', 'PRINTERS SCANNERS','LOGITECH PEBBLE M350 BLUETOOTH MOUSE','BLANK MEDIA & MEDIA STORAGE', 'BLANK MEDIA MEDIA STORAGE', 'COMPUTER NETWORKING', 'COMPUTER PERIPHERALS & ACCESSORIES', 'COMPUTER PERIPHERALS ACCESSORIES','APPLE PRODUCTDESCRIPTION','2TARGUS GEOLITE ESSENTIAL CASE','AERIALS REMOTE CONTROLS','6  SPACE GREY 32GB','2TOSHIBA PORTABLE HARD DRIVE','2LOGITECH PEBBLE M350 BLUETOOTH MOUSE'},
        "SOFTWARE"              :{'MICROSOFT OFFICE HOME AND STUDENT 2019,', 'MICROSOFT OFFICE HOME AND STUDENT 2019','2MICROSOFT OFFICE HOME AND STUDENT 2019,','COMPUTER SOFTWARE',},
        'AUDIO ACCESSORIES'     :{'AUDIO ACCESSORIES'},
        'BABY'                  :{'BABY CHANGING','BABY & CHILD TRAVEL', 'BABYCHILD TRAVEL','BABY CHILD TRAVEL','BABY FEEDING','BABY PLAY EQUIPMENT','BABYWEAR',},
        'BAGS'                  :{'LUGGAGE', 'BAGS WALLETSACCESSORIES', 'BAGSCARRY CASES','BAGS & CARRY CASES', 'BAGS CARRY CASES', 'BAGS WALLETS ACCESSORIES', 'BAGS, WALLETS & ACCESSORIES',},
        'OUTDOOR ACCESSORIES'   :{'OUTDOOR ACCESSORIES','OUTDOOR FURNITURE','GARDENING EQUIPMENT', 'BARBECUESACCESSORIES','BARBECUES & ACCESSORIES', 'BARBECUES ACCESSORIES',},
        'BEAUTY AND SAFETY'     :{'SUNCARE','BATHBODYCARE', 'HEALTHBEAUTY ELECTRICAL', 'HEATINGCOOLING APPLIANCES', 'SUNGLASSES & READING GLASSES', 'SUNGLASSES READING GLASSES', 'SUNGLASSESREADING GLASSES','MAKEUP','HAIRCARE', 'HEALTH & BEAUTY ELECTRICAL', 'HEALTH BEAUTY ACCESSORIES', 'HEALTH BEAUTY ELECTRICAL','FRAGRANCE','BATH & BODYCARE', 'BATH BODYCARE', 'FACIAL SKINCARE',},
        'HOUSE LINEN'           :{'TABLE LINEN','SOFT FURNISHINGS', 'CHILDRENS FURNITURE','LINGERIE & HOISERY', 'LINGERIE HOISERY', 'LIVINGDINING FURNITURE','LINGERIEHOISERY', 'LAUNDRYCLOTHESCARE','LAUNDRY & CLOTHESCARE', 'LAUNDRY CLOTHESCARE','BATH LINEN','BED LINEN','CARPETS RUGS FLOORING', 'CARPETS, RUGS & FLOORING','CARPETS RUGSFLOORING'},
        'HOUSE ACCESSORIES'     :{'TABLEWARE', 'WINDOW DRESSING','KITCHEN SCALESMEASURES','KITCHEN UTENSILSGADGETS', 'STANDSBRACKETS','STANDS & BRACKETS','PRESERVINGBAKING EQUIPMENT', 'STANDS BRACKETS','PRESERVING & BAKING EQUIPMENT', 'PRESERVING BAKING EQUIPMENT','LIVING & DINING FURNITURE', 'LIVING DINING FURNITURE','LIGHTING','BARWARE','KITCHEN ACCESSORIES', 'KITCHEN SCALES & MEASURES', 'KITCHEN SCALES MEASURES', 'KITCHEN STORAGE', 'KITCHEN UTENSILS & GADGETS', 'KITCHEN UTENSILS GADGETS','HOUSEHOLD CLEANING','HOME AND PERSONAL SECURITY', 'HOME OFFICE', 'HOME SAFETY EQUIPMENT','HEATING & COOLING APPLIANCES', 'HEATING COOLING APPLIANCES','FOOD STORAGE','FITTED KITCHENS','BATHROOM', 'BATHROOM ACCESSORIES', 'BATHROOM FIXTURES', 'BEDROOM FURNITURE','CHRISTMAS DECORATIONS','COOKING APPLIANCES', 'COOKWARE','DECORATING', 'DECORATIVE ACCESSORIES','DOOR FURNITURE','DISPOSABLE TABLEWARE CUTLERY','EASTER DECORATIONS',},
        'CHILDRENS'             :{'TOYS','SCHOOLWEAR', 'CHILDRENS ACCESSORIES','GIRLSWEAR','CHILDREN S ACCESSORIES', 'CHILDREN S FOOTWEAR', 'CHILDREN S FURNITURE', 'CHILDRENS FOOTWEAR','BOYSWEAR',},
        'NURSERY'               :{'NURSERY ACCESSORIES', 'NURSERY EQUIPMENT FURNITURE', 'NURSERY FURNITURE', 'NURSERY LINEN', 'NURSERY TOYS',},
        'WOMEN'                 :{'WOMEN S ACCESSORIES', 'WOMENS NIGHTWEAR', 'WOMEN S CLOTHES', 'WOMEN S FOOTWEAR', 'WOMEN S NIGHTWEAR', 'WOMENS ACCESSORIES', 'WOMENS CLOTHES', 'WOMENS FOOTWEAR'},
        'MEN'                   :{'MEN S ACCESSORIES','MENS ACCESSORIES', 'MENS FOOTWEAR','MENS SPORTSWEAR','MENS UNDERWEAR SOCKS','MENS UNDERWEARSOCKS', 'MEN S CLOTHES', 'MEN S FOOTWEAR', 'MEN S NIGHTWEAR', 'MEN S SPORTSWEAR', 'MEN S UNDERWEAR SOCKS', 'MENS CLOTHES', 'MENS NIGHTWEAR', 'MENS UNDERWEAR & SOCKS',},
        'CABLES ADAPTERS'       :{'CABLES & ADAPTERS', 'CABLES ADAPTERS', 'CABLESADAPTERS',},
        'BOOKS'                 :{'BOOKS',},
        'OFFICE ACCESSORIES'    :{'STATIONERY SUNDRIES','GREETING CARDSPERSONALISED STATIONERY','DIARIESORGANISERS', 'STORAGE & ORGANISATION', 'STORAGEORGANISATION', 'STORAGE ORGANISATION','PENS PENCILS','PAPER NOTEBOOKS','DIARIES & ORGANISERS', 'DIARIES ORGANISERS','FILING DESK ACCESSORIES'},
        'GAMES'                 :{'GAMES', 'GAMING',},
        'SPORT'                 :{'SPORTS EQUIPMENT','GYM EQUIPMENT',},
        'POWER ACCESSORIES'     :{'FULFILMENT CHARGE','POWER & BATTERIES', 'POWER BATTERIES', 'POWERBATTERIES',},
        'HI-FI'                 :{'TELEVISIONS & HOME CINEMA',  'TELEVISIONSHOME CINEMA','TELEVISIONS HOME CINEMA','VIDEOS DVD DIGITAL EQUIPMENT','HI-FI','PORTABLE AUDIO EQUIPMENT',},
        'FOOD'                  :{'PICNICWARE','FOOD PREPARATION','GIFT FOOD DRINK','HOT DRINK PREPARATION',},
        'IMAGING EQUIPMENT'     :{'IMAGING ACCESSORIES', 'IMAGING EQUIPMENT',},
        'JEWELLERY'             :{'JEWELLERY & WATCHES', 'JEWELLERY WATCHES', 'JEWELLERYWATCHES',},
        'TELEPHONE ACCESSORIES' :{'TECHNOLOGY ACCESSORIES', 'TELEPHONES FAX MACHINESTWO-WAY RADIOS','TELEPHONE ACCESSORIES', 'TELEPHONES FAX MACHINES TWO-WAY RADIOS', 'TELEPHONES, FAX MACHINES & TWO-WAY RADIOS'},
        'OTHER'                 :{'THEMED GIFTS','PRODUCT','PARTY DECORATIONS','CRAFT','DRESSMAKING','GIFT WRAP', 'GREETING CARDS & PERSONALISED STATIONERY', 'GREETING CARDS PERSONALISED STATIONERY',},
        UNKNOWN_KEY             :{UNKNOWN_KEY,'FILINGDESK ACCESSORIES', 'LAUNDRY APPLIANCES', 'PAPERNOTEBOOKS'},
        'SERVICE'               :{'SERVICE','WARRANTY',},  
}

ITEM_CATEGORIES = {}

def initITEM_CATEGORIES():
    if len(ITEM_CATEGORIES)==0:
        for key, val in SUPER_CATEGORIES.items():
            ITEM_CATEGORIES[val] = key
            ITEM_CATEGORIES[_clean_categories(val)] = key
    return ITEM_CATEGORIES
# ----------------------------------------------------------------------------------
#                        PRE-PROCESSING
# ----------------------------------------------------------------------------------
items_to_correct = {
                "6  SPACE GREY 32GB"   : "COMPUTER STORAGE PERIPHERALS",
                "BLANK MEDIA MEDIA STORAGE": "COMPUTER STORAGE PERIPHERALS",
                "TOSHIBA PORTABLE HARD DRIVE": "COMPUTER STORAGE PERIPHERALS",
                "TARGUS GEOLITE ESSENTIAL CASE" : "COMPUTER CASE",
                "BAGSCARRY CASES"       : "BAGS CARRY CASES",
                "NURSERY EQUIPMENT FURNITURE"   : "NURSERY FURNITURE",                              
                "CABLESADAPTERS"        : "CABLES ADAPTERS",
                "CARPETS RUGSFLOORING"  : "CARPETS RUGS FLOORING",
                "COMPUTER PERIPHERALSACCESSORIES"   : "COMPUTER PERIPHERALS ACCESSORIES",
                "DIARIESORGANISERS"     : "DIARIES ORGANISERS",
                "DECORATING"            : "DECORATIVE ACCESSORIES",
                "HP ELITEBOOK 850V6"    : "COMPUTERS",
                "IMAGING ACCESSORIES"   : "IMAGING EQUIPMENT",
                "FILINGDESK ACCESSORIES": "FILING DESK ACCESSORIES",
                "GAMES"                 : "GAMING",
                "GIFT FOOD DRINK"       : "FOOD",
                "HOT DRINK PREPARATION" : "FOOD",
                "GREETING CARDSPERSONALISED STATIONERY": "GREETING CARDS PERSONALISED STATIONERY",
                "BATHBODYCARE"              : "BATH BODYCARE",
                "HEALTHBEAUTY ELECTRICAL"   : "HEALTH BEAUTY ELECTRICAL",               
                "HEATINGCOOLING APPLIANCES" : "HEATING COOLING APPLIANCES",
                "JEWELLERYWATCHES"      : "JEWELLERY WATCHES",
                "LAUNDRYCLOTHESCARE"    : "LAUNDRY CLOTHESCARE",
                "LINGERIEHOISERY"       : "LINGERIE HOISERY",
                "LIVINGDINING FURNITURE":"LIVING DINING FURNITURE",
                "LOGITECH PEBBLE M350 BLUETOOTH MOUSE" : "COMPUTER PERIPHERALS ACCESSORIES",
                "MICROSOFT OFFICE HOME AND STUDENT 2019" : "COMPUTER SOFTWARE",
                "MENS UNDERWEARSOCKS"   : "MENS UNDERWEAR SOCKS",
                "PRESERVINGBAKING EQUIPMENT":"PRESERVING BAKING EQUIPMENT",
                "PRINTERSSCANNERS"      : "PRINTERS SCANNERS",
                "PORTABLE AUDIO EQUIPMENT" : "AUDIO ACCESSORIES",
                "NURSERY TOYS"          : "TOYS",
                "BABY PLAY EQUIPMENT"   : "TOYS",                                
                "KITCHEN SCALESMEASURES": "KITCHEN SCALES MEASURES",
                "KITCHEN UTENSILSGADGETS": "KITCHEN UTENSILS GADGETS",
                "POWERBATTERIES"        : "POWER BATTERIES",
                "STANDSBRACKETS"        : "STANDS BRACKETS",
                "STORAGEORGANISATION"   : "STORAGE ORGANISATION",
                "SUNGLASSESREADING GLASSES":"SUNGLASSES READING GLASSES",
                "TELEPHONES FAX MACHINESTWO-WAY RADIOS" : "TELEPHONES FAX MACHINES TWO-WAY RADIOS",
                "TELEVISIONSHOME CINEMA":"TELEVISIONS HOME CINEMA",
            }

# items_to_correct = {
#                 "6  SPACE GREY 32GB"    : "COMPUTER STORAGE PERIPHERALS",
#                 "BLANK MEDIA MEDIA STORAGE": "COMPUTER STORAGE PERIPHERALS",
#                 "TOSHIBA PORTABLE HARD DRIVE": "COMPUTER STORAGE PERIPHERALS",
#                 "TARGUS GEOLITE ESSENTIAL CASE" : "COMPUTER CASE",

#                 "BAGSCARRY CASES"       : "BAGS CARRY CASES",

#                 "BABY CHANGING"         : "BABY & CHILD FURNITURE",
#                 "BABY CHILD TRAVEL"     : "BABY & CHILD FURNITURE",
#                 "BABY FEEDING"          : "BABY & CHILD FURNITURE",
#                 "BABY PLAY EQUIPMENT"   : "BABY & CHILD FURNITURE",
#                 "BABYCHILD TRAVEL"      : "BABY & CHILD FURNITURE",
#                 "BABYWEAR"              : "BABY & CHILD FURNITURE",
#                 "BABYCHILD TRAVEL"      : "BABY & CHILD FURNITURE",
#                 "BABYWEAR"              : "BABY & CHILD FURNITURE",
#                 "CHILDRENS ACCESSORIES" : "BABY & CHILD FURNITURE",
#                 "CHILDRENS FOOTWEAR"    : "BABY & CHILD FURNITURE",
#                 "CHILDRENS FURNITURE"   : "BABY & CHILD FURNITURE",
#                 "CHILDRENS FOOTWEAR"    : "BABY & CHILD FURNITURE",
#                 "SCHOOLWEAR"            : "BABY & CHILD FURNITURE",
#                 "GIRLSWEAR"             : "BABY & CHILD FURNITURE",
#                 "BOYSWEAR"              : "BABY & CHILD FURNITURE",

#                 "NURSERY ACCESSORIES"   : "BABY & CHILD FURNITURE",
#                 "NURSERY EQUIPMENT FURNITURE"   : "BABY & CHILD FURNITURE",
#                 "NURSERY FURNITURE"     : "BABY & CHILD FURNITURE",
                
#                 "CABLESADAPTERS"        : "CABLES ADAPTERS",
#                 "CARPETS RUGSFLOORING"  : "CARPETS RUGSFLOORING",
                
#                 "COMPUTER PERIPHERALSACCESSORIES"   : "COMPUTER PERIPHERALS ACCESSORIES",
#                 "COMPUTER NETWORKING"  : "COMPUTER PERIPHERALS ACCESSORIES",
#                 "PRINTERS SCANNERS"    : "COMPUTER PERIPHERALS ACCESSORIES",
#                 "DIARIESORGANISERS"     : "DIARIES ORGANISERS",
#                 "DECORATING"            : "DECORATIVE ACCESSORIES",
#                 "HP ELITEBOOK 850V6"    : "COMPUTERS",
#                 "IMAGING ACCESSORIES"   : "IMAGING EQUIPMENT",
#                 "FILINGDESK ACCESSORIES": "FILING DESK ACCESSORIES",
#                 "GAMES"                 : "GAMING",

#                 "GIFT FOOD DRINK"       : "FOOD",
#                 "HOT DRINK PREPARATION" : "FOOD",

#                 "GIFT WRAP"             : "FURNITURE",
#                 "GREETING CARDSPERSONALISED STATIONERY": "GREETING CARDSPERSONALISED STATIONERY",
#                 "BATHBODYCARE"              : "HEALTH BEAUTY",
#                 "BATH BODYCARE"             : "HEALTH BEAUTY",
#                 "HEALTH BEAUTY ACCESSORIES" : "HEALTH BEAUTY",
#                 "HEALTH BEAUTY ELECTRICAL"  : "HEALTH BEAUTY",
#                 "HEALTHBEAUTY ELECTRICAL"   : "HEALTH BEAUTY",
#                 "FACIAL SKINCARE"           : "HEALTH BEAUTY",
#                 "FRAGRANCE"                 : "HEALTH BEAUTY",
#                 "HAIRCARE"                  : "HEALTH BEAUTY",
#                 "MAKEUP"                    : "HEALTH BEAUTY",

#                 "MENS CLOTHES"              : "MENS ACCESSORIES",
#                 "MENS FOOTWEAR"             : "MENS ACCESSORIES",
#                 "MENS NIGHTWEAR"            : "MENS ACCESSORIES",
#                 "MENS UNDERWEAR SOCKS"      : "MENS ACCESSORIES",


#                 "HEATINGCOOLING APPLIANCES" : "HEATING COOLING APPLIANCES",
#                 "JEWELLERYWATCHES"      : "JEWELLERY WATCHES",
#                 "LAUNDRYCLOTHESCARE"    : "LAUNDRY CLOTHESCARE",
#                 "LINGERIEHOISERY"       : "LINGERIE HOISERY",
#                 "LIVINGDINING FURNITURE":"LIVING DINING FURNITURE",
#                 "LOGITECH PEBBLE M350 BLUETOOTH MOUSE" : "COMPUTER PERIPHERALS ACCESSORIES",
#                 "MICROSOFT OFFICE HOME AND STUDENT 2019" : "COMPUTER SOFTWARE",
#                 "MENS UNDERWEARSOCKS"   : "MENS UNDERWEAR SOCKS",
#                 "PRESERVINGBAKING EQUIPMENT":"PRESERVING BAKING EQUIPMENT",
#                 "PRINTERSSCANNERS"      : "PRINTERS SCANNERS",
#                 "PORTABLE AUDIO EQUIPMENT" : "AUDIO ACCESSORIES",

#                 "NURSERY TOYS"          : "TOYS",
#                 "BABY PLAY EQUIPMENT"   : "TOYS",

#                 "BATH LINEN"            : "HOUSE LINEN",
#                 "BED LINEN"             : "HOUSE LINEN",
#                 "NURSERY LINEN"         : "HOUSE LINEN",
#                 "TABLE LINEN"           : "HOUSE LINEN",
                
#                 "BARBECUES ACCESSORIES" : "GARDENING EQUIPMENT",

#                 "BATHROOM ACCESSORIES"  : "ROOM ACCESSORIES",
#                 "BATHROOM FIXTURES"     : "ROOM ACCESSORIES",
#                 "BEDROOM FURNITURE"     : "ROOM ACCESSORIES",
#                 "WINDOW DRESSING"       : "ROOM ACCESSORIES",
#                 "KITCHEN ACCESSORIES"   : "ROOM ACCESSORIES",
#                 "KITCHEN SCALES MEASURES": "ROOM ACCESSORIES",
#                 "KITCHEN SCALESMEASURES": "ROOM ACCESSORIES",
#                 "KITCHEN STORAGE"       : "ROOM ACCESSORIES",
#                 "KITCHEN UTENSILS GADGETS":"ROOM ACCESSORIES",
#                 "KITCHEN SCALESMEASURES": "ROOM ACCESSORIES",
#                 "KITCHEN UTENSILSGADGETS": "ROOM ACCESSORIES",
#                 "DOOR FURNITURE"        :"ROOM ACCESSORIES",

#                 "POWERBATTERIES"        : "POWER BATTERIES",
#                 "STANDSBRACKETS"        : "STANDS BRACKETS",
#                 "STORAGEORGANISATION"   : "STORAGE ORGANISATION",
#                 "SUNGLASSESREADING GLASSES":"SUNGLASSES READING GLASSES",
#                 "TELEPHONES FAX MACHINESTWO-WAY RADIOS" : "TELEPHONES FAX MACHINES TWO-WAY RADIOS",
#                 "TELEVISIONSHOME CINEMA":"TELEVISIONSHOME CINEMA",
#             }

def pre_processing(X, y=None,save_file_path = None, force=0, with_drop=True, verbose=0):
    """1. Fusion des DF X et Y
       2. Suppression des colonnes de code
       3. Uniformisation des écritures des items, make et model
       4. Recaclculer le nombre d'items dans le panier
       5. Ajouter le montant total du panier
       6. Réorganisation des colonnes
       X. Sauvegarde du fichier

    Args:
        X (DataFrame): _description_
        y (DataFrame, optional): _description_. Defaults to None.
        save_file_path (str, optional): _description_. Defaults to None.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "pre_processing"
    dataset = None

    if not force and exists(save_file_path):
        dataset = load_dump_file(data_set_path=save_file_path, file_name=None, verbose=verbose)
        if verbose>0:
            info(short_name, f"dump file exist {dataset.shape} --> {surligne_text('LOAD')}")

    if dataset is None:
        # 0. 
        
        # 1. Fusion des DF X et Y
        if y is not None:
            dataset = pd.merge(left=X, right=y, on='ID', indicator=True)
            if verbose>0:
                info(short_name, f'\n{dataset["_merge"].value_counts()}')
            
            dataset = dataset.drop(columns=["_merge"])   
        else:
            dataset = X.copy()
        dataset = df_sparse_to_dense(dataset)

        # 2. Suppression des colonnes de goods_code
        if verbose>0:
            info(short_name, f'input {dataset.shape}')
        if with_drop:
            dataset = drop_numeroted_data_col(df=dataset, cols_name=["goods_code"], verbose=verbose)
        if verbose>0:
            info(short_name, f'without googs_code {dataset.shape}')

        # 3. Uniformisation des écritures des items
        # Certains items sont identiques mais écris différemment, une étape d'uniformisation est nécessaire...
        for i in tqdm(range(1, 25), desc="clean", disable=verbose<1):
            col = f'item{i}'
            
            for old, new in items_to_correct.items():
                dataset.loc[dataset[col].notna(), col] = dataset.loc[dataset[col].notna(), col].str.replace(old, new)
            
            for col in [f'item{i}', f'make{i}',  f'model{i}']:
                dataset.loc[dataset[col].notna(), col] = dataset.loc[dataset[col].notna(), col].apply(lambda x: _clean_categories(input_str=x))
            # à l'origine il y avait 173 items, après nettoyage => 136 items
            # à l'origine il y avait 829 maker, après nettoyage => 827 makers
        
        # 4. Recaclculer le nombre d'items dans le panier
        # 5. Ajouter le montant total du panier
        dataset['cart_amount'] = 0
        dataset['cart_nb_product'] = 0
        for i in range(1, 25):
            mask = dataset[f'Nbr_of_prod_purchas{i}'].notna()
            if dataset[mask].shape[0]>0:
                dataset.loc[mask, 'cart_nb_product'] = dataset.loc[mask, 'cart_nb_product'] + dataset.loc[mask, f'Nbr_of_prod_purchas{i}']
                dataset.loc[mask, 'cart_amount'] = dataset.loc[mask, 'cart_amount'] + dataset.loc[mask, f'cash_price{i}']
        
        # 6. On déplace les colonnes intéressantes au début de la DF
        cols = list(dataset.columns)
        for c_n in ['Nb_of_items', 'cart_nb_product','fraud_flag',  'cart_amount']:
            try:
                cols.remove(c_n)
                cols.insert(1, c_n)
            except :
                pass

        dataset = dataset[cols]

        if verbose>0:
            info(short_name, f'output {dataset.shape}')

        dump_file(data_set_path=save_file_path, df=dataset, verbose=verbose)
    
    return dataset


def encode_data(data_set_path,df, file_name= "encoded_data.csv", force=0, with_drop=1, to_encode_data=['item', 'make'], add_item_group=1,verbose=0):
    """
    0. Recharchement du dernier fichier généré
    1. drop unused columns
    2. Encode `to_encode_data` columns
    3. Typing datas to reduce memory space
    4. drop encoded data
    5. Add amount column
    6. Groups items if `add_item_group` 
    Final. Save result and return if
    
    Note :  If target file ever exist and not `force` load the file.

    Args:
        data_set_path (str): _description_
        df (DataFrame or None): _description_
        file_name (str, optional): _description_. Defaults to "encoded_data.csv".
        force (int, optional): _description_. Defaults to 0.
        with_drop (int, optional): _description_. Defaults to 1.
        to_encode_data (list, optional): _description_. Defaults to ['item', 'make'].
        add_item_group (int, optionnal) : 1 to group item, 2 to remove items columns after, 0 to do nothing, deefault 1
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "encode_data"
    df_res = load_dump_file(data_set_path=data_set_path, file_name=file_name, force=force, verbose=verbose)
    save_file_path = join(data_set_path, file_name)
    
    if df_res is None:
        # Local constants
        col_to_cal = ['Nbr_of_prod_purchas', 'cash_price']
        
        if not force:
            # Parcours de la liste des encodages à l'envers pour voir si le traitement n'a pas déjà été initié et interrompue
            to_keep = []
            if verbose>1: debug(short_name, f"STEP 0 => look for last dump...")
            for i in range(len(to_encode_data)-1, -1, -1):
                to_enc = to_encode_data[i]
                f_name = save_file_path.replace(".csv", f'_{to_enc.upper()}.csv')
                if exists(f_name):
                    if verbose>1: debug(short_name, f"STEP 0 => look for last dump... {to_enc} found, loading...")
                    # Chargement du dernier fichier sauvegardé
                    df_res = load_dump_file(f_name, file_name=None, verbose=verbose)
                    if verbose>0:
                        info(short_name, f"STEP 0 => Last step execution {to_enc} with {df_res.shape}---> {surligne_text('LOAD')}")
                    break
                else:
                    # Pour que le tableau final soit dans le même ordre que to_encode_data d'origine
                    to_keep.insert(0, to_enc)

            to_encode_data = to_keep
        
        if df_res is None:
            df_res = df.copy()
            if verbose>0:
                info(short_name, f"{df_res.shape} datas to encode")
        
        if len(to_encode_data)>0:
            if verbose>0:
                info(short_name, f"STEP 0 => remain {len(to_encode_data)} to encod {to_encode_data}")
        
            # 1. on réduit la taille de la df aux données que l'on souhaite conserver uniquement pour réduire le taille en mémoire et le temps de traitement
            if verbose>1:
                debug(short_name, f"STEP 1 => remove unsed columns")
            to_del_col = deepcopy(NUMEROTED_COL_PREFIX)
            
            for e in to_encode_data+col_to_cal:
                try:
                    to_del_col.remove(e)
                except:
                    pass
            
            for to_del in to_del_col:
                df_res = drop_numeroted_data_col(df=df_res, cols_name=[to_del], verbose=verbose)
                
            if verbose>0:
                info(short_name, f"STEP 1 => {len(to_del_col)} unsed columns {surligne_text('DROP')} --> {df_res.shape}")

            # Local constants
            timing = {
                'item' : '> 30 min',
                'make' : '~= 270 min ~= 4h30'
            }
            for to_enc in tqdm(to_encode_data, disable=verbose<1):

                # /!\ 42 min de traitement lors du forçage
                if verbose>0:
                    warn(short_name, f"STEP 2 => encoding {to_enc.upper()} to features {surligne_text(timing.get(to_enc, 'UNKNOWN duration'), 'red')}...")

                list_data = _get_data_list(df=df_res, col_name=to_enc, verbose=verbose-1)
                df_res = _encode_numeroted_data_to_features(df=df_res, data_list=list_data, data_col_name=to_enc, with_drop=with_drop, verbose=verbose)

                df_res = df_res.reset_index()
                df_res = df_res.set_index('index')
                df_res = drop_numeroted_data_col(df=df_res, cols_name=[to_enc], verbose=verbose)
                f_name = save_file_path.replace(".csv", f'_{to_enc.upper()}.csv')
                dump_file(data_set_path=f_name, df=df_res, verbose=verbose)
                            
                if verbose>0:
                    info(short_name, f"STEP 2 => encoding {to_enc.upper()} to features --> {df_res.shape} {surligne_text('SAVE')}")
                    if verbose>1:
                        debug(short_name, f"{f_name}")
        else:
            if verbose>0:
                info(short_name, f"STEP 0 => NO remain to encod.")
            
        df_res = drop_numeroted_data_col(df=df_res, cols_name=col_to_cal, verbose=verbose)
        if verbose>0:
            info(short_name, f"STEP 3 => columns {col_to_cal} --> {df_res.shape} {surligne_text('DROP')}")

        df_res = add_amounts(x_df_input=df_res, verbose=verbose)
        if verbose>0:
            info(short_name, f"STEP 4 => Add Amount {surligne_text('DONE')}")

        if add_item_group:
            if verbose>1: debug(short_name, f"STEP 4 => Group items")
            with_drop= 1 if add_item_group == 2 else 0
            f_name = save_file_path.replace(".csv", f'_groups.csv')
            df_res = add_item_group_datas_to_features(df=df_res,data_set_path=f_name, with_drop=with_drop, verbose=verbose)
            if verbose>0:
                info(short_name, f"STEP 4 => Group items {surligne_text('DONE')}")

        # Sauvegarde du fichier
        if verbose>1: debug(short_name, f"Final STEP => Saving File ...")
        dump_file(data_set_path=data_set_path, df=df_res, file_name=file_name, verbose=verbose)
        
    return df_res

def _encode_numeroted_data_to_features(df, data_list, data_col_name='item', with_drop=0, verbose=0):
    short_name = "encode_numeroted_data_to_features"
    if verbose > 0: print(f"[{short_name}]\tINFO: Conversion {data_col_name.upper()} to feature... START")
    # Il faut prévoir le cas inconnu pour y ajouter des données inconnues.
    data_list.add("UNKNOWN")
    df_res = df.copy()
    
    for current_name in tqdm(data_list, desc=f"{data_col_name}_list", disable=verbose<1):
        col = data_col_name+"_"+current_name+"_nb"
        df_res[col] = df_res.apply(lambda x : nb_by_col(current_name=current_name, data_col_name=data_col_name, row=x, col_addition='Nbr_of_prod_purchas', verbose=verbose-1), axis=1)
        # Typage des données
        df_res[col] = df_res[col].round(decimals=0)
        df_res[col] = df_res[col].astype("int8")
        col = data_col_name+"_"+current_name+"_cash"
        df_res[col] = df_res.apply(lambda x : nb_by_col(current_name=current_name, data_col_name=data_col_name, row=x, col_addition='cash_price', verbose=verbose-1), axis=1)
        # Typage des données
        df_res[col] = df_res[col].round(decimals=2)
        df_res[col] = df_res[col].astype("float16")

    if verbose > 0: print(f"[{short_name}]\tINFO: Conversion {data_col_name.upper()} to feature............ END")
    
    if with_drop:
        if verbose>0:
            info(short_name, f"Drop {data_col_name} columns ... ")
        df_res = drop_numeroted_data_col(df=df_res, cols_name=[data_col_name], verbose=verbose)
       
    return df_res

def add_item_group_datas_to_features(df,data_set_path, file_name=None,force=0, with_drop=0, verbose=0):
    """Group items by categories

    Args:
        df (DataFrame): _description_
        with_drop (int, optional): To drop the items columns. Defaults to 0.
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "add_item_group_datas_to_features"
    df_res = load_dump_file(data_set_path=data_set_path, file_name=file_name, force=force, verbose=verbose)
            
    if df_res is None:

        data_col_name='item'

        df_columns = prefixed_cols(df=df, col_prefix=data_col_name, verbose=verbose)
        remains_columns_cash = []
        remains_columns_nb = []
        for col in df_columns:
            if col.endswith('_cash'):
                remains_columns_cash.append(col)
            elif  col.endswith('_nb'):
                remains_columns_nb.append(col)

        for cate in SUPER_CATEGORIES.keys():
            # on ne traite que les super catégories connues dans un premier temps
            if UNKNOWN_KEY != cate:
                cols_names_nb,cols_names_cash,remains_columns_nb, remains_columns_cash = get_groups_columns(cate=cate, df_columns=df_columns, 
                                                                                            remains_columns_nb=remains_columns_nb, 
                                                                                            remains_columns_cash=remains_columns_cash, 
                                                                                            data_col_name=data_col_name, verbose=verbose-1)
                try:
                    if len(cols_names_cash) != len(cols_names_nb):
                        warn(short_name, f"{cate} Le nombre de données est différent : {cols_names_nb} vs {cols_names_cash}")
                    if len(cols_names_nb)>0:
                        df["group_"+data_col_name+"_"+cate+"_nb"] = df[cols_names_nb].sum(axis=1)
                        if with_drop:
                            df = df.drop(cols_names_nb, axis=1)
                    if len(cols_names_cash)>0:
                        df["group_"+data_col_name+"_"+cate+"_cash"] = df[cols_names_cash].sum(axis=1)
                        if with_drop:
                            df = df.drop(cols_names_cash, axis=1)
                except Exception as err:
                    if verbose>0:
                        warn(cate, f'{err}\n-------------------------------------------------------------------------')
                        if verbose>1:
                            debug(cate, f'{cols_names_nb}')
                            debug(cate, f'{cols_names_cash}')
        
        # Traitement des valeurs inconnues
        unk_cash = "group_"+data_col_name+"_"+UNKNOWN_KEY+"_cash"
        unk_nb = "group_"+data_col_name+"_"+UNKNOWN_KEY+"_nb"
        df[unk_cash] = 0
        df[unk_nb] = 0
        if len(remains_columns_nb)>0:
            if verbose > 1:
                debug(short_name, f"{len(remains_columns_nb)} {UNKNOWN_KEY} nb columns to proceed")
            df[unk_nb] = df[remains_columns_nb].sum(axis=1)
            if with_drop:
                df = df.drop(remains_columns_nb, axis=1)
                
        if len(remains_columns_cash)>0:
            if verbose > 1:
                debug(short_name, f"{len(remains_columns_cash)} {UNKNOWN_KEY} cash columns to proceed")
            df[unk_cash] = df[remains_columns_cash].sum(axis=1)
            if with_drop:
                df = df.drop(remains_columns_cash, axis=1)
        
        df_res = df
        dump_file(data_set_path, df=df_res, file_name=file_name, verbose=verbose)

    if verbose > 0: info(short_name, f"{df_res.shape} result data.")
    return df_res




def get_groups_columns(cate, df_columns, remains_columns_nb, remains_columns_cash, data_col_name="item", verbose=0):
    """
    Filter the group's columns switch the DataFrame's columns
    
    Args:
        cate (str): Group's name
        df_columns (list(str)): DataFrame column's name
        remains_columns_nb (list(str)): _description_
        remains_columns_cash (list(str)): _description_
        data_col_name (str, optional): _description_. Defaults to "item".
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        tuple(cols_names_nb,cols_names_cash,remains_columns_nb, remains_columns_cash): _description_
    """
    short_name="clean_groups"
    cols_names_nb = set()
    cols_names_cash = set()
    for col in SUPER_CATEGORIES.get(cate, set()):
        curr = data_col_name+"_"+col+"_nb"
        if curr in df_columns:
            cols_names_nb.add(curr)
            try:
                remains_columns_nb.remove(curr)
            except:
                pass
        curr = data_col_name+"_"+col+"_cash"
        if curr in df_columns:
            cols_names_cash.add(curr)
            try:
                remains_columns_cash.remove(curr)
            except:
                pass
    
    if verbose>0:
        now_nb = len(cols_names_nb)
        now_cash = len(cols_names_cash)
        info(short_name, f"{cate} => {now_nb} NB + {now_cash} cash columns to proceed")
        if verbose>1:
            debug(short_name, f"{cate} =>{cols_names_cash}")
            debug(short_name, f"{cate} =>{cols_names_nb}")
        

    return cols_names_nb,cols_names_cash,remains_columns_nb, remains_columns_cash

def test_encode_to_train_compatibility(file_path, train_columns, force=False, verbose=0):
    """Load the official test df and add : amount and reducre data by typing, then save the new df.
    If the df have been save, just load it.

    Args:
        file_path (str): _description_
        train_columns (list(str)): _description_
        verbose (int, optional): _description_. Defaults to 0.

    Returns:
        DataFrame: _description_
    """
    short_name = "preprocess_test_df"
    save_path = join(dirname(file_path), DATASET_FILES_NAMES.get("dataset_test_train_compatible"))
    test_origin = load_dump_file(data_set_path=save_path, force=force, verbose=verbose)
    
    if test_origin is None:
        test_origin = load_dump_file(data_set_path=file_path, force=force, verbose=verbose)
        prev_shape = test_origin.shape

        # Ajout des colonnes manquantes
        test_origin = add_amounts(test_origin, verbose=verbose)

        test_cols = test_origin.columns
        for col in train_columns:
            if col not in test_cols:
                test_origin[col] = 0
        
        # Traitement des colonnes inconnues lors du train
        to_drop_cols = []
        # to_drop_dict = defaultdict(defaultdict(list))

        to_drop_dict = {
            'item'  : defaultdict(list),
            'make'  : defaultdict(list),
            'model' : defaultdict(list),
        }

        # Expected format :      
        # to_drop_dict = {
        #     'item' : {'cash': [], 'nb':[]},
        #     'make' : {'cash': [], 'nb':[]},
        #     'model' : {'cash': [], 'nb':[]},
        # }
        for col in test_origin.columns:
            if col not in train_columns:
                to_drop_cols.append(col)
                spl = col.split("_")
                k = spl[0]
                t = spl[-1]
                to_drop_dict[k][t].append(col)

        if len(to_drop_cols)>0:
            for data_col_name, t_dict in to_drop_dict.items():
                for t, col_to_drop in t_dict.items():
                    # Addition des colonnes du type t (cash or nb) à la UNKNOWN de data_col_name (item or make)
                    try:
                        unk_nb = data_col_name+"_"+UNKNOWN_KEY+"_"+t
                        test_origin[unk_nb] = test_origin[unk_nb] + test_origin[col_to_drop].sum(axis=1)
                    except Exception as err:
                        if verbose>0:
                            warn(short_name, f"Error when proceed : {unk_nb} with {col_to_drop} --> {err}.")
                    
                    # Additionne les items non connu au groupe UNKNOWN
                    if data_col_name == 'item':
                        try:
                            unk_nb = "group_"+data_col_name+"_"+UNKNOWN_KEY+"_"+t
                            test_origin[unk_nb] = test_origin[unk_nb] + test_origin[col_to_drop].sum(axis=1)
                        except Exception as err:
                            if verbose>0:
                                warn(short_name, f"Error when proceed : {unk_nb} with {col_to_drop} --> {err}.")
        
        test_origin = test_origin[train_columns]
        if verbose>0:
            info(short_name, f"{test_origin.shape} test données mises à jour (on load : {prev_shape})")

        dump_file(data_set_path=save_path, df=test_origin, verbose=verbose)

    if verbose>0:
        info(short_name, f"{test_origin.shape} test données chargées")
    return test_origin

def save_make_light(df, data_set_path, file_name="train_MAKE_light_2023_01_19.csv", verbose=0):
    short_name = "save_make_light"
    cols = prefixed_cols(df=df, col_prefix="item_", verbose=verbose)
    
    df_light =df.copy()
    df_light = df_light.drop(columns=cols)
    if verbose>0:
        info(short_name, f"before : {df.shape} after : {df_light.shape}")
    
    dump_file(data_set_path=data_set_path, file_name=file_name, df=df_light, verbose=verbose)

    return df_light

# ----------------------------------------------------------------------------------
#                        DATA FUNCTIONS
# ----------------------------------------------------------------------------------

def nb_by_col(current_name, data_col_name, row, col_addition='Nbr_of_prod_purchas', verbose=0):
    nb_item = 0
    for i in range(1, 25):
        item_val = row[f'{data_col_name}{i}']
        if isinstance(item_val, str) and current_name == item_val:
            nb_item += row[f'{col_addition}{i}']
    return round(nb_item, 2)

def drop_numeroted_data_col(df, cols_name, verbose=0):
    short_name = "drop_numeroted_data_col"
    n_df = df.copy()
    nb_col_droped = 0 
    if verbose>1:
        debug(short_name, f"input shape {n_df.shape}")  
        
    for i in tqdm(range(1, 25), desc=f"[{short_name}\t", disable=verbose<2)  :
        for col_name in cols_name:
            try:
                n_df = n_df.drop(columns=[f'{col_name}{i}'])
                nb_col_droped += 1
            except:
                pass
    
    if verbose>0:
        info(short_name, f"{nb_col_droped} columns droped")
        if verbose>1:
            debug(short_name, f"output shape {n_df.shape}")   
    return n_df


def _clean_categories(input_str):
    # BLANK MEDIA & MEDIA STORAGE => BLANK MEDIA MEDIA STORAGE
    output_str = input_str
    if isinstance(output_str, str):
        output_str = output_str.replace(" & ", "")
        # CHILDREN S ACCESSORIES => CHILDRENS ACCESSORIES
        output_str = output_str.replace(" S ","S ")
        # TELEPHONES, FAX MACHINES & TWO-WAY RADIOS => TELEPHONES FAX MACHINES TWO-WAY RADIOS
        output_str = output_str.replace(",", "")
        # 2HP ELITEBOOK 850V6 => HP ELITEBOOK 850V6
        if output_str.startswith("2"):
            output_str = output_str[1:]
        
        output_str = output_str.replace("SACCESSORIES", "S ACCESSORIES")
        output_str = output_str.replace("TDESCRIPTION", "T DESCRIPTION")
        
        output_str = output_str.strip()
    return output_str

def get_data_list(df, col_name, verbose=0):
    return _get_data_list(df=df, col_name=col_name, verbose=verbose)

def _get_data_list(df, col_name, verbose=0):
    short_name = f'get_{col_name}_list'
    
    items_list = set()
    for i in tqdm(range(1, 25), desc=f"{col_name}_list", disable=verbose<1):
        items_list = items_list | set(df[f'{col_name}{i}'].unique())
    
    if verbose>1:
        debug(short_name, f'{len(items_list)} with NA value')
    
    items_list.remove(np.nan)

    if verbose>0:
        info(short_name, f'{len(items_list)} {col_name} without NA value')
        if verbose>1:
            debug(short_name, f'',items_list)
    return items_list


# ----------------------------------------------------------------------------------
#                        PRE-PROCESSING V
# ----------------------------------------------------------------------------------
import re
from enum import Enum

def convert_code_produit(ref_produit, verbose=0):
    try:
        code_produit = int(ref_produit)
    except:
        end = ""
        try:
            end = re.findall(r'\d+', ref_produit)
            if len(end)>0:
                end = end[0]
            else:
                end = ""
        except:
            pass
        code_produit = int("999000"+end)
    return code_produit

class GammeType(Enum):
    less = 10
    bronze = 20
    argent = 30
    gold = 40
    platine = 50

def gamme(pu):   
    g_ = GammeType.less
    if pu > 1500.0:
        g_ = GammeType.platine
    elif pu > 1000 :
        g_ = GammeType.gold
    elif pu > 500 :
        g_ = GammeType.argent
    elif pu > 20 :
        g_ = GammeType.bronze
    return g_.value

def get_categorie_df(save_file_path, dataset_categorie,  limit = 30, verbose=0):
    short_name = "get_categorie_df"
    
    if save_file_path is not None and exists(save_file_path):
        dataset_categorie  = load_dump_file(data_set_path=save_file_path, verbose=verbose-1)
        if verbose > 0: info(short_name, f"{dataset_categorie.shape} data from {basename(save_file_path)} {surligne_text('LOAD')}")
    elif dataset_categorie is not None:
        if isinstance(dataset_categorie, list):
            dataset_categorie = pd.concat(dataset_categorie)

        try:
            dataset_categorie = df_sparse_to_dense(dataset_categorie)
        except:
            pass
        dataset_categorie = dataset_categorie.value_counts()
        dataset_categorie = dataset_categorie.reset_index()
        dataset_categorie = dataset_categorie.rename(columns={0:'count'})
        dataset_categorie.loc[dataset_categorie['count']<limit, 'categorie'] = UNKNOWN_KEY
        dataset_categorie.loc[dataset_categorie['count']<limit, 'count'] = dataset_categorie.loc[dataset_categorie['count']<limit, 'count'].sum()
        dataset_categorie = dataset_categorie.drop_duplicates(['categorie'], keep='first')

        dataset_categorie = dataset_categorie.sort_values('categorie')
        dataset_categorie = dataset_categorie.reset_index(drop=True)
        dataset_categorie = dataset_categorie.reset_index()
        dataset_categorie = dataset_categorie.rename(columns={'index':'embedding'})
        
        dump_file(data_set_path=save_file_path, df=dataset_categorie, verbose=verbose-1)
        if verbose > 0: info(short_name, f"{dataset_categorie.shape} data in {basename(save_file_path)} {surligne_text('SAVE')}")
    return dataset_categorie

def set_categorie_embedding(df_cat, categorie):
    cat = UNKNOWN_KEY
    if df_cat[df_cat['categorie']==categorie].shape[0]>0:
        cat = df_cat.loc[df_cat['categorie']==categorie, 'embedding'].values[0]
    else:
        cat = df_cat.loc[df_cat['categorie']==UNKNOWN_KEY, 'embedding'].values[0]
    return cat

def transpose_df_by_item(input_df, dataset_categorie=None, save_categorie_path=None, limit=2, data_set_save_path=None, verbose=0):
    short_name = "transpose_df_by_item"
    start_cols = ['ID', 'Nb_of_items', 'fraud_flag',]

    if 'fraud_flag' not in input_df.columns:
        start_cols.remove('fraud_flag')

    cols = deepcopy(start_cols)
    local_numeroted_col_prefix = deepcopy(NUMEROTED_COL_PREFIX)

    for e in NUMEROTED_COL_PREFIX:
        if e+"1" in list(input_df.columns):
            cols.append(e+"1")
        else:
            try:
                local_numeroted_col_prefix.remove(e)
            except:
                pass
    
    dataset_transpose = input_df[cols]
    if verbose>0:
        info(short_name, f"input shape {input_df.shape}")
        
    for i in range(2, 25):
        cols = deepcopy(start_cols)
        
        to_rename = {}
        for e in local_numeroted_col_prefix:
            cols.append(e+str(i))
            to_rename[e+str(i)] = e+"1"
            
        temp =  input_df[cols]
        temp = temp.rename(columns=to_rename)
        temp = temp[temp['item1'].notna()]
        dataset_transpose = pd.concat([dataset_transpose, temp])
        if verbose>1:
            debug(short_name, f"temp shape {dataset_transpose.shape}")

    if verbose>1:
        debug(short_name, f"Sorting...")
    dataset_transpose = dataset_transpose.sort_values(by=['ID'])
    dataset_transpose = dataset_transpose.reset_index(drop=True)

    if verbose>1:
        debug(short_name, f"Typing...")
    dataset_transpose['Nb_of_items'] = dataset_transpose['Nb_of_items'].astype(int)
    dataset_transpose['Nbr_of_prod_purchas1'] = dataset_transpose['Nbr_of_prod_purchas1'].astype(int)

    cols = ['goods_code1', 'model1', 'item1',  'make1', 'Nbr_of_prod_purchas1', 'cash_price1',  'ID', 'Nb_of_items', 'fraud_flag']
    if 'fraud_flag' not in input_df.columns:
        cols.remove('fraud_flag')

    dataset_transpose=dataset_transpose[cols]
    dataset_transpose=dataset_transpose.rename(columns={'goods_code1':'ref_produit', 'item1':'categorie', 'model1': 'model', 'make1':'make','Nbr_of_prod_purchas1':"Q", 'cash_price1':'item_amount'})
    dataset_transpose['PU'] = dataset_transpose['item_amount'] / dataset_transpose['Q']
    dataset_transpose['code_produit'] = dataset_transpose['ref_produit'].apply(lambda x:convert_code_produit(ref_produit=x, verbose=verbose))

    dataset_transpose['gamme'] = dataset_transpose['PU'].apply(lambda x: gamme(pu=x))

    if dataset_categorie is None:
        dataset_categorie = get_categorie_df(save_file_path=save_categorie_path,dataset_categorie=dataset_transpose[['categorie']], limit=limit)
        
    if dataset_categorie is not None:
        dataset_transpose['cat_embed'] = dataset_transpose['categorie'].apply(lambda x: set_categorie_embedding(df_cat=dataset_categorie, categorie=x))

    if verbose>0:
        info(short_name, f"output shape {dataset_transpose.shape}")
    
    if data_set_save_path is not None:
        dump_file(data_set_path=data_set_save_path, df=dataset_transpose, verbose=verbose)

    return dataset_transpose

