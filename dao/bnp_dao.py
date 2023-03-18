#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module to explore the database cojoden

Project: BNP
=======

Usage:
======
    python bnp.dao.py

    configuration file : local_mysql.env
"""
__authors__     = ("Aurélie RAOUL")
__contact__     = ("aurelie.raoul@yahoo.fr")
__copyright__   = "MIT"
__date__        = "2023-03-11"
__version__     = "1.0.0"

from dotenv import dotenv_values
from os import getcwd
from tqdm import tqdm
import argparse
from math import isnan

from os.path import join, exists
# Récupère le répertoire du programme
execution_path = getcwd()
project_name = "projet_bnp_paribas_fraudeurs"
# Permet de gérer le cas d'une exécution dans un notebook par exemple
execution_path = execution_path.split("PROJETS")[0]
execution_path = join(execution_path, "PROJETS", project_name)
import sys
sys.path.append(execution_path)

from abstract_dao import GenericMySQLDao
from model.bnp_paribas_util import load_dump_file, df_sparse_to_dense

# ----------------------------------------------------------------------------------
#                         DATABASE INFORMATIONS
# ----------------------------------------------------------------------------------
# recupere les données du dotenv
local_env_path = join(execution_path, "dao","local_mysql.env")
DENV = dotenv_values(local_env_path)
global_db_user = DENV['db_user']
global_db_pwd = DENV['db_pwd']
global_db_host = DENV['db_host']
global_db_name = DENV['db_name']

# Les tables sont dans l'ordre de création
TABLES_NAME = ["categorie", "catalogue", "fabricant"]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                               CLASSES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class BnpDao(GenericMySQLDao):
    
    def __init__(self):
        """Constructeur database datas come from local_mysql.env """
        super(BnpDao, self).__init__(tables_names=TABLES_NAME, host=global_db_host, db_name=global_db_name, db_user=global_db_user, db_pwd=global_db_pwd)

    def insert_make(self, make, contact=None, verbose=0):
        """
        Args:
            make (str or list(str)): _description_
            contact (str or list(str), optional): _description_. Defaults to None.
            verbose (int, optional): log level. Defaults to 0.

        Raises:
            AttributeError: input parameter missing

        Returns:
            _type_: _description_
        """
        short_name = "insert_make"
        if make is None or len(make)==0:
            raise AttributeError("Designation require")

        if isinstance(make, str): 
            make = [make]
            contact = [contact]
        if contact is None:
            contact = [None for _ in range(len(make))]
        
        sql = f"INSERT INTO `bnp`.`fabricant` (`id`, `designation`, `contact`) VALUES "
        sep = ""

        for des, cont in zip(make, contact):
            des = f"'{des.strip()}'" if not des.startswith("'") else des
            if cont is None or len(cont.strip())>0:
                cont = 'NULL'
            else:
                cont = f"'{cont.strip()}'" if not cont.startswith("'") else cont
            
            sql += f"{sep}\n(NULL, {des}, {cont})" 
            sep = ","

        sql += f";"
        res = super(BnpDao, self).execute_sql(sql=sql, verbose=verbose)
        res = super(BnpDao, self).execute_sql(sql="SELECT max(`id`) FROM `bnp`.`fabricant`;")
        if res is not None and len(res)>0:
            res = res[0][0]
        return res
    
    def insert_category(self, category, verbose=0):
        """
        Args:
            category (str or list(str)): _description_
            verbose (int, optional): log level. Defaults to 0.

        Raises:
            AttributeError: input parameter missing

        Returns:
            _type_: _description_
        """
        if category is None or len(category)==0:
            raise AttributeError("Libelle require")

        if isinstance(category, str): 
            category = [category]
        
        sql = f"INSERT INTO `bnp`.`categorie` (`id_categorie`, `libelle`) VALUES "
        sep = ""
        for des in category:
            des = f"'{des.strip()}'" if not des.startswith("'") else des
            sql += f"{sep}\n(NULL, {des})" 
            sep = ","

        sql += f";"
        res = super(BnpDao, self).execute_sql(sql=sql, verbose=verbose)
        res = super(BnpDao, self).execute_sql(sql="SELECT max(`id_categorie`) FROM `bnp`.`categorie`;")
        if res is not None and len(res)>0:
            res = res[0][0]
        return res
    
    def insert_or_update_product(self, product_code, model, pu, make, category, verbose=0):
        short_name = "insert_or_update_product"
        if product_code is None:
            raise AttributeError("product_code require")

        cp = f"'{product_code}'" if not isinstance(product_code, str) or not product_code.startswith("'") else product_code
        m = "NULL" if model is None else model
        m  = f"'{model.strip()}'" if model is not None and (not isinstance(model, str) or not model.startswith("'")) else m
        p = pu if pu is not None else 'NULL'
        # Récupération de l'identifiant du fabriquant et de la catégorie
        f = make
        if f is not None and not (isinstance(make, float) and isnan(make)):
            if isinstance(f, str):
                f = self.select_make(make=f, insert_if_not_exist=True, verbose=verbose)
                if f is not None and len(f)>0:
                    f = f[0][0]
        else:
            f = "NULL"
        c = category
        if c is not None and not (isinstance(category, float) and isnan(category)):
            if isinstance(c, str):
                c = self.select_category(category=c, insert_if_not_exist=True, verbose=verbose)
                if c is not None and len(c)>0:
                    c = c[0][0]
        else:
            c = "NULL"

        sql = ""
        exist = self.select_product(product_code=product_code, verbose=verbose)
        if exist is not None and len(exist)>0:
            exist = exist[0]
            current = (cp.replace("'", ""), m.replace("'", ""), p, f, c)
            change = exist != current
            
            if change:
                sql = f"UPDATE `bnp`.`catalogue` SET "
                sep = ""
                if model is not None:
                    sql += sep + f" modele = {m}"
                    sep = ", "

                if pu is not None and not (isinstance(pu, float) and isnan(pu)):
                    sql += sep + f" PU = {p}"
                    sep = ", "
                
                if make is not None and not (isinstance(make, float) and isnan(make)):
                    sql += sep + f" fabricant = {f}"
                    sep = ", "
                
                if category is not None and not (isinstance(category, float) and isnan(category)):
                    sql += sep + f" categorie = {c}"
                    sep = ", "
                
                sql = sql + f" WHERE code_produit = {cp} ;"
            else:
                if verbose>1:
                    print(f"[{short_name}]\tDEBUG : no update require for {exist}.")
        else:
            sql = f"INSERT INTO `bnp`.`catalogue` (`code_produit`, `modele`, `PU`, `fabricant`, `categorie`) VALUES "                         
            sql += f"({cp}, {m}, {p}, {f}, {c});" 
        
        res = None
        if sql is not None and len(sql) >0:
            res = super(BnpDao, self).execute_sql(sql=sql, verbose=verbose)
        return res
        

    def insert_catalog(self, product_codes, models, pus, make, categories, verbose=0):
        short_name = "insert_catalog"
        if product_codes is None or len(product_codes)==0:
            raise AttributeError("product_codes require")

        if isinstance(product_codes, str): 
            product_codes = [product_codes]
            models = [models]
            pus = [pus]
            make = [make]
            categories = [categories]
        
        models      = [None for _ in range(len(product_codes))] if models is None else models
        pus         = [None for _ in range(len(product_codes))] if pus is None else pus
        make        = [None for _ in range(len(product_codes))] if make is None else make
        categories  = [None for _ in range(len(product_codes))] if categories is None else categories

        nb_insert = 0
        nb_ever_exist = 0
        for cp, m, p, f, c in tqdm(zip(product_codes, models, pus, make, categories), total=len(product_codes), desc=short_name, disable=verbose<1):
            
            res = self.insert_or_update_product(product_code=cp, model=m, pu=p, make=f, category=c, verbose=verbose)
            nb_insert += 1 
        if verbose>0:
            print(f"[{short_name}]\tINFO:{nb_insert} rows inserted and {nb_ever_exist} rows ever existed")

        return nb_insert + nb_ever_exist
    
    def select_category(self, category, insert_if_not_exist=False, limit=100, verbose=0):
        if category is None or len(category)==0:
            raise AttributeError("Libelle require")
        
        if not category.startswith("'"):
            category = f"'{category.strip()}'"

        sql = f"SELECT * FROM categorie WHERE libelle={category}" 
        if limit is not None:
            sql += f" LIMIT {limit}"
        sql += ";"
        res = super(BnpDao, self).execute_sql(sql = sql, verbose=verbose)
        if insert_if_not_exist and res is None or len(res) == 0:
            res = self.insert_category(category=category, verbose=verbose)
            res = self.select_category(category=category, verbose=verbose)
        return res
    
    def select_make(self, make, insert_if_not_exist=False, limit=100, verbose=0):
        if make is None or len(make)==0:
            raise AttributeError("Designation require")
        
        if not make.startswith("'"):
            make = f"'{make.strip()}'"

        sql = f"SELECT * FROM fabricant WHERE designation={make}" 
        if limit is not None:
            sql += f" LIMIT {limit}"
        sql += ";"
        res = super(BnpDao, self).execute_sql(sql = sql, verbose=verbose)
        if insert_if_not_exist and res is None or len(res) == 0:
            res = self.insert_make(make=make, verbose=verbose)
            res = self.select_make(make=make, verbose=verbose)
        return res
    
    def select_product(self, product_code=None, model=None, pu=None, make=None, category=None, limit=100, verbose=0):
        short_name = "select_product"
        sql_start = "SELECT * FROM catalogue as cat "
        sql_where = ""
        join_key = " WHERE"
        
        if (product_code is not None):
            product_code = f"'{product_code}'" if not isinstance(product_code, str) or not product_code.startswith("'") else product_code
            sql_where += f"{join_key} code_produit = {product_code} "
            join_key = "AND"
        if (model is not None and len(model.strip())>0):
            model = f"'{model.strip()}'" if not model.startswith("'") else model
            sql_where += f"{join_key} modele = {model} "
            join_key = "AND"
        if pu is not None:
            sql_where += f"{join_key} PU = {pu} "
            join_key = "AND"
        if make is not None:
            if isinstance(make, int):
                sql_where += f"{join_key} fabricant = {make} "
                join_key = "AND"
            elif isinstance(make, str):
                make = f"'{make.strip()}'" if not make.startswith("'") else make
                sql_start += ", fabricant as f"
                sql_where += f"{join_key} cat.fabricant = f.id AND f.designation = {make} "
                join_key = "AND"
            else:
                if verbose>0:
                    print(f"[{short_name}]\tWARN : value error on make : '{make}'")

        if category is not None:
            if isinstance(category, int):
                sql_where += f"{join_key} categorie = {category} "
                join_key = "AND"
            elif isinstance(category, str):
                category = f"'{category.strip()}'" if not category.startswith("'") else category
                sql_start += ", categorie as c"
                sql_where += f"{join_key} cat.categorie = c.id_categorie AND c.libelle = {category} "
                join_key = "AND"
            else:
                if verbose>0:
                    print(f"[{short_name}]\tWARN : value error on make : '{category}'")

        sql = sql_start + sql_where
        
        if limit is not None:
            sql += f" LIMIT {limit}"
        sql += ";"
        res = super(BnpDao, self).execute_sql(sql = sql, verbose=verbose)
        return res


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              PUBLIC FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def populate_makes(catalog_file_path, mydao=None, verbose=0):
    short_name = "popuplate_makes"
    if verbose>1:
        print(f"[{short_name}] \tDEBUG :Dataset path : {catalog_file_path}")
    
    catalog_df = load_dump_file(data_set_path=catalog_file_path, verbose=verbose)
    catalog_df = df_sparse_to_dense(catalog_df, verbose=verbose)
    makes = catalog_df.loc[catalog_df["make"].notna(), "make"].unique()
    makes = sorted(makes)

    if mydao is None:
        mydao = BnpDao()

    res = mydao.insert_make(make=makes, verbose=verbose)
    if verbose>0:
        print(f"[{short_name}] \tDEBUG : {res}")
    return res

def populate_categories(catalog_file_path, mydao=None, verbose=0):   
    short_name = "populate_categories"
    if verbose>1:
        print(f"[{short_name}] \tDEBUG :Dataset path : {catalog_file_path}")
    
    catalog_df = load_dump_file(data_set_path=catalog_file_path, verbose=verbose)
    catalog_df = df_sparse_to_dense(catalog_df, verbose=verbose)
    categories = catalog_df.loc[catalog_df["categorie"].notna(), "categorie"].unique()
    categories = sorted(categories)

    if mydao is None:
        mydao = BnpDao()

    res = mydao.insert_category(category=categories, verbose=verbose)
    if verbose>0:
        print(f"[{short_name}] \tDEBUG : {res}")
    return res

def populate_catalogue(catalog_file_path, mydao=None, verbose=0):   
    short_name = "populate_catalogue"
    if verbose>1:
        print(f"[{short_name}] \tDEBUG :Dataset path : {catalog_file_path}")
    
    catalog_df = load_dump_file(data_set_path=catalog_file_path, verbose=verbose)
    catalog_df = df_sparse_to_dense(catalog_df, verbose=verbose)

    product_codes = catalog_df["code_produit"].values
    models       = catalog_df["model"].values
    pus           = catalog_df["PU"].values
    make    = catalog_df["make"].values
    categories    = catalog_df["categorie"].values

    if mydao is None:
        mydao = BnpDao()

    res = mydao.insert_catalog(product_codes=product_codes, models=models, pus=pus, make=make, categories=categories, verbose=verbose)
    if verbose>0:
        print(f"[{short_name}] \tDEBUG : {res}")
    return res

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              PRIVATED FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _initialise_database(populate=True, verbose=0):

    with tqdm(total=1, desc=f"[GenericMySQLDao > initialise_database]", disable=verbose<1) as pbar:
        ma_dao = BnpDao()

        res = ma_dao.tables_list(verbose=verbose)
        assert res is not None and len(res)>0
        if verbose>1: print("liste des tables:",res)
        pbar.update(1)
                
        return ma_dao

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _test(verbose = 1):
    
    with tqdm(total=7, desc=f"[GenericMySQLDao]", disable=verbose<1) as pbar:
        
        ma_dao = _initialise_database(populate=False, verbose=verbose)
        pbar.update(1)
        
        return ma_dao


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %%                                              ARGUMENTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
parser = argparse.ArgumentParser(description='The score DAO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--path',      default=join(execution_path, "dao"), help='Database path')
parser.add_argument('-v', '--verbosity', default='0',   type=int, choices=[0, 1, 2, 3], help='Verbosity level')
parser.add_argument('-pop','--populate', default='False',                               help='Populate the database')
parser.add_argument('-t', '--test',      default='False',                               help='Run tests')
parser.add_argument('-d', '--data_set_path', help='Run tests')
parser.add_argument('-f', '--file_name_start', help='Run tests')

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    args.test = True # TODO à supprimer lorsqu'il ne s'agit plus de tests
    
    verbose                    = args.verbosity      if not args.test       else 1
    populate                   = args.populate       if not args.test       else True
    data_set_path              = args.data_set_path  if not args.test       else join(execution_path, 'dataset')
    file_name_start            = args.file_name_start  if not args.test     else '2023-02-23_19-51'
    
    catalog_file_path = join(data_set_path, file_name_start, "catalog.csv")

    if args.test:
        ma_dao = _test(verbose = verbose)
        if populate: 
            populate_catalogue(catalog_file_path=catalog_file_path, mydao=ma_dao, verbose=verbose)
            # populate_categories(catalog_file_path=catalog_file_path, mydao=ma_dao,verbose=verbose)
            # populate_makes(catalog_file_path=catalog_file_path, mydao=ma_dao, verbose=verbose)
    else:
        ma_dao = _initialise_database(populate=populate, verbose=verbose) 
    
