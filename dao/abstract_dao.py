#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" To manage MySQL common functions
Project: GenericMySQLDao
=======

Usage:
======
    configuration file : local_mysql.env
"""
__authors__     = ("Aurélie RAOUL")
__contact__     = ("aurelie.raoul@yahoo.fr")
__copyright__   = "MIT"
__date__        = "2023-03-11"
__version__     = "1.0.0"

import mysql.connector
from mysql.connector import errorcode
from dotenv import dotenv_values
import pandas as pd
import sqlalchemy as sa
from os import getcwd, remove
from tqdm import tqdm
import argparse

from os.path import join, exists
# Récupère le répertoire du programme
execution_path = getcwd()
project_name = "projet_bnp_paribas_fraudeurs"

# Permet de gérer le cas d'une exécution dans un notebook par exemple
execution_path = execution_path.split("PROJETS")[0]
execution_path = join(execution_path, "PROJETS", project_name)
import sys
sys.path.append(execution_path)

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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                               CLASSES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class GenericMySQLDao:
    """Traite tout ce qui concerne la base de données """

    def __init__(self, tables_names, host=global_db_host, db_user=global_db_user, db_pwd=global_db_pwd, db_name=global_db_name):
        """Constructeur

        Args:
            tables_names (list): table name in creation order
            host (str, optional): Defaults local_mysql.env file data.
            db_user (str, optional): Defaults local_mysql.env file data.
            db_pwd (str, optional): Defaults local_mysql.env file data.
            db_name (str, optional): Defaults local_mysql.env file data.
        """
        self.tables_names = tables_names
        self.db_client = "mysql"
        self.db_host = host
        self.db_user = db_user
        self.db_pwd = db_pwd
        self.db_name = db_name
        
    # ----------------------------------------------------------------------------------
    #                         DATABASE STATUS
    # ----------------------------------------------------------------------------------
    def tables_list(self, verbose=0):
        short_name = "tables_list"
        res = None
        try:
            res = self.execute_sql(sql=f"SHOW TABLES FROM {self.db_name};", verbose=verbose)
            return res
        except Exception as error:
            err_str = str(error)
            if "NoneType" not in err_str and 'cursor' not in err_str:
                print(f"[{short_name}] ERROR : {err_str}")
                raise error
        return res
    
    def database_missing_tables(self, verbose=0):
        short_name = "database_missing_tables"
        missing_table = self.tables_names.copy()
        try:
            res = self.tables_list(verbose=verbose)
            
            for row in res:
                table_name = row[0]
                if table_name in missing_table:
                    missing_table.remove(table_name)
        except Exception as error:
            err_str = str(error)
            if "NoneType" not in err_str and 'cursor' not in err_str:
                print(f"[{short_name}] ERROR : {err_str}")
                raise error

        return missing_table

    def database_exist(self, verbose=0):
        return len(self.database_missing_tables(verbose=verbose)) == 0
        
    # ----------------------------------------------------------------------------------
    #                         DATABASE RESET
    # ----------------------------------------------------------------------------------
    def reset_database(self, verbose=0):
        try:
            # Prise en compte du cas où la BDD n'existe pas
            self.drop_database(verbose=verbose)
        except:
            pass
        self.initialize_data_base(verbose=verbose)

    def drop_database(self, verbose=0):
        sql = f"drop database if exists {self.db_name}; "
        self.execute_sql(sql=sql, verbose=verbose)

    # ----------------------------------------------------------------------------------
    # %%                        DATABASE CONNECTION
    # ----------------------------------------------------------------------------------
    def data_base_connection(self):
        """Create the database connection and return it.

        Returns:
            connection
        """
        short_name = "data_base_connection"
        connection = None
        try:
            connection = mysql.connector.connect(
                user=self.db_user,
                password=self.db_pwd,
                host=self.db_host,
                database=self.db_name)

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print(f"[{short_name}] \tERROR : Something is wrong with your user name or password : {err}")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print(f"[{short_name}] \tERROR : Database does not exist : {err}")
            else:
                print(f"[{short_name}] \tERROR : {err}")

        return connection 

    # ----------------------------------------------------------------------------------
    #      RECUPERATION DES TABLES SOUS FORME DE DF
    # ----------------------------------------------------------------------------------
    def get_table_df(self, table_name, dataset_path, file_name=None, write_file=True, verbose=0):
        """Read the table, create a DataFrame with it and write the DF CSV file

        Args:
            table_name (str): table name
            dataset_path (str): the path to write the CSV file for the table
            file_name (str, optional): The CSV file name. Defaults to None => 'cojoden_bdd_{table_name}.csv'
            write_file (bool, optional): To write the CSV file or not. Defaults to True.
            verbose (int, optional): Log level. Defaults to 0.

        Returns:
            DataFrame: The table DataFrame
        """
        short_name = "get_table_df"
        
        dbConnection =self.create_engine(verbose=verbose)

        df = pd.read_sql(sql=f'SELECT * FROM {table_name}', con=dbConnection)

        if df is not None and write_file:
            if file_name is None:
                file_name=f'bdd_{table_name}.csv'
            file_path = join(dataset_path, file_name)
            df.to_csv(file_path, index=False)
            if verbose>0:
                print(f'[{short_name}] File write ---> {file_path}')
        return df

    # ----------------------------------------------------------------------------------
    #                         DATABASE REQUEST
    # ----------------------------------------------------------------------------------
    def execute_sql(self, sql, verbose=0):
        """Execute SQL query

        Args:
            sql (str): _description_
            verbose (int, optional): Log level. Defaults to 0.

        Raises:
            error: _description_
            error: _description_

        Returns:
            _type_: _description_
        """
        short_name = "execute_sql"
        if sql is None or len(sql.strip())==0:
            raise AttributeError("Query is missing")
        
        conn = None
        cur = None
        # Séparation des try / except pour différencier les erreurs
        try:
            conn = self.data_base_connection()
            cur = conn.cursor()
            if verbose > 1:
                print(f"[{short_name}] DEBUG : connexion à la BDD")
            try:
                sql = sql.replace("''", "'")
                if verbose > 1 :
                    print(f"[{short_name}] DEBUG : \n{sql}")
                cur.execute(sql)
                if "INSERT" in sql or "UPDATE" in sql or "CREATE" in sql or "DROP" in sql:
                    conn.commit()

                if "INSERT" in sql:
                    res = cur.lastrowid
                else:
                    res = cur.fetchall()
                if verbose>1:
                    print(f"[{short_name}] DEBUG : {res}")

            except Exception as error:
                print(f"[{short_name}] ERROR : Erreur exécution SQL :")
                print(f"[{short_name}] ERROR :\t- {error}")
                print(f"[{short_name}] ERROR :\t- {sql}")
                raise error
        except Exception as error:
            print(f"[{short_name}] ERROR : Erreur de connexion à la BDD :")
            print(f"[{short_name}] ERROR :\t- {error}")
            print(f"[{short_name}] ERROR :\t- {sql}")
            raise error
        finally:
            try:
                if verbose > 1:
                    print(f"[{short_name}] DEBUG : Le curseur est fermé")
                cur.close()
            except Exception:
                pass
            try:
                if verbose > 1:
                    print(f"[{short_name}] DEBUG : La connexion est fermée")
                conn.close()
            except Exception:
                pass       
        return res

    # %% create_engine
    def create_engine(self, verbose=0):
        short_name = "create_sql_url"
        # connect_args={'ssl':{'fake_flag_to_enable_tls': True}, 'port': 3306}
        connection_url = sa.engine.URL.create(
            drivername=self.db_client,
            username=self.db_user,
            password=self.db_pwd,
            host=self.db_host,
            database=self.db_name
        )
        if verbose > 1:
            print(f"[{short_name}] DEBUG : {connection_url}")
        db_connection = sa.create_engine(connection_url, pool_recycle=3600) # ,connect_args= connect_args)
        return db_connection


    # ----------------------------------------------------------------------------------
    #                         DATABASE INITIALISATION
    # ----------------------------------------------------------------------------------
    def initialize_data_base(self, reset_if_exist=False, verbose=0):
        """Create the database with the SQL creation script.

        Args:
            reset_if_exist (boolean, optional): True to drop the database before creation, Default = False.
            verbose (int, optionnal) : Log level, Default = 0

        Returns:
            (connection, cursor): The database connection and the cursor
        """
        short_name = "initialize_data_base"
        if reset_if_exist:
            self.drop_database(verbose=verbose)
            if verbose > 0:
                print(f"[{short_name}] INFO : Database ----- DROP")
        
        if not self.database_exist():
            connection = None
            cursor = None
            request = ""
            try:
                connection = mysql.connector.connect(
                    user=self.db_user,
                    password=self.db_pwd,
                    host=self.db_host)
                cursor = connection.cursor()
                request = f'create database {self.db_name};'
                if verbose > 1:
                    print(f"[{short_name}] DEBUG : \n{request}")
                res = cursor.execute(request)
                if verbose > 1:
                    print(f"[{short_name}] DEBUG : res : {res}")
                connection.commit()
                request = f'use {self.db_name};'
                if verbose > 1:
                    print(f"[{short_name}] DEBUG : \n{request}")
                res = cursor.execute(request)
                if verbose > 1:
                    print(f"[{short_name}] DEBUG : res : {res}")
            except Exception as msg:
                print(f"[{short_name}] \tERROR : \n\t- {request} \n\t- {msg}")
            finally:
                try:
                    if verbose > 1:
                        print(f"[{short_name}] DEBUG : Le curseur est fermé")
                    cursor.close()
                except Exception:
                    pass
                try:
                    if verbose > 1:
                        print(f"[{short_name}] DEBUG : La connexion est fermée")
                    connection.close()
                except Exception:
                    pass
        # Appel dynamique des fonctions de création des tables
        missing_table = self.database_missing_tables(verbose=verbose)
        for table in missing_table:
            globals()["create_table_"+table]( verbose=verbose)
        if verbose > 0:
            print(f"[{short_name}] INFO : Tables {missing_table} ----- CREATED")
            
    def initialize_data_base_via_script(self, script_path, reset_if_exist=False, verbose=0):
        """Create the database with the SQL creation script.

        Args:
            script_path (str, optional): the SQL creation script. Defaults to 'dataset/cojoden_avance_creation_script.sql'.
            reset_if_exist (boolean, optional): True to drop the database before creation, Default = False.
            verbose (int, optionnal) : Log level, Default = 0

        Returns:
            (connection, cursor): The database connection and the cursor
        """
        short_name = "initialize_data_base"
        if reset_if_exist:
            self.reset_database()
        if not self.database_exist():
            connection = None
            cursor = None
            try:
                connection = mysql.connector.connect(
                    user=self.db_user,
                    password=self.db_pwd,
                    host=self.db_host)
                cursor = connection.cursor()
                with open(script_path, 'r') as sql_file:
                    lines = sql_file.readlines()
                    request = ""
                    for line in lines:
                        if not line.startswith("--"):
                            request += line.strip()
                            if request.endswith(";"):
                                try:
                                    res = cursor.execute(request)
                                    connection.commit()
                                except Exception as msg:
                                    print(f"[{short_name}] \tERROR : \n\t- {line} \n\t- {msg}")
                                request = ""
            finally:
                try:
                    if verbose > 1:
                        print(f"[{short_name}] DEBUG : Le curseur est fermé")
                    cursor.close()
                except Exception:
                    pass
                try:
                    if verbose > 1:
                        print(f"[{short_name}] DEBUG : La connexion est fermée")
                    connection.close()
                except Exception:
                    pass
        elif verbose>0:
            print(f"[{short_name}] \tINFO : the database ever exist")

        return connection, cursor


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              PRIVATED FUNCTIONS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def _initialise_database(populate=True, verbose=0):

    with tqdm(total=1, desc=f"[GenericMySQLDao > initialise_database]", disable=verbose<1) as pbar:
        tables_names = ["categorie", "fabricant", "catalogue"]
        ma_dao = GenericMySQLDao(tables_names=tables_names)

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

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    args.test = True # TODO à supprimer lorsqu'il ne s'agit plus de tests
    
    verbose                    = args.verbosity      if not args.test       else 1
    populate                   = args.populate       if not args.test       else True
    
    if args.test:
        _test(verbose = verbose)
    else:
        ma_dao = _initialise_database(populate=populate, verbose=verbose) 
