#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Module to explore the database cojoden

Project: BNP
=======

Usage:
======
    python cart.py

"""
__authors__     = ("Aurélie RAOUL")
__contact__     = ("aurelie.raoul@yahoo.fr")
__copyright__   = "MIT"
__date__        = "2023-03-18"
__version__     = "1.0.0"

from tqdm import tqdm

from os import getcwd
from os.path import join, exists
import numpy as np
from random import randint
import pandas as pd

# Récupère le répertoire du programme
execution_path = getcwd()
project_name = "projet_bnp_paribas_fraudeurs_app"
# Permet de gérer le cas d'une exécution dans un notebook par exemple
execution_path = execution_path.split("PROJETS")[0]
execution_path = join(execution_path, "PROJETS", project_name)
import sys
sys.path.append(execution_path)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                               CLASSES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Product():

    def __init__(self, item_category, cash_price, make, model,code, nb):
        self.item = item_category
        self.cash_price = cash_price
        self.make = make
        self.model = model
        self.code = code
        self.nb = nb

    def __str__(self):
        return f"{self.nb} x {self.item} = {self.cash_price} $, {self.make}, {self.model}, {self.code}"

    def __repr__(self) -> str:
        return self.__str__()


class Cart():

    def __init__(self, id=None, products=[], fraud=-1):
        if id is None:
            id = randint(a=10000, b=100000000)
        self.id = id
        self.products = products
        self.fraud = fraud

    def __str__(self):
        p_str = "\n".join([str(prod) for prod in self.products])
        return f"{self.id}, {self.fraud} :\n{p_str}"

    def __repr__(self) -> str:
        return self.__str__()

    def add_product(self, product):
        if product is not None:
            if self.products is None:
                self.products = []
            self.products.append(product)

    def to_df(self):
        datas = self.to_dict()

        for k, v in datas.items():
            datas[k] = [v]

        df = pd.DataFrame.from_dict(datas)
        return df

    def to_dict(self, cart_row=24):
        # ID,item1,item2,cash_price1,cash_price2,make1,make2,model1,model2,goods_code1,goods_code2,Nbr_of_prod_purchas1,Nbr_of_prod_purchas2,Nb_of_items
        cart_dict = {
            "ID" : self.id,
        }

        for i in range(1, cart_row+1):
            pi = i-1
            if pi < len(self.products):
                prod = self.products[pi]
                cart_dict.update({ 
                        f'item{i}': prod.item, 
                        f'cash_price{i}': prod.cash_price, 
                        f'make{i}': prod.make, 
                        f'model{i}': prod.model, 
                        f'goods_code{i}': prod.code, 
                        f'Nbr_of_prod_purchas{i}': prod.nb, 
                    })
            else:
                cart_dict.update({ 
                        f'item{i}': np.nan,
                        f'cash_price{i}': np.nan,
                        f'make{i}': np.nan, 
                        f'model{i}': np.nan,
                        f'goods_code{i}': np.nan,
                        f'Nbr_of_prod_purchas{i}': np.nan,
                    })
        cart_dict.update({ 
                    'Nb_of_items': len(self.products),
                })
        return cart_dict
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                                              TESTS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == "__main__":

    cart = Cart()
    prod1 = Product(item_category='TARGUS GEOLITE ESSENTIAL CASE',
                    cash_price='100', make='TARGUS',
                    nb=1, model='TARGUS GEOLITE ESSENTIAL CASE', code='99900041'
                    )
    cart.add_product(prod1)

    prod2 = Product(item_category='APPLE S',
                    cash_price='1000', make='APPLE SMART',
                    nb=1, model='APPLE SMART', code='99900044'
                    )
    cart.add_product(prod2)

    prod3 = Product(item_category='COMPUTERS',
                    cash_price='110', make='KOBO',
                    nb=1, model='KOBO CLARA HD EREADER 6 ILLUMINATED TOUCH SCREEN W',
                    code='237534446'
                    )
    cart.add_product(prod3)
    print(cart)

    print(cart.to_df())
                          
                          
                          
                          