# -*- coding: utf-8 -*-
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
from cart import *

# flask --app app_flask run
# flask --app app_flask run --debug

from api_commons import do_prediction
from cart import *
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def form():
    return render_template('home.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():

    record = request.form
    print("------------------------------------------------")
    print(record)
    cart = Cart()
    
    for i in range(1, 25):
        code    = record.get(f"code{i}", None)
        model   = record.get(f"model{i}", None)
        item    = record.get(f"item{i}", None)
        nb      = record.get(f"nb{i}", None)
        cash    = record.get(f"cash{i}", None)
        make    = record.get(f"make{i}", None)
        if nb is not None:
            nb = int(nb)
            if nb > 0:
                prod = Product(item_category=item, cash_price=cash, make=make, model=model,code=code, nb=nb)
                cart.add_product(product=prod)
            else:
                # On quitte la boucle dès que nous sommes au dernier produit
                break
        else:
            # On quitte la boucle dès que nous sommes au dernier produit
            break

    print("------------------------------------------------")
    print(str(cart))
    print("------------------------------------------------")

    predict = do_prediction(cart=cart)
    print(predict)

    return render_template('greeting.html', say=predict)

if __name__ == "__main__":
    app.run()