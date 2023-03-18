import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score


def evaluate(X_test, y_test, y_pred, verbose=0):
    y_t_c = y_test
    if len(y_test.shape)==1 or y_test.shape[1]==1:
        y_t_c = complete_y_cols(X_test=X_test, y_param=y_test)
    try:
        y_t_c = y_t_c.drop('index', axis=1)
    except:
        pass

    y_p_c = y_pred
    if len(y_pred.shape)==1:
        y_p_c = complete_y_cols(X_test=X_test, y_param=y_pred)

    return pr_auc_score(y_true=y_t_c, y_pred_proba=y_p_c), y_t_c, y_p_c


def pr_auc_score(y_true, y_pred_proba):
    ''' 
    Return the area under the Precision-Recall curve.
    
    Args:
        - y_true (pd.DataFrame): Dataframe with a unique identifier for each observation (first column) and the ground truth observations (second column).
        - y_pred_proba (pd.DataFrame): Dataframe with a unique identifier for each observation (first column) and the predicted probabilities estimates for the minority class (second column).
        
    Returns:
        float
    '''
    
    y_true_sorted = y_true.sort_values(by='ID').reset_index(drop=True)
    y_pred_proba_sorted = y_pred_proba.sort_values(by='ID').reset_index(drop=True)
    pr_auc_score = average_precision_score(np.ravel(y_true_sorted.iloc[:, 1]), np.ravel(y_pred_proba_sorted.iloc[:, 1]))

    return pr_auc_score


def complete_y_cols(X_test, y_param):
    new_y = None
    if 'ID' in X_test.columns:
        new_y = X_test[['ID']].copy()
    else:
        new_y = X_test[['cart_ID']].copy()
        new_y = new_y.rename(columns={'cart_ID':'ID'})

    new_y = new_y.reset_index()
    new_y = new_y.set_index('index')
    new_y['fraud_flag'] = y_param
    return new_y


# The following lines show how the csv files are read
if __name__ == '__main__':
    import pandas as pd
    y_true_path = 'Y_test.csv'  # path of the y_true csv file
    y_pred_proba_path = 'Y_test_benchmark.csv'  # path of the y_pred csv file
    y_true = pd.read_csv(y_true_path)
    y_pred_proba = pd.read_csv(y_pred_proba_path)
    print(pr_auc_score(y_true, y_pred_proba))





