import numpy as np
from scipy import stats
from sklearn import metrics

def accuracy_fn(y_true, y_pred):
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred).astype(int)
        y_true = np.asarray(y_true).astype(int)
    return np.mean(y_true == y_pred)

def pearsonr_fn(y_true, y_pred):
    r, pval = stats.pearsonr(y_true, y_pred)
    return r

def rmse_fn(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def r2_fn(y_true, y_pred):
    return metrics.r2_score(y_true, y_pred)

def mae_fn(y_true, y_pred):
    return metrics.mean_absolute_error(y_true, y_pred)

def spearmanr_fn(y_true, y_pred):
    return stats.spearmanr(y_true, y_pred)[0]


METRICS = {'accuracy': (accuracy_fn, 0.0, np.greater),
           'pearsonr': (pearsonr_fn, 0.0, np.greater),
           'loss': (None, np.inf, np.less),
           'rmse': (rmse_fn, np.inf, np.less),
           'mae': (mae_fn, np.inf, np.less),
           'r2': (r2_fn, 0.0, np.greater),
           'spearmanr': (spearmanr_fn, 0.0, np.greater)}

DATASET_METRICS = {'pdbbind': ['rmse', 'pearsonr', 'mae', 'r2', 'spearmanr'],
                   'reacbio': ['rmse', 'pearsonr', 'mae', 'r2', 'spearmanr'],
                   'scope': 'accuracy',
                   'enzyme': 'accuracy'}

EVAL_METRICS = {'pdbbind': {'patch': 'loss', 'backbone': 'rmse', 'surface': 'rmse', 'backbone2surface': 'rmse',
                            'surface2backbone': 'rmse', 'backbone2patch': 'rmse', 'patch2backbone': 'rmse'},
                'reacbio': {'patch': 'loss', 'backbone': 'rmse', 'surface': 'rmse', 'backbone2surface': 'rmse',
                            'surface2backbone': 'rmse', 'backbone2patch': 'rmse', 'patch2backbone': 'rmse'},
                'scope': {'patch': 'loss', 'backbone': 'accuracy', 'surface': 'accuracy',
                          'backbone2surface': 'accuracy', 'surface2backbone': 'accuracy', 'patch2backbone': 'accuracy'},
                'enzyme': {'patch': 'loss', 'backbone': 'accuracy', 'surface': 'accuracy',
                           'backbone2surface': 'accuracy', 'backbone2patch': 'accuracy',
                           'surface2backbone': 'accuracy', 'patch2backbone': 'accuracy'}}
