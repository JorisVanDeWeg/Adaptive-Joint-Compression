import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


def compute_param_count(model):
    if hasattr(model, 'coef_'):
        coef_size = model.coef_.size
        intercept_size = getattr(model, 'intercept_', np.array([])).size
        return coef_size + intercept_size
    elif hasattr(model, 'estimators_'):
        estimators = (
            model.estimators_.ravel()
            if isinstance(model.estimators_, np.ndarray)
            else model.estimators_
        )
        return sum(getattr(est, 'tree_', None).node_count for est in estimators if hasattr(est, 'tree_'))
    elif hasattr(model, 'tree_'):
        return model.tree_.node_count
    elif hasattr(model, 'support_vectors_'):
        return model.support_vectors_.size
    elif isinstance(model, GaussianNB):
        if hasattr(model, 'theta_') and hasattr(model, 'var_'):
            return model.theta_.size + model.var_.size
        return 0
    elif isinstance(model, KNeighborsClassifier):
        if hasattr(model, '_fit_X'):
            return model._fit_X.size
        return 0
    elif isinstance(model, MLPClassifier):
        w_params = sum(w.size for w in model.coefs_)
        b_params = sum(b.size for b in model.intercepts_)
        return w_params + b_params
    else:
        raise ValueError(f"Model type {type(model)} not supported for parameter count estimation.")


def compute_flops(model, X_sample):
    _, n_feats = X_sample.shape
    if isinstance(model, LogisticRegression):
        n_classes = model.coef_.shape[0]
        return 2 * n_feats * n_classes
    if isinstance(model, SVC):
        sv = model.support_vectors_
        return 2 * n_feats * sv.shape[0]
    if isinstance(model, RandomForestClassifier):
        depths = [est.tree_.max_depth for est in model.estimators_]
        return sum(depths)
    if isinstance(model, GradientBoostingClassifier):
        estimators = model.estimators_.ravel()
        depths = [est.tree_.max_depth for est in estimators]
        return sum(depths)
    if isinstance(model, GaussianNB):
        n_classes = model.theta_.shape[0]
        return 4 * n_feats * n_classes
    if isinstance(model, KNeighborsClassifier):
        n_train = model._fit_X.shape[0]
        return 2 * n_feats * n_train
    if isinstance(model, DecisionTreeClassifier):
        return model.tree_.max_depth
    if isinstance(model, MLPClassifier):
        flops = 0
        for W, b in zip(model.coefs_, model.intercepts_):
            flops += 2 * W.size
            flops += b.size
        return flops
    raise ValueError(f"Model type {type(model)} not supported for FLOPs estimation.")


pipelines = {
    'MLPClassifier':     MLPClassifier(),
    'SVC':               SVC(probability=True),
    'RandomForest':      RandomForestClassifier(),
}

param_grids = {
'LogisticRegression': {
    'C': [0.1, 1, 10, 50, 100, 150, 200],
    'solver': ['lbfgs', 'saga'],
    'penalty': ['l2'],
    'multi_class': ['multinomial'],
    'max_iter': [50, 100, 150, 200, 300, 400]
    },

'SVC': {
    'C': [0.1, 1, 10, 100, 200],
    'max_iter': [500, 1000, 2000],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
    },

'RandomForest': {
    'n_estimators': [10, 50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample', None]
    },

'GradientBoosting': {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10]
    },

'GaussianNB': {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    },

'KNeighbors': {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [30, 50, 100],
    },

'DecisionTreeClassifier': {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
    },

'MLPClassifier': {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [200, 300, 400]
    },
}


def estimate_random_forest_flops(model, input_dim):
    total_flops = 0
    n_trees = len(model.estimators_)
    for tree in model.estimators_:
        n_nodes = tree.tree_.node_count
        avg_depth = np.log2(n_nodes)
        flops_per_tree = avg_depth
        total_flops += flops_per_tree
    return int(total_flops)


def estimate_mlp_flops(model):
    flops = 0
    for i in range(len(model.coefs_)):
        n_input = model.coefs_[i].shape[0]
        n_output = model.coefs_[i].shape[1]
        flops += 2 * n_input * n_output
        flops += n_output
    return int(flops)


def estimate_logistic_regression_flops(model):
    n_features = model.coef_.shape[1]
    n_classes = model.coef_.shape[0] if len(model.coef_.shape) > 1 else 1
    flops = 2 * n_features * n_classes
    flops += n_classes
    return int(flops)


def estimate_svc_flops(model):
    if not hasattr(model, 'support_vectors_'):
        return 0
    n_support_vectors = model.support_vectors_.shape[0]
    n_features = model.support_vectors_.shape[1]
    n_classes = len(model.classes_) if hasattr(model, 'classes_') else 2
    flops = 3 * n_support_vectors * n_features
    flops += n_support_vectors
    if model.probability:
        flops += 2 * n_classes
    return int(flops)
