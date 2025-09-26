# scripts/run_experiments.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pickle
import time
import joblib
import sklearn
import tensorflow as tf

from keras import layers, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.data import AUTOTUNE

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix)

from hybrid.data import extract_window_features, compute_features
from hybrid.models import *
from hybrid.shallow import *
from hybrid.training import *
from hybrid.utils import *

file_name = 'models_dic' + '_1'

print("Starting experiment...")

for Dataset in ['PU']:
    classes_type = 'multiclass_extended'
    size_timeseries = 2
    number_of_windows = 20

    if Dataset == 'PU':
        size_timeseries = 4
        pu_time_series_df = pd.read_parquet(f'Dataset/pu_time_series_df_{size_timeseries}sec_{classes_type}.parquet', engine='pyarrow')
        X = np.array(pu_time_series_df['6_ts'].tolist())
        y = pu_time_series_df['label'].values
    elif Dataset == 'CWRU':
        size_timeseries = 2
        cwru_time_series_df = pd.read_parquet(f'Dataset/cwru_time_series_df_{size_timeseries}sec_{classes_type}.parquet', engine='pyarrow')
        X = np.array(cwru_time_series_df['time_series'].tolist())
        y = cwru_time_series_df['label'].values
    elif Dataset == 'JKU':
        raise ValueError("JKU dataset not available yet")

    window_size = int(X.shape[1]/number_of_windows)

    X_features = extract_window_features(X, window_size)
    X_features = np.array(X_features)

    folds = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X_features, y):
        X_train, X_test = X_features[train_index], X_features[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        X_test, y_test = sklearn.utils.shuffle(X_test, y_test) 
        
        scaler = StandardScaler()
        n_samples, n_windows, n_features = X_train.shape
        X_train = scaler.fit_transform(X_train.reshape(-1, n_features))
        X_train = X_train.reshape(n_samples, n_windows, n_features)

        n_samples, n_windows, n_features = X_test.shape
        X_test = scaler.transform(X_test.reshape(-1, n_features))
        X_test = X_test.reshape(n_samples, n_windows, n_features)

        folds.append((X_train, X_test, y_train, y_test))

    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = len(np.unique(y))

    dataset_folds = []
    for X_train, X_test, y_train, y_test in folds:
        full_ds = (
            tf.data.Dataset
            .from_tensor_slices((X_train, y_train))
            .shuffle(buffer_size=len(X_train))
        )
        val_size = int(0.2 * len(X_train))
        train_size = len(X_train) - val_size

        val_ds = (
            full_ds
            .take(val_size)
            .batch(32)
            .cache()
            .prefetch(AUTOTUNE)
        )
        train_ds = (
            full_ds
            .skip(val_size)
            .batch(32)
            .cache()
            .prefetch(AUTOTUNE)
        )
        test_ds = (
            tf.data.Dataset
            .from_tensor_slices((X_test, y_test))
            .batch(32)
            .cache()
            .prefetch(AUTOTUNE)
        )

        dataset_folds.append((train_ds, val_ds, test_ds))

    models_dic = {}

    for numb_windows in [1]:
        window_size = int(X.shape[1]/numb_windows)

        X_features = extract_window_features(X, window_size)
        X_features = np.array(X_features)

        max_position = X_features.shape[1]
        num_classes = len(np.unique(y))

        folds2 = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X_features, y):
            X_train, X_test = X_features[train_index], X_features[test_index]
            y_train, y_test = y[train_index], y[test_index]

            X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
            X_test, y_test = sklearn.utils.shuffle(X_test, y_test) 
            
            scaler = StandardScaler()
            n_samples, n_windows, n_features = X_train.shape
            X_train = scaler.fit_transform(X_train.reshape(-1, n_features))
            X_train = X_train.reshape(n_samples, n_windows, n_features)

            n_samples, n_windows, n_features = X_test.shape
            X_test = scaler.transform(X_test.reshape(-1, n_features))
            X_test = X_test.reshape(n_samples, n_windows, n_features)

            folds2.append((X_train, X_test, y_train, y_test))

    pipelines = {
        'SVC':               SVC(probability=True),
        'RandomForest':      RandomForestClassifier(),
    }

    param_grids = {
        'SVC': {
            'C': [0.1, 1, 10, 100, 200],
            'max_iter': [500, 1000, 2000], 
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        },
        'RandomForest': {
            'n_estimators': [10, 50, 75, 100, 125, 150],
            'max_depth': [None, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 4, 6, 8, 10],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'MLPClassifier': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [200, 300, 400]
        },
    }

    print_status("Starting shallow models training and evaluation...")
    for name, estimator in pipelines.items():
        grid = GridSearchCV(estimator, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        print(f"Training {name}")
        for fold_idx, (X_tr, X_te, y_tr, y_te) in enumerate(folds2, 1):
            ns, nw, nf = X_tr.shape
            Xtr = X_tr.reshape(ns, nw * nf)
            Xte = X_te.reshape(X_te.shape[0], nw * nf)

            grid.fit(Xtr, y_tr)
            best = grid.best_estimator_

            t0 = time.time()
            y_tr_hat = best.predict(Xtr)
            y_te_hat = best.predict(Xte)
            t_pred = time.time() - t0

            metrics = {
                'train_accuracy':  accuracy_score(y_tr, y_tr_hat),
                'test_accuracy' :  accuracy_score(y_te, y_te_hat),
                'train_precision': precision_score(y_tr, y_tr_hat, average='macro'),
                'test_precision' : precision_score(y_te, y_te_hat, average='macro'),
                'train_recall':    recall_score(y_tr, y_tr_hat, average='macro'),
                'test_recall':     recall_score(y_te, y_te_hat, average='macro'),
                'train_f1':        f1_score(y_tr, y_tr_hat, average='macro'),
                'test_f1':         f1_score(y_te, y_te_hat, average='macro'),
            }

            pcount = compute_param_count(best)
            if isinstance(best, RandomForestClassifier):
                flops = estimate_random_forest_flops(best, Xte.shape[1])
            elif isinstance(best, MLPClassifier):
                flops = estimate_mlp_flops(best)
            elif isinstance(best, SVC):
                flops = estimate_svc_flops(best)
            else:
                raise ValueError(f"Model type {type(best)} not supported for FLOPs estimation.")  

            joblib.dump(best, 'shallow.joblib')
            model_size = os.path.getsize('shallow.joblib') / 1024
                
            key = f"{name}_{fold_idx}"
            models_dic[key] = {
                'best_params': grid.best_params_,
                **metrics,
                'param_count': pcount,
                'flops': flops,
                'pred_time_s': t_pred,
                'model_size_bytes': model_size,
                'model_size_params': compute_parameters_size(None, pcount),
                'param_count (eff)': pcount,
                'dataset': Dataset
            }

    print_status("Shallow models trained and evaluated.")

    del folds2
    del X

    base_config = {
        "epochs": 300,
        "batch_size": 32,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "initial_learning_rate": 0.001,
        "decay_rate": 0.97,
        "epochs_per_decay": 25,
        "model_name_prefix": "Base",
        "plot_bool": False,
        "verbose": 0,
        "breaker_fold": 5,
    }

    if Dataset == 'PU':
        base_config['model'] = lambda: cnn_attention(
            input_shape=input_shape,
            num_classes=num_classes,
            cnn_filters=[32],
            transformer_dim=64,
            num_heads=8,
            ff_dim=64,
            num_transformer_blocks=1,
            dropout_rate=0.1)
    elif Dataset == 'CWRU':
        base_config['model'] = lambda: cnn_attention(
            input_shape=input_shape,
            num_classes=num_classes,
            cnn_filters=[16],
            transformer_dim=32,
            num_heads=4,
            ff_dim=32,
            num_transformer_blocks=1,
            dropout_rate=0.1)

    print_status("Starting base model experiment...")
    results_base = run_base_experiment(base_config, folds, dataset_folds)
    models_dic.update(results_base)
    print_status("Base model trained and evaluated.")

    compact_config = {
        "epochs": 300,
        "batch_size": 32,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "initial_learning_rate": 0.001,
        "decay_rate": 0.97,
        "epochs_per_decay": 25,
        "model_name_prefix": "Compact",
        "plot_bool": False,
        "verbose": 0,
        "breaker_fold": 5,
    }

    if Dataset == 'PU':
        compact_config['model'] = lambda: cnn_attention(
            input_shape=input_shape,
            num_classes=num_classes,
            cnn_filters=[32],
            transformer_dim=16,
            num_heads=8,
            ff_dim=16,
            num_transformer_blocks=1,
            dropout_rate=0.1)
    elif Dataset == 'CWRU':
        compact_config['model'] = lambda: cnn_attention(
            input_shape=input_shape, 
            num_classes=num_classes,
            cnn_filters=[8],
            transformer_dim=16,
            num_heads=4,
            ff_dim=16,
            num_transformer_blocks=1,
            dropout_rate=0.1)

    print_status("Starting compact model experiment...")
    results_compact = run_base_experiment(compact_config, folds, dataset_folds)
    models_dic.update(results_compact)
    print_status("Compact model trained and evaluated.")

    svd_config = {
        "epochs": 500,
        "batch_size": 32,
        "input_shape": input_shape,
        "initial_learning_rate": 0.001,
        "decay_rate": 0.98,
        "epochs_per_decay": 25,
        "model_name_prefix": "Hybrid",
        "plot_bool": False,
        "verbose": 0,
        "save_plots": False,
        "output_folder": "Results/TrainingPlots",
        "breaker_fold": 1,
        "lambda_d_values": [0.01, 0.1, 1.0, 5.0],
        "lambda_q_values": [0.01, 0.1, 1, 5],
        "decomposition": "SVD",
        "quan_gates_init_modes": ["on"],
        "decomp_gates_init_modes":  ["random"],
        "Dataset": Dataset,
    }
    if Dataset == 'PU':
        svd_config['model'] = lambda: cnn_attention_hybrid(
            input_shape, num_classes,
            cnn_filters=[32],
            transformer_dim=64,
            num_heads=8,
            ff_dim=64,
            num_transformer_blocks=1,
            dropout_rate=0.1)
    elif Dataset == 'CWRU':
        svd_config['initial_learning_rate'] = 0.004
        svd_config['model'] = lambda: cnn_attention_hybrid(
            input_shape, num_classes, 
            cnn_filters=[16],
            transformer_dim=32,
            num_heads=4,
            ff_dim=32,
            num_transformer_blocks=1,
            dropout_rate=0.1)

    print_status("Starting hybrid SVD experiment...")
    results_svd = run_hybrid_experiment(svd_config, folds, dataset_folds)
    models_dic.update(results_svd)
    print_status("Hybrid SVD model trained and evaluated.") 

    print_status("Starting hybrid Tucker experiment...")
    svd_config['decomposition'] = "Tucker"
    svd_config['model_name_prefix'] = "Hybrid Tucker"
    results_tucker = run_hybrid_experiment(svd_config, folds, dataset_folds)
    models_dic.update(results_tucker)
    print_status("Hybrid Tucker model trained and evaluated.")

    RESULTS_PATH = f"{file_name}.pkl"
    merge_and_save(models_dic, RESULTS_PATH)

    svd_config = {
        "epochs": 500,
        "batch_size": 32,
        "input_shape": input_shape,
        "initial_learning_rate": 0.001,
        "decay_rate": 0.98,
        "epochs_per_decay": 25,
        "model_name_prefix": "Compact Hybrid",
        "plot_bool": False,
        "verbose": 0,
        "save_plots": False,
        "output_folder": "Results/TrainingPlots",
        "breaker_fold": 1,
        "lambda_d_values": [0.01, 0.1, 1.0, 5.0],
        "lambda_q_values": [0.01, 0.1, 1],
        "decomposition": "SVD",
        "quan_gates_init_modes": ["on"],
        "decomp_gates_init_modes":  ["random"],
        "Dataset": Dataset,
    }
    if Dataset == 'PU':
        svd_config['model'] = lambda: cnn_attention_hybrid(
            input_shape=input_shape,
            num_classes=num_classes,
            cnn_filters=[32],
            transformer_dim=16,
            num_heads=8,
            ff_dim=16,
            num_transformer_blocks=1,
            dropout_rate=0.1)
    elif Dataset == 'CWRU':
        svd_config['model'] = lambda: cnn_attention_hybrid(
            input_shape=input_shape, 
            num_classes=num_classes,
            cnn_filters=[8],
            transformer_dim=16,
            num_heads=4,
            ff_dim=16,
            num_transformer_blocks=1,
            dropout_rate=0.1)

    print_status("Starting compact hybrid SVD experiment...")
    results_svd = run_hybrid_experiment(svd_config, folds, dataset_folds)
    models_dic.update(results_svd)
    print_status("Hybrid SVD model trained and evaluated.") 

    print_status("Starting hybrid Tucker experiment...")
    svd_config['decomposition'] = "Tucker"
    svd_config['model_name_prefix'] = "Compact Hybrid Tucker"
    results_tucker = run_hybrid_experiment(svd_config, folds, dataset_folds)
    models_dic.update(results_tucker)
    print_status("Hybrid Tucker model trained and evaluated.")

    RESULTS_PATH = f"{file_name}.pkl"
    merge_and_save(models_dic, RESULTS_PATH)

    print_status(f"Done! All models trained and evaluated. Results saved to '{file_name}.pkl'.")

if __name__ == "__main__":
    pass
