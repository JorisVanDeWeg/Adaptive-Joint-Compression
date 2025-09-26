import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score

from hybrid.utils.metrics import flops_measure, compute_parameters_size, get_tflite_model_size


def run_base_experiment(config, folds, dataset_folds):
    results = {}

    input_shape = config["input_shape"]
    num_classes = config["num_classes"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    verbose = config["verbose"]
    breaker_fold = config["breaker_fold"]
    model_name_prefix = config["model_name_prefix"]

    initial_lr = config["initial_learning_rate"]
    decay_rate = config["decay_rate"]
    epochs_per_decay = config["epochs_per_decay"]

    for i, ((train_ds, val_ds, test_ds), (X_train, X_test, y_train, y_test)) in enumerate(zip(dataset_folds, folds)):
        print(f"\nRunning fold {i + 1} out of {len(folds)}")

        steps_per_epoch = len(X_train) // batch_size
        decay_steps = epochs_per_decay * steps_per_epoch

        model = config['model']()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True
        )
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            jit_compile=True,
            run_eagerly=False
        )

        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                            batch_size=batch_size, verbose=verbose)

        train_loss, train_accuracy = model.evaluate(train_ds, verbose=0)
        test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)

        t0 = time.time()
        y_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        t_pred = time.time() - t0

        y_train_prob = model.predict(X_train, verbose=0)
        y_train_pred = np.argmax(y_train_prob, axis=1)

        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': precision_score(y_train, y_train_pred, average='macro'),
            'test_precision': precision_score(y_test, y_pred, average='macro'),
            'train_recall': recall_score(y_train, y_train_pred, average='macro'),
            'test_recall': recall_score(y_test, y_pred, average='macro'),
            'train_f1': f1_score(y_train, y_train_pred, average='macro'),
            'test_f1': f1_score(y_test, y_pred, average='macro'),
        }

        pcount = model.count_params()
        flops = flops_measure(model, input_shape)
        model_size = get_tflite_model_size(model)

        model_name = f"{model_name_prefix}_{i + 1}"
        results[model_name] = {
            **metrics,
            'param_count': pcount,
            'flops': flops,
            'pred_time_s': t_pred,
            'model_size_bytes': model_size,
            'model_size_params': compute_parameters_size(None, pcount),
            'param_count (eff)': pcount,
            'dataset': config.get("dataset", "unknown")
        }

        if i + 1 == breaker_fold:
            break

    return results
