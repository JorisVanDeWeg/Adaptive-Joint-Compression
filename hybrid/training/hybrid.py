import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from hybrid.models.hybrid_models import (
    get_all_layers, custom_loss, compute_model_parameter_storage,
    get_effective_params_hybrid, get_conv_rank_distribution,
    get_dense_rank_distribution, get_quantization_selection
)
from hybrid.utils.metrics import flops_measure, get_tflite_model_size


class TemperatureAnneal(tf.keras.callbacks.Callback):
    def __init__(self, decay_rate=0.9, T_min=0.5, verbose=False):
        super().__init__()
        self.decay_rate = decay_rate
        self.T_min      = T_min
        self.verbose    = verbose

    def on_epoch_begin(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "temperature"):
                old = layer.temperature.numpy()
                new = max(self.T_min, old * self.decay_rate)
                layer.temperature.assign(new)
                if self.verbose:
                    print(f"[{layer.name}] temperature {old:.3f} → {new:.3f}")


def run_hybrid_experiment(config, folds, dataset_folds):
    results = {}
    histories = {}
    input_shape = config["input_shape"]

    from hybrid.models.hybrid_layers import HybridDense
    for fold_i, ((train_ds, val_ds, test_ds), (X_train, X_test, y_train, y_test)) in enumerate(zip(dataset_folds, folds)):
        print(f"Running fold {fold_i+1} out of {len(folds)}")
        HybridDense.default_decomposition = config["decomposition"]
        for gate_mode in config["quan_gates_init_modes"]:
            HybridDense.default_gate_init_mode_quan = gate_mode
            for gate_mode in config["decomp_gates_init_modes"]:
                HybridDense.default_gate_init_mode_decomp = gate_mode

                for lambda_d in config["lambda_d_values"]:
                    for lambda_q in config["lambda_q_values"]:

                        hybrid_model = config['model']()
                        loss_fn = lambda y_true, y_pred: custom_loss(
                            y_true, y_pred, hybrid_model,
                            lambda_d=lambda_d, lambda_q=lambda_q
                        )
                        dummy_input = tf.zeros((1, *input_shape))
                        _ = hybrid_model(dummy_input, training=False)

                        steps_per_epoch = len(X_train) // config["batch_size"]
                        decay_steps = config["epochs_per_decay"] * steps_per_epoch

                        optimizer = tf.keras.optimizers.Adam(
                            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                                initial_learning_rate=config['initial_learning_rate'],
                                decay_steps=decay_steps,
                                decay_rate=config['decay_rate'],
                                staircase=True
                            )
                        )

                        hybrid_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], jit_compile=True, run_eagerly=False)

                        anneal_cb = TemperatureAnneal(decay_rate=0.95, T_min=0.5, verbose=False)

                        history = hybrid_model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=config["epochs"],
                            batch_size=config["batch_size"],
                            callbacks=[anneal_cb],
                            verbose=config["verbose"]
                        )
                        
                        for layer in get_all_layers(hybrid_model):
                            if hasattr(layer, "freeze_inference_weights"):
                                layer.freeze_inference_weights()
                        print(f"Model is frozen for inference.")
                        
                        _, train_accuracy = hybrid_model.evaluate(train_ds, verbose=0)
                        _, test_accuracy = hybrid_model.evaluate(test_ds, verbose=0)

                        t0 = time.time()
                        y_prob = hybrid_model.predict(X_test, verbose=0)
                        y_pred = np.argmax(y_prob, axis=1)
                        t_pred = time.time() - t0

                        y_train_prob = hybrid_model.predict(X_train, verbose=0)
                        y_train_pred = np.argmax(y_train_prob, axis=1)

                        from sklearn.metrics import precision_score, recall_score, f1_score
                        metrics = {
                            'train_accuracy':  train_accuracy,
                            'test_accuracy':   test_accuracy,
                            'train_precision': precision_score(y_train, y_train_pred, average='macro'),
                            'test_precision':  precision_score(y_test, y_pred, average='macro'),
                            'train_recall':    recall_score(y_train, y_train_pred, average='macro'),
                            'test_recall':     recall_score(y_test, y_pred, average='macro'),
                            'train_f1':        f1_score(y_train, y_train_pred, average='macro'),
                            'test_f1':         f1_score(y_test, y_pred, average='macro'),
                        }

                        pcount = hybrid_model.count_params()
                        flops = flops_measure(hybrid_model, config["input_shape"])
                        model_size = get_tflite_model_size(hybrid_model)

                        prefix = f"{config['model_name_prefix']} (gate={gate_mode}, $\\lambda_d$={lambda_d}, $\\lambda_q$={lambda_q})"
                        key = f"{prefix}_{fold_i+1}"
                        results[key] = {
                            **metrics,
                            'param_count': pcount,
                            'flops': flops,
                            'pred_time_s': t_pred,
                            'model_size_bytes': model_size,
                            'model_size_params': compute_model_parameter_storage(hybrid_model),
                            'param_count (eff)': hybrid_model.effective_params(),
                            'conv_rank_distribution': get_conv_rank_distribution(hybrid_model),
                            'dense_rank_distribution': get_dense_rank_distribution(hybrid_model),
                            'quantization_selection': get_quantization_selection(hybrid_model),
                            'dataset': config['Dataset']
                        }

                        if history is not None and config["plot_bool"]:
                            plt.plot(history.history['accuracy'])
                            plt.plot(history.history['val_accuracy'])
                            plt.title('Model Accuracy')
                            plt.ylabel('Accuracy')
                            plt.xlabel('Epoch')
                            plt.legend(['Train', 'Validation'], loc='upper left')
                            plt.show()

                        if history is not None:
                            train_curve = history.history.get('accuracy', None)
                            val_curve   = history.history.get('val_accuracy', None)
                            prefix = f"{config['model_name_prefix']} (gate={gate_mode}, lambda_d={lambda_d}, lambda_q={lambda_q})"
                            bucket = histories.setdefault(prefix, {'train': [], 'val': []})
                            if train_curve is not None:
                                bucket['train'].append(train_curve)
                            if val_curve is not None:
                                bucket['val'].append(val_curve)
        
        if fold_i + 1 == config["breaker_fold"]:
            if config.get("save_plots", False):
                _save_mean_std_plots(histories, config.get("output_folder", "./plots"))
            return results

    if config.get("save_plots", False):
        _save_mean_std_plots(histories, config.get("output_folder", "./plots"))
    return results


def _save_mean_std_plots(histories, output_folder, split='val'):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    if not histories:
        return
    os.makedirs(output_folder, exist_ok=True)

    def _safe(name):
        return (str(name)
                .replace(' ', '_')
                .replace('(', '').replace(')', '')
                .replace(',', '').replace('=', '_')
                .replace('/', '_').replace('\\', '_'))

    for prefix, buckets in histories.items():
        curves = buckets.get(split, [])
        if not curves:
            continue
        E = min(len(c) for c in curves)
        M = np.stack([np.asarray(c[:E], float) for c in curves], axis=0)
        mean = M.mean(axis=0)
        std  = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(1, E + 1)

        plt.figure()
        plt.plot(x, mean, label='Mean')
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, label='±1 SD')
        plt.xlabel('Epoch')
        plt.ylabel(f'{split.capitalize()} Accuracy')
        plt.title(f'{prefix} — {split.capitalize()} Accuracy (Mean ± Std over {M.shape[0]} Folds)')
        plt.legend()
        plt.tight_layout()

        fname = f"{_safe(prefix)}_{split}_accuracy_mean_std.pdf"
        plt.savefig(os.path.join(output_folder, fname), bbox_inches='tight')
        plt.close()
