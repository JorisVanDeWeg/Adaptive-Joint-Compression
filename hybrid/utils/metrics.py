import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import contextlib
import numpy as np
import os


def flops_measure(model, input_shape):
    infer_fn = tf.function(lambda x: model(x, training=False))
    concrete = infer_fn.get_concrete_function(
        tf.TensorSpec([1] + list(input_shape), tf.float32)
    )
    frozen_fn = convert_variables_to_constants_v2(concrete)
    graph_def = frozen_fn.graph.as_graph_def()

    g = tf.Graph()
    with g.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=g,
            run_meta=run_meta,
            cmd='op',
            options=opts
        )
    return flops.total_float_ops


def compute_parameters_size(dist=None, param_count=None):
    if dist:
        total_bits = sum(bits * count for bits, count in dist.items())
        total_bytes = total_bits // 8
    elif param_count:
        total_bytes = param_count * 4
    else:
        total_bytes = np.nan
    return total_bytes


def get_tflite_model_size(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            tflite_model = converter.convert()
    return len(tflite_model) / 1024.0


def print_status(message):
    print(f"\n\n{'#' * 50}\n{'#' * 25}\n{message}\n{'#' * 25}\n{'#' * 50}\n\n")
