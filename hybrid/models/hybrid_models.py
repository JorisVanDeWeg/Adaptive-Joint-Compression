from keras import layers, Model
import tensorflow as tf
import numpy as np

from .hybrid_layers import HybridConv1D, HybridMultiHeadAttention, HybridDense


def cnn_attention_hybrid(
        input_shape,
        num_classes,
        cnn_filters=[32, 64],
        transformer_dim=64,
        num_heads=4,
        ff_dim=32,
        num_transformer_blocks=2,
        dropout_rate=0.1,
        structure=[True, True],
        ):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = inputs

    for i, num_filters in enumerate(cnn_filters):
        if structure[0]:
            x = HybridConv1D(filters=num_filters, kernel_size=3, activation='relu', padding='same', name=f'lowrank_conv1d_{i}')(x)
        else:
            x = layers.Conv1D(filters=num_filters, kernel_size=3, activation='relu', padding='same', name=f'conv1d_{i}')(x)
        
    if structure[0]:
        x = HybridConv1D(filters=transformer_dim, kernel_size=3, activation='relu', padding='valid', name='lowrank_conv1d_final')(x)
    else:
        x = layers.Conv1D(filters=transformer_dim, kernel_size=3, activation='relu', padding='valid', name='conv1d_final')(x)

    if structure[1]:
        x = layers.GlobalAveragePooling1D(name='global_avg_pool1d')(x)
        x = layers.Reshape((1, transformer_dim), name='reshape_to_seq')(x)
    
    for block in range(num_transformer_blocks):
        attn_output = HybridMultiHeadAttention(
            num_heads=num_heads,
            key_dim=transformer_dim // num_heads,
            dropout_rate=dropout_rate,
            name=f'lowrank_mha_{block}')(x, x, x)
        
        attn_output = layers.Dropout(dropout_rate, name=f'mha_dropout_{block}')(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'mha_layernorm_{block}')(x + attn_output)

        ffn = HybridDense(ff_dim, activation='relu', name=f'lowrank_ffn1_{block}')(x)
        ffn = HybridDense(transformer_dim, name=f'lowrank_ffn2_{block}')(ffn)
        ffn = layers.Dropout(dropout_rate, name=f'ffn_dropout_{block}')(ffn)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ffn_layernorm_{block}')(x + ffn)

    x = layers.GlobalAveragePooling1D(name='global_avg_pool_final')(x)
    x = layers.Dropout(dropout_rate, name='final_dropout')(x)
    outputs = HybridDense(num_classes, name='output_dense')(x)
    outputs = layers.Activation('softmax', name='softmax_out')(outputs)

    model = Model(inputs=inputs, outputs=outputs, name='cnn_attention_hybrid')
    model.effective_params = lambda: get_effective_params_hybrid(model)
    return model


def get_all_layers(layer):
    layers_found = set()
    stack = [layer]
    while stack:
        current = stack.pop()
        if current in layers_found:
            continue
        layers_found.add(current)
        if hasattr(current, "layers"):
            stack.extend(current.layers)
        for attr in dir(current):
            try:
                attr_val = getattr(current, attr)
            except Exception:
                continue
            if isinstance(attr_val, tf.keras.layers.Layer) and attr_val not in layers_found:
                stack.append(attr_val)
    return list(layers_found)


def collect_gate_logits(model):
    gate_logits = []
    for layer in get_all_layers(model):
        if hasattr(layer, "g_logits_decomp") and layer.g_logits_decomp is not None:
            gate_logits.append((f"{layer.name}_decomp", layer.g_logits_decomp))
        if hasattr(layer, "g_logits_decomp2") and layer.g_logits_decomp2 is not None:
            gate_logits.append((f"{layer.name}_decomp2", layer.g_logits_decomp2))
        if hasattr(layer, "g_logits_quan") and isinstance(layer, "dict"):
            for b, logit in layer.g_logits_quan.items():
                gate_logits.append((f"{layer.name}_quan_{b}", logit))
        if hasattr(layer, "g_logits_tr") and layer.g_logits_tr is not None:
            gate_logits.append((f"{layer.name}_tr", layer.g_logits_tr))
    return gate_logits


def custom_loss(y_true, y_pred, model, lambda_d=0.01, lambda_q=0.01):
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce = tf.reduce_mean(ce)

    decomp_layer_terms = []
    for layer in get_all_layers(model):
        comps = []
        if hasattr(layer, "g_logits_decomp") and layer.g_logits_decomp is not None:
            comps.append(tf.sigmoid(layer.g_logits_decomp))
        if hasattr(layer, "g_logits_decomp2") and layer.g_logits_decomp2 is not None:
            comps.append(tf.sigmoid(layer.g_logits_decomp2))
        if comps:
            probs = tf.concat(comps, axis=0)
            layer_term = tf.reduce_mean(probs)
            decomp_layer_terms.append(layer_term)

    if decomp_layer_terms:
        L_D = tf.add_n(decomp_layer_terms) / tf.cast(len(decomp_layer_terms), tf.float32)
    else:
        L_D = tf.constant(0.0, dtype=ce.dtype)

    quant_layer_terms = []
    for layer in get_all_layers(model):
        gdict = getattr(layer, "g_logits_quan", None)
        if isinstance(gdict, dict) and gdict:
            bws = sorted(gdict.keys())
            probs = tf.concat([tf.sigmoid(gdict[b]) for b in bws], axis=0)
            cum = tf.math.cumprod(probs)
            layer_term = tf.reduce_mean(cum)
            quant_layer_terms.append(layer_term)

    if quant_layer_terms:
        L_Q = tf.add_n(quant_layer_terms) / tf.cast(len(quant_layer_terms), tf.float32)
    else:
        L_Q = tf.constant(0.0, dtype=ce.dtype)

    return ce + lambda_d * L_D + lambda_q * L_Q


def compute_model_parameter_storage(model):
    total_bits = 0
    for layer in get_all_layers(model):
        if hasattr(layer, "effective_params") and hasattr(layer, "get_max_effective_bitwidth"):
            _, quantized_param_count = layer.effective_params()
            bitwidth = layer.get_max_effective_bitwidth()
            if bitwidth is not None:
                total_bits += quantized_param_count * bitwidth
    return total_bits // 8


def get_effective_params_hybrid(model):
    total = 0.0
    quan_total = 0.0
    for layer in model.layers:
        if hasattr(layer, "effective_params"):
            total_add, quant_total_add = layer.effective_params()
            total += total_add
            quan_total += quant_total_add
        elif hasattr(layer, "layers"):
            for sub_layer in layer.layers:
                if hasattr(sub_layer, "effective_params"):
                    total_add, quant_total_add = sub_layer.effective_params()
                    total += total_add
                    quan_total += quant_total_add
    return total


def get_conv_rank_distribution(model):
    dist = {}
    for layer in get_all_layers(model):
        if layer.__class__.__name__ == "HybridConv1D" and getattr(layer, "g_logits_decomp", None) is not None:
            full = layer._full_logits()
            probs = tf.sigmoid(full)
            active = probs > layer.threshold
            ac = int(tf.reduce_sum(tf.cast(active, tf.int32)))
            tot = int(tf.size(full))
            dist[layer.name] = {
                "active_count": ac,
                "total": tot,
                "active_fraction": (ac / tot) if tot else 0.0,
            }
    return dist


def get_dense_rank_distribution(model):
    dist = {}
    for layer in get_all_layers(model):
        if layer.__class__.__name__ != "HybridDense":
            continue
        entry = {}
        if getattr(layer, "g_logits_decomp", None) is not None:
            full_r = layer._make_full_logits(layer.g_logits_decomp)
            probs_r = tf.sigmoid(full_r)
            active_r = probs_r > layer.threshold
            ac1 = int(tf.reduce_sum(tf.cast(active_r, tf.int32)))
            tot = int(tf.size(full_r))
            entry["active_count_1"] = ac1
            entry["total"] = tot
            entry["active_fraction_1"] = (ac1 / tot) if tot else 0.0
        if getattr(layer, "g_logits_decomp2", None) is not None:
            full_c = layer._make_full_logits(layer.g_logits_decomp2)
            probs_c = tf.sigmoid(full_c)
            active_c = probs_c > layer.threshold
            ac2 = int(tf.reduce_sum(tf.cast(active_c, tf.int32)))
            entry["active_count_2"] = ac2
            entry["active_fraction_2"] = (ac2 / entry["total"]) if entry.get("total", 0) else 0.0
        if entry:
            dist[layer.name] = entry
    return dist


def get_quantization_selection(model):
    out = {}
    for layer in get_all_layers(model):
        gdict = getattr(layer, "g_logits_quan", None)
        if not isinstance(gdict, dict) or not gdict:
            continue
        bitwidths = sorted(gdict.keys())
        probs = {int(b): float(tf.sigmoid(gdict[b]).numpy()) for b in bitwidths}
        selected = 2
        active = True
        for b in bitwidths:
            p = probs[int(b)]
            if active and p > float(layer.threshold):
                selected = int(b)
            else:
                active = False
        out[layer.name] = {
            "selected_bitwidth": selected,
            "gate_probs": probs
        }
    return out
