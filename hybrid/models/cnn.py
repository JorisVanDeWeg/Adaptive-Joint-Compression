from keras import layers, Model
import tensorflow as tf


def cnn_attention(
    input_shape,
    num_classes,
    cnn_filters=[32, 64],
    transformer_dim=64,
    num_heads=4,
    ff_dim=32,
    num_transformer_blocks=2,
    dropout_rate=0.1
):
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = inputs

    for i, num_filters in enumerate(cnn_filters):
        x = layers.Conv1D(filters=num_filters, kernel_size=3, activation='relu', padding='same', name=f'conv1d_{i}')(x)

    x = layers.Conv1D(filters=transformer_dim, kernel_size=3, activation='relu', padding='valid', name='conv1d_final')(x)
    x = layers.GlobalAveragePooling1D(name='global_avg_pool1d')(x)
    x = layers.Reshape((1, transformer_dim), name='reshape_to_seq')(x)

    for block in range(num_transformer_blocks):
        attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=transformer_dim // num_heads,
                                                name=f'mha_{block}')(x, x)
        attn_output = layers.Dropout(dropout_rate, name=f'mha_dropout_{block}')(attn_output)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'mha_layernorm_{block}')(x + attn_output)

        ffn = layers.Dense(ff_dim, activation='relu', name=f'ffn_dense1_{block}')(x)
        ffn = layers.Dense(transformer_dim, name=f'ffn_dense2_{block}')(ffn)
        ffn = layers.Dropout(dropout_rate, name=f'ffn_dropout_{block}')(ffn)
        x = layers.LayerNormalization(epsilon=1e-6, name=f'ffn_layernorm_{block}')(x + ffn)

    x = layers.GlobalAveragePooling1D(name='global_avg_pool_final')(x)
    x = layers.Dropout(dropout_rate, name='final_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output_dense')(x)
    return Model(inputs=inputs, outputs=outputs)
