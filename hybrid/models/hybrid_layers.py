from keras import layers
import numpy as np
import tensorflow as tf
import math


@tf.custom_gradient
def ste_round(x):
    y = tf.round(x)
    def grad(dy):
        return dy
    return y, grad


class HybridDense(layers.Layer):
    default_gate_init_mode_quan = "random"
    default_gate_init_mode_decomp = "random"
    default_decomposition = "SVD"

    def __init__(self, d_out,
                 bitwidths=[4, 8, 16, 32],
                 init_alpha=-1.0,
                 init_beta=1.0,
                 threshold=0.5,
                 temperature=2.0,
                 use_bias=True,
                 activation=None,
                 gate_init_mode_quan=None,
                 gate_init_mode_decomp=None,
                 decomposition=None,
                 **kwargs):
        super(HybridDense, self).__init__(**kwargs)
        self.d_out = d_out
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.bitwidths = bitwidths
        self.threshold = threshold
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.gate_init_mode_quan = gate_init_mode_quan if gate_init_mode_quan is not None else HybridDense.default_gate_init_mode_quan
        self.gate_init_mode_decomp = gate_init_mode_decomp if gate_init_mode_decomp is not None else HybridDense.default_gate_init_mode_decomp
        self.decomposition = decomposition if decomposition is not None else HybridDense.default_decomposition
        self.temperature = tf.Variable(initial_value=temperature, trainable=False, dtype=self.dtype, name="temperature")
        self._frozen = False

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        if self.decomposition == "SVD":
            self.rank = max(1, int((d_in * self.d_out) / (d_in + self.d_out)))
        elif self.decomposition == "Tucker":
            b = d_in + self.d_out
            c = -d_in * self.d_out
            discriminant = b**2 - 4*c
            if discriminant < 0:
                raise ValueError("No valid Tucker rank found for the given input and output dimensions.")
            r_max = (-b + math.sqrt(discriminant)) / (2)
            self.rank = max(1, int(r_max))
        else:
            raise ValueError("decomposition must be 'SVD' or 'Tucker'")
        
        self.A = self.add_weight(name='A', shape=(d_in, self.rank), initializer='glorot_uniform', trainable=True)
        self.B = self.add_weight(name='B', shape=(self.rank, self.d_out), initializer='glorot_uniform', trainable=True)
        if self.decomposition == "SVD":
            self.E = self.add_weight(name='E', shape=(self.rank,), initializer='glorot_uniform', trainable=True)
        elif self.decomposition == "Tucker":
            self.E = self.add_weight(name='E', shape=(self.rank, self.rank), initializer='glorot_uniform', trainable=True)

        if self.rank > 1:
            if self.gate_init_mode_decomp == "off":
                gate_initializer = tf.constant_initializer(-5.0)
            elif self.gate_init_mode_decomp == "on":
                gate_initializer = tf.constant_initializer(5.0)
            else:
                gate_initializer = tf.keras.initializers.RandomNormal()
            self.g_logits_decomp = self.add_weight(name='g_logits_decomp', shape=(self.rank - 1, ),
                                                   initializer=gate_initializer, trainable=True)
            if self.decomposition == "Tucker":
                self.g_logits_decomp2 = self.add_weight(name='g_logits_decomp2', shape=(self.rank - 1,),
                                                        initializer=gate_initializer, trainable=True)
            else:
                self.g_logits_decomp2 = None
        else:
            self.g_logits_decomp = None
            self.g_logits_decomp2 = None

        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(self.d_out,), initializer='zeros', trainable=True)

        self.alpha = self.add_weight(name='alpha', shape=(), initializer=tf.constant_initializer(self.init_alpha), trainable=True)
        init_abs_delta = self.init_beta - self.init_alpha
        init_log_delta = np.log(np.exp(init_abs_delta) - 1.0)
        self.log_delta = self.add_weight(name='log_delta', shape=(), initializer=tf.constant_initializer(init_log_delta), trainable=True)

        if self.gate_init_mode_quan == "off":
            gate_initializer = tf.constant_initializer(-5.0)
        elif self.gate_init_mode_quan == "on":
            gate_initializer = tf.constant_initializer(5.0)
        else:
            gate_initializer = tf.keras.initializers.RandomNormal()

        self.g_logits_quan = {}
        for b in self.bitwidths:
            self.g_logits_quan[b] = self.add_weight(name=f'g_logit_quan_{b}', shape=(1,), initializer=gate_initializer, trainable=True)

        super().build(input_shape)

    def quantize_tensor(self, x, s):
        return s * ste_round(x / s)

    def _make_full_logits(self, g_logits):
        if self.rank == 1:
            return tf.constant([10.0], dtype=tf.float32)
        base = tf.constant([10.0], dtype=tf.float32)
        return tf.concat([base, g_logits], axis=0)
    
    def quantize_weight_matrix(self, W, training):
        beta = self.alpha + tf.nn.softplus(self.log_delta)
        W_clipped = tf.clip_by_value(W, self.alpha, beta)

        s_prev = (beta - self.alpha) / 3.0
        W_q = self.quantize_tensor(W_clipped - self.alpha, s_prev) + self.alpha

        mask = 1.0
        for b in self.bitwidths:
            sb = s_prev / (2**b + 1.0)
            eps = self.quantize_tensor(W_clipped - W_q, sb)
            gate_prob = tf.sigmoid(self.g_logits_quan[b])
            if training:
                z = tf.cast(tf.random.uniform(tf.shape(gate_prob)) < gate_prob, tf.float32)
                bit_mask = z + gate_prob - tf.stop_gradient(gate_prob)
            else:
                bit_mask = tf.cast(gate_prob > self.threshold, tf.float32)
            mask = mask * bit_mask
            W_q = W_q + mask * eps
            s_prev = sb
        return W_q

    def freeze_inference_weights(self):
        full_logits = self._make_full_logits(self.g_logits_decomp) / self.temperature
        g_prob = tf.sigmoid(full_logits)

        if self.decomposition == "SVD":
            active = tf.where(g_prob > self.threshold)[:, 0]
            A_eff = tf.gather(self.A, active, axis=1)
            E_eff = tf.gather(self.E, active, axis=0)
            B_eff = tf.gather(self.B, active, axis=0)
        else:
            full_logits2 = self._make_full_logits(self.g_logits_decomp2) / self.temperature
            g_prob2 = tf.sigmoid(full_logits2)
            active_rows = tf.where((g_prob > self.threshold))[:, 0]
            active_columns = tf.where((g_prob2 > self.threshold))[:, 0]
            A_eff = tf.gather(self.A, active_rows, axis=1)
            E_temp = tf.gather(self.E, active_rows, axis=0)
            E_eff = tf.gather(E_temp, active_columns, axis=1)
            B_eff = tf.gather(self.B, active_columns, axis=0)

        self.A_codes, self.s_A, self.alpha_A = self._compute_quantization_codes(A_eff)
        self.E_codes, self.s_E, self.alpha_E = self._compute_quantization_codes(E_eff)
        self.B_codes, self.s_B, self.alpha_B = self._compute_quantization_codes(B_eff)
        self._frozen = True
            
    def _compute_quantization_codes(self, W):
        alpha = self.alpha
        beta  = alpha + tf.nn.softplus(self.log_delta)
        Wc = tf.clip_by_value(W, alpha, beta)
        s = (beta - alpha) / 3.0
        W_q = tf.round((Wc - alpha) / s) * s + alpha
        s_final = s
        b_prev = 2
        active = tf.constant(1.0, dtype=W.dtype)

        for b in sorted(set(self.bitwidths)):
            s_next = s * ((2**b_prev - 1) / (2**b - 1))
            residual = Wc - W_q
            eps = tf.round(residual / s_next) * s_next
            gate_prob = tf.sigmoid(self.g_logits_quan[b])
            hard_mask = tf.cast(gate_prob > self.threshold, W.dtype)
            active = active * hard_mask
            W_q = W_q + active * eps
            s_final = tf.where(active > 0.5, s_next, s_final)
            s = s_next
            b_prev = b

        code = tf.cast(tf.round((W_q - alpha) / s_final), tf.int64)
        return code, s_final, alpha

    def call(self, inputs, training=True):
        if self._frozen and not training:
            A_eff = tf.cast(self.A_codes, tf.float32) * self.s_A + self.alpha_A
            E_eff = tf.cast(self.E_codes, tf.float32) * self.s_E + self.alpha_E
            B_eff = tf.cast(self.B_codes, tf.float32) * self.s_B + self.alpha_B
            if self.decomposition == "SVD":
                z = tf.matmul(inputs, A_eff)
                z = z * E_eff
                out = tf.matmul(z, B_eff)
            else:
                z = tf.matmul(inputs, A_eff)
                z = tf.matmul(z, E_eff)
                out = tf.matmul(z, B_eff)
            if self.use_bias:
                out = tf.nn.bias_add(out, self.bias)
            if self.activation is not None:
                out = self.activation(out)
            return out
        else:
            g_prob_rows = tf.sigmoid(self._make_full_logits(self.g_logits_decomp) / self.temperature)
            if self.decomposition == "Tucker":
                g_prob_cols = tf.sigmoid(self._make_full_logits(self.g_logits_decomp2) / self.temperature)
            if training:
                sample = tf.cast(tf.random.uniform(tf.shape(g_prob_rows)) < g_prob_rows, self.dtype)
                mask_rows = sample + g_prob_rows - tf.stop_gradient(g_prob_rows)
                if self.decomposition == "Tucker":
                    sample2 = tf.cast(tf.random.uniform(tf.shape(g_prob_cols)) < g_prob_cols, self.dtype)
                    mask_cols = sample2 + g_prob_cols - tf.stop_gradient(g_prob_cols)
            else:
                mask_rows = tf.cast(g_prob_rows > self.threshold, self.dtype)
                if self.decomposition == "Tucker":
                    mask_cols = tf.cast(g_prob_cols > self.threshold, self.dtype)

        A_hat = self.quantize_weight_matrix(self.A, training)
        B_hat = self.quantize_weight_matrix(self.B, training)
        E_hat = self.quantize_weight_matrix(self.E, training)
        
        if self.decomposition == "SVD":
            masked_E = mask_rows * E_hat
            z = tf.matmul(inputs, A_hat)
            z *= masked_E
            out = tf.matmul(z, B_hat)
        elif self.decomposition == "Tucker":
            A_masked = A_hat * mask_rows
            B_masked = B_hat * tf.expand_dims(mask_cols, 1)
            E_masked = E_hat * tf.tensordot(mask_rows, mask_cols, axes=0)
            z = tf.matmul(inputs, A_masked)
            z = tf.matmul(z, E_masked)
            out = tf.matmul(z, B_masked)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias)
        if self.activation is not None:
            out = self.activation(out)
        return out

    def effective_params(self):
        if not self._frozen:
            raise RuntimeError("Effective parameters can only be computed after freezing the weights.")
        num_A_codes = tf.shape(self.A_codes)[0] * tf.shape(self.A_codes)[1]
        if self.decomposition == "SVD":
            num_E_codes = tf.shape(self.E_codes)[0]
        else:
            num_E_codes = tf.shape(self.E_codes)[0] * tf.shape(self.E_codes)[1]
        num_B_codes = tf.shape(self.B_codes)[0] * tf.shape(self.B_codes)[1]
        total_quan = int(num_A_codes) + int(num_E_codes) + int(num_B_codes)
        num_scalars = 2
        num_bias = int(self.d_out) if self.use_bias else 0
        total = total_quan + num_scalars + num_bias
        return total, total_quan
    
    def get_max_effective_bitwidth(self):
        max_used = 2
        active = True
        for b in sorted(self.bitwidths):
            gate_prob = tf.sigmoid(self.g_logits_quan[b])
            if active and gate_prob > self.threshold:
                max_used = b
            else:
                active = False
        return max_used


class HybridMultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, **kwargs):
        super(HybridMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        d_model = int(input_shape[-1])
        proj_dim = self.num_heads * self.key_dim
        self.q_dense = HybridDense(proj_dim, name="q_dense")
        self.k_dense = HybridDense(proj_dim, name="k_dense")
        self.v_dense = HybridDense(proj_dim, name="v_dense")
        self.out_dense = HybridDense(d_model, name="out_dense")
        self.dropout = layers.Dropout(self.dropout_rate)
        super(HybridMultiHeadAttention, self).build(input_shape)
    
    def call(self, query, key, value, training=True):
        batch_size = tf.shape(query)[0]
        seq_len = tf.shape(query)[1]
        q = self.q_dense(query, training=training)
        k = self.k_dense(key, training=training)
        v = self.v_dense(value, training=training)
        
        def reshape_to_heads(x):
            x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.key_dim))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        q = reshape_to_heads(q)
        k = reshape_to_heads(k)
        v = reshape_to_heads(v)
        
        dk = tf.cast(self.key_dim, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        attn = tf.matmul(weights, v)
        
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, (batch_size, seq_len, self.num_heads * self.key_dim))
        output = self.out_dense(attn, training=training)
        return output
    
    def effective_params(self):
        total = 0
        quan_total = 0
        for layer in [self.q_dense, self.k_dense, self.v_dense, self.out_dense]:
            t, q = layer.effective_params()
            total += t
            quan_total += q
        return total, quan_total
    

class HybridConv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding="same", use_bias=True, activation=None, 
                 init_alpha=-1.0, init_beta=1.0, bitwidths=[4, 8, 16, 32], threshold=0.5,
                 temperature=2.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
        self.init_alpha = float(init_alpha)
        self.init_beta = float(init_beta)
        self.bitwidths = bitwidths
        self.threshold = threshold
        self.temperature = tf.Variable(temperature, trainable=False, dtype=tf.float32)
        self.gate_init_mode_quan = HybridDense.default_gate_init_mode_quan
        self.gate_init_mode_decomp = HybridDense.default_gate_init_mode_decomp
        self.decomposition = HybridDense.default_decomposition
        self._frozen_bias = None
        self._frozen_kernel1 = None
        self._frozen_kernel2 = None
        self._frozen = False

    def build(self, input_shape):
        C_in = int(input_shape[-1])
        self.rank = max(1, (C_in * self.filters * self.kernel_size) // (C_in * self.kernel_size + self.filters))

        self.conv1 = tf.keras.layers.Conv1D(
            filters=self.rank,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=False,
            name="conv1_lowrank"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=1,
            padding=self.padding,
            use_bias=self.use_bias,
            name="conv2_projection"
        )

        if self.rank > 1:
            if self.gate_init_mode_decomp == "off":
                gate_initializer = tf.constant_initializer(-5.0)
            elif self.gate_init_mode_decomp == "on":
                gate_initializer = tf.constant_initializer(5.0)
            else:
                gate_initializer = "random_normal"
            self.g_logits_decomp = self.add_weight(
                name="g_logits_decomp", shape=(self.rank - 1,),
                initializer=gate_initializer, trainable=True)
        else:
            self.g_logits_decomp = None

        self.alpha = self.add_weight(name='alpha', shape=(), initializer=tf.constant_initializer(self.init_alpha), trainable=True)
        init_abs_delta = self.init_beta - self.init_alpha
        init_log_delta = np.log(np.exp(init_abs_delta) - 1.0)
        self.log_delta = self.add_weight(name='log_delta', shape=(), initializer=tf.constant_initializer(init_log_delta), trainable=True)
        
        if self.gate_init_mode_quan == "off":
            gate_initializer = tf.constant_initializer(-5.0)
        elif self.gate_init_mode_quan == "on":
            gate_initializer = tf.constant_initializer(5.0)
        else:
            gate_initializer = 'random_normal'

        self.g_logits_quan = {}
        for b in self.bitwidths:
            self.g_logits_quan[b] = self.add_weight(name=f'g_logit_quan_{b}', shape=(1,), initializer=gate_initializer, trainable=True)
        
        super().build(input_shape)

    def _full_logits(self):
        if self.g_logits_decomp is None:
            return tf.constant([10.0], dtype=self.dtype)
        return tf.concat([tf.constant([10.0], dtype=self.dtype), self.g_logits_decomp], axis=0)
    
    def quantize_weight_matrix(self, W, training):
        alpha = self.alpha
        beta  = alpha + tf.nn.softplus(self.log_delta)
        Wc    = tf.clip_by_value(W, alpha, beta)
        s_prev = (beta - alpha) / 3.0
        W_q    = ste_round((Wc - alpha) / s_prev) * s_prev + alpha
        active = tf.constant(1.0, dtype=W.dtype)
        b_prev = 2

        for b in sorted(set(self.bitwidths)):
            if b <= b_prev:
                continue
            sb = s_prev * ((2**b_prev - 1.0) / (2**b - 1.0))
            residual = Wc - W_q
            eps      = ste_round(residual / sb) * sb
            gate_prob = tf.sigmoid(self.g_logits_quan[b])
            if training:
                z        = tf.cast(tf.random.uniform(tf.shape(gate_prob)) < gate_prob, W.dtype)
                bit_mask = gate_prob + tf.stop_gradient(z - gate_prob)
            else:
                bit_mask = tf.cast(gate_prob > self.threshold, W.dtype)
            active = active * bit_mask
            W_q    = W_q + active * eps
            s_prev = sb
            b_prev = b
        return W_q

    def call(self, x, training=True):
        if self._frozen and not training:
            z = tf.nn.conv1d(x, self._frozen_kernel1, stride=1, padding=self.padding.upper())
            y = tf.nn.conv1d(z, self._frozen_kernel2, stride=1, padding=self.padding.upper())
            if self._frozen_bias is not None:
                y = tf.nn.bias_add(y, self._frozen_bias)
        else:
            g_prob = tf.sigmoid(self._full_logits() / self.temperature)
            if training:
                rand = tf.random.uniform(tf.shape(g_prob))
                z_gate = tf.cast(rand < g_prob, tf.float32)
                mask = g_prob + tf.stop_gradient(z_gate - g_prob)
            else:
                mask = tf.cast(g_prob > self.threshold, tf.float32)

            z = self.conv1(x)
            _ = self.conv2(z)
            
            k1 = self.quantize_weight_matrix(self.conv1.kernel, training)
            k2 = self.quantize_weight_matrix(self.conv2.kernel, training)

            z = tf.nn.conv1d(x, k1, stride=1, padding=self.padding.upper())
            z *= mask[None, None, :]
            y = tf.nn.conv1d(z, k2, stride=1, padding=self.padding.upper())

            if self.use_bias:
                b = self.conv2.bias
                y = tf.nn.bias_add(y, b)

        if self.activation is not None:
            y = self.activation(y)
        return y

    def freeze_inference_weights(self):
        if not self.conv1.built:
            self.conv1.build(self.input_shape)
        if not self.conv2.built:
            self.conv2.build(self.conv1.compute_output_shape(self.input_shape))
        g_prob = tf.sigmoid(self._full_logits() / self.temperature)
        idxs = tf.where(g_prob > self.threshold)[:, 0]
        if tf.size(idxs) == 0:
            idxs = tf.constant([0], dtype=tf.int64)
        k1 = tf.gather(self.conv1.kernel, idxs, axis=-1)
        k2 = tf.gather(self.conv2.kernel, idxs, axis=-2)
        b = self.conv2.bias if self.use_bias else None
        self._frozen_kernel1 = self.quantize_weight_matrix(k1, training=False)
        self._frozen_kernel2 = self.quantize_weight_matrix(k2, training=False)
        if b is not None:
            self._frozen_bias = tf.identity(b)
        else:
            self._frozen_bias = None
        self._frozen = True

    def effective_params(self):
        g_prob = tf.sigmoid(self._full_logits())
        r_eff = tf.reduce_sum(tf.cast(g_prob > self.threshold, tf.float32))
        C_in = tf.cast(tf.shape(self.conv1.kernel)[1], tf.float32)
        num_k1_codes = C_in * r_eff * self.kernel_size
        num_k2_codes = r_eff * self.filters
        num_bias = self.filters if self.use_bias else 0
        num_scalars = 2
        total = num_k1_codes + num_k2_codes + num_bias + num_scalars
        total_quan = num_k1_codes + num_k2_codes
        total = int(total)
        total_quan = int(total_quan)
        return total, total_quan

    def get_max_effective_bitwidth(self):
        max_used = 2
        active = True
        for b in sorted(self.bitwidths):
            gate_prob = tf.sigmoid(self.g_logits_quan[b])
            if active and gate_prob > self.threshold:
                max_used = b
            else:
                active = False
        return max_used
