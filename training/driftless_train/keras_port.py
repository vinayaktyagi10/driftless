"""Rebuild the trained TCN natively in Keras and transfer the PyTorch weights.

Why not just convert the ONNX
-----------------------------
onnx2tf proved unreliable for this architecture. Bisecting the graph showed the
**residual connection** is what breaks it: a single block without `x + y`
converts, and with it the TFLite graph fails to prepare
("num_input_elements != num_output_elements (3936 != 3)", where 3936 = 48x82 is a
padded intermediate). Worse, a variant that did convert disagreed with PyTorch by
5.4e-2 -- silently wrong, which is the dangerous kind. Shipping role 01 a model on
that basis would surface as a mystery on stage.

So we define the same network with Keras layers, copy the trained weights across,
and use the first-party Keras -> TFLite converter. Parity against PyTorch is
asserted at both hops.

Layout note for role 01: Keras is channels-last, so the TFLite input is
(1, win, n_channels) -- time-major -- whereas the ONNX input is (1, n_channels,
win). This is the natural layout for an Android ring buffer anyway.
"""

from __future__ import annotations

import numpy as np


def build_keras_tcn(n_channels: int, win: int, width: int, n_out: int,
                    dilations: tuple[int, ...], kernel: int = 3):
    """Keras mirror of SpeedHeadingTCN. Channels-last, causal, no dropout."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(win, n_channels), dtype="float32", name="imu_window")

    # Input normalisation, baked in as constants (set later from the checkpoint).
    x_mean = tf.Variable(np.zeros(n_channels, np.float32), trainable=False,
                         name="x_mean")
    x_std = tf.Variable(np.ones(n_channels, np.float32), trainable=False,
                        name="x_std")
    y_mean = tf.Variable(np.zeros(n_out, np.float32), trainable=False,
                         name="y_mean")
    y_std = tf.Variable(np.ones(n_out, np.float32), trainable=False, name="y_std")

    h = layers.Lambda(lambda t: (t - x_mean) / x_std, name="normalise")(inp)
    h = layers.Conv1D(width, 1, name="stem")(h)

    for i, d in enumerate(dilations):
        pad = (kernel - 1) * d
        y = layers.ZeroPadding1D(padding=(pad, 0), name=f"pad{i}")(h)
        y = layers.Conv1D(width, kernel, dilation_rate=d, name=f"conv{i}")(y)
        # epsilon MUST be set explicitly: Keras defaults to 1e-3 while PyTorch's
        # GroupNorm uses 1e-5, and that alone put the ported model 5e-3 out.
        y = layers.GroupNormalization(groups=4, axis=-1, epsilon=1e-5,
                                      name=f"norm{i}")(y)
        y = layers.Activation("gelu", name=f"act{i}")(y)
        h = layers.Add(name=f"res{i}")([h, y])

    # Take the final timestep, then apply the pointwise head. The head is 1x1
    # convolutions, so this is identical to applying it and slicing afterwards,
    # and it is cheaper.
    h = layers.Lambda(lambda t: t[:, -1:, :], name="last_step")(h)
    h = layers.Conv1D(width, 1, name="head0")(h)
    h = layers.Activation("gelu", name="head_act")(h)
    h = layers.Conv1D(n_out, 1, name="head1")(h)
    h = layers.Reshape((n_out,), name="flatten")(h)
    out = layers.Lambda(lambda t: t * y_std + y_mean, name="denormalise")(h)

    model = keras.Model(inp, out, name="speed_heading_tcn")
    model._driftless_consts = (x_mean, x_std, y_mean, y_std)
    return model


def transfer_weights(torch_model, keras_model) -> None:
    """Copy PyTorch parameters into the Keras mirror.

    Conv1d kernels differ in layout: PyTorch is (out, in, k), Keras is
    (k, in, out), so each kernel is transposed (2, 1, 0).
    """
    sd = {k: v.detach().cpu().numpy() for k, v in torch_model.state_dict().items()}

    def set_conv(keras_name: str, torch_prefix: str) -> None:
        w = sd[f"{torch_prefix}.weight"]        # (out, in, k)
        b = sd[f"{torch_prefix}.bias"]
        keras_model.get_layer(keras_name).set_weights([w.transpose(2, 1, 0), b])

    def set_norm(keras_name: str, torch_prefix: str) -> None:
        keras_model.get_layer(keras_name).set_weights(
            [sd[f"{torch_prefix}.weight"], sd[f"{torch_prefix}.bias"]])

    set_conv("stem", "stem")
    n_blocks = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    for i in range(n_blocks):
        set_conv(f"conv{i}", f"blocks.{i}.conv")
        set_norm(f"norm{i}", f"blocks.{i}.norm")
    set_conv("head0", "head.0")
    set_conv("head1", "head.2")

    x_mean, x_std, y_mean, y_std = keras_model._driftless_consts
    x_mean.assign(sd["x_mean"])
    x_std.assign(sd["x_std"])
    y_mean.assign(sd["y_mean"])
    y_std.assign(sd["y_std"])


def keras_from_checkpoint(torch_model, n_channels: int, win: int, width: int,
                          n_out: int, dilations: tuple[int, ...]):
    km = build_keras_tcn(n_channels, win, width, n_out, dilations)
    transfer_weights(torch_model, km)
    return km


def to_tflite(keras_model, quantise: str = "none") -> bytes:
    """Convert to TFLite.

    `quantise` is "none" by default and should stay that way. Setting
    tf.lite.Optimize.DEFAULT applies dynamic-range int8 weight quantisation,
    which moved the outputs 2-4% off PyTorch -- and for a model this small
    (78 KB float32) it buys nothing. float16 was worse still: up to 109% relative
    error, because the de-normalisation constants span too wide a range.
    """
    import tensorflow as tf

    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if quantise == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantise == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def run_tflite(model_bytes: bytes, X_nwc: np.ndarray,
               disable_delegates: bool = False) -> np.ndarray:
    """Run a TFLite model on (N, win, C) input, one window at a time."""
    import tensorflow as tf

    kw = {}
    if disable_delegates:
        kw["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp = tf.lite.Interpreter(model_content=model_bytes, **kw)
    interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]

    ys = []
    for i in range(len(X_nwc)):
        interp.set_tensor(inp["index"], X_nwc[i:i + 1].astype(np.float32))
        interp.invoke()
        ys.append(interp.get_tensor(outp["index"]).reshape(-1))
    return np.stack(ys)
