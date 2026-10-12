"""Model definitions: baseline CNN, proposed CNN (BN + SE attention) and comparison models."""
from tensorflow.keras import layers, models, regularizers


def se_block(x, ratio=8):
    """Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018)."""
    ch = x.shape[-1]
    s = layers.GlobalAveragePooling2D()(x)
    s = layers.Dense(max(ch // ratio, 1), activation="relu")(s)
    s = layers.Dense(ch, activation="sigmoid")(s)
    s = layers.Reshape((1, 1, ch))(s)
    return layers.Multiply()([x, s])


def _conv_block(x, filters, bn, se):
    x = layers.Conv2D(filters, (3, 3), use_bias=not bn)(x)
    if bn:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if se:
        x = se_block(x)
    return layers.MaxPooling2D((2, 2))(x)


def build_cnn(bn=False, se=False, dropout=0.3, l2=1e-3, name="cnn"):
    """bn=False, se=False is exactly the baseline CNN from the notebook (225,034 params)."""
    inp = layers.Input(shape=(28, 28, 1))
    x = _conv_block(inp, 32, bn, se)
    x = _conv_block(x, 64, bn, se)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(10, activation="softmax")(x)
    return models.Model(inp, out, name=name)


def build_lenet5(name="lenet5"):
    """LeNet-5 (LeCun et al., 1998) in its original tanh / average-pooling form."""
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(6, 5, padding="same", activation="tanh"),
        layers.AveragePooling2D(2),
        layers.Conv2D(16, 5, activation="tanh"),
        layers.AveragePooling2D(2),
        layers.Flatten(),
        layers.Dense(120, activation="tanh"),
        layers.Dense(84, activation="tanh"),
        layers.Dense(10, activation="softmax"),
    ], name=name)


def build_vgg_small(dropout=0.3, name="vgg_small"):
    """Small VGG-style CNN (stacked 3x3 convolutions, Simonyan & Zisserman, 2015)."""
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(10, activation="softmax"),
    ], name=name)
