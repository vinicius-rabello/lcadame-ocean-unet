import numpy as np
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
    Concatenate,
    Cropping2D
)

### Define Data-driven architecture ######
def stn(input_shape):
    inputs = Input(shape=input_shape)

    padd = ZeroPadding2D(((1, 1), (0, 0)))(inputs)

    layer1_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(padd)
    layer2_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer1_conv
    )
    layer3_pool = MaxPooling2D(pool_size=(2, 2))(layer2_conv)

    layer4_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer3_pool
    )
    layer5_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer4_conv
    )
    layer6_pool = MaxPooling2D(pool_size=(2, 2))(layer5_conv)

    layer7_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer6_pool
    )
    layer8_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer7_conv
    )

    layer10_up = Concatenate(axis=-1)(
        [
            Conv2D(32, (2, 2), activation="relu", padding="same")(
                UpSampling2D(size=(2, 2))(layer8_conv)
            ),
            layer5_conv,
        ]
    )
    layer11_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer10_up
    )
    layer12_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer11_conv
    )

    layer14_up = Concatenate(axis=-1)(
        [
            Conv2D(32, (2, 2), activation="relu", padding="same")(
                UpSampling2D(size=(2, 2))(layer12_conv)
            ),
            layer2_conv,
        ]
    )
    layer15_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer14_up
    )
    layer16_conv = Conv2D(32, (5, 5), activation="relu", padding="same")(
        layer15_conv
    )

    cropped_outputs = Cropping2D(((1, 1), (0, 0)))(layer16_conv)
    outputs = Conv2D(2, (5, 5), activation="linear", padding="same")(cropped_outputs)

    model = Model(inputs, outputs)

    return model




model = stn(input_shape=(46, 68, 2))
model.compile(optimizer=Adam(), loss="mean_squared_error", metrics=["accuracy"])
model.summary()
#model.load_weights("G_46_68.weights.h5")
#mean_ = np.array([0.00025023, -0.00024681])
#var_ = np.array([9.9323115, 0.18261143])