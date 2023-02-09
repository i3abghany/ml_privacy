import ml_privacy_meter
import tensorflow as tf
import tensorflow.compat.v1.keras.layers as keraslayers
import qkeras as qkeraslayers
import numpy as np
from sklearn.model_selection import train_test_split

def get_fcn():
    initializer = tf.compat.v1.keras.initializers.random_normal(0.0, 0.01)
    model = tf.compat.v1.keras.Sequential(
        [
            qkeraslayers.QDense(
                1024,
                input_shape=(600,),
                kernel_initializer=initializer,
                kernel_quantizer=qkeraslayers.quantized_bits(16),
                bias_quantizer=qkeraslayers.quantized_bits(16),
                bias_initializer='zeros'
            ),
            qkeraslayers.QActivation('quantized_tanh(16)'),

            qkeraslayers.QDense(
                512,
                kernel_initializer=initializer,
                kernel_quantizer=qkeraslayers.quantized_bits(16),
                bias_quantizer=qkeraslayers.quantized_bits(16),
                bias_initializer='zeros'
            ),
            qkeraslayers.QActivation('quantized_tanh(16)'),

            qkeraslayers.QDense(
                256,
                kernel_initializer=initializer,
                kernel_quantizer=qkeraslayers.quantized_bits(16),
                bias_quantizer=qkeraslayers.quantized_bits(16),
                bias_initializer='zeros'
            ),
            qkeraslayers.QActivation('quantized_tanh(16)'),

            qkeraslayers.QDense(
                128,
                kernel_initializer=initializer,
                kernel_quantizer=qkeraslayers.quantized_bits(16),
                bias_quantizer=qkeraslayers.quantized_bits(16),
                bias_initializer='zeros'
            ),
            qkeraslayers.QActivation('quantized_tanh(16)'),

            qkeraslayers.QDense(
                100,
                kernel_initializer=initializer,
                kernel_quantizer=qkeraslayers.quantized_bits(16),
                bias_quantizer=qkeraslayers.quantized_bits(16),
                bias_initializer='zeros'
            )
        ]
    )
    return model

def get_purchase_dataset():
    # YOU WILL HAVE TO CLONE THE FOLLOWING REPOSITORY FIRST: https://github.com/xehartnort/Purchase100-Texas100-datasets'
    filename = 'datasets/purchase100.npz'
    data = np.load(filename)
    x, y = data['features'], data['labels']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1234)
    return x_train, y_train, x_test, y_test, (600, ), 100

model = get_fcn()
x_train, y_train, x_test, y_test, input_shape, num_classes = get_purchase_dataset()

num_datapoints = 5000
x_target_train, y_target_train = x_train[:num_datapoints], y_train[:num_datapoints]

# population data (training data is a subset of this)
x_population = np.concatenate((x_train, x_test))
y_population = np.concatenate((y_train, y_test))

print(model)
model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.fit(x_train, y_train, epochs=1)


datahandlerA = ml_privacy_meter.utils.attack_data.AttackData(x_population=x_population,
                                                             y_population=y_population,
                                                             x_target_train=x_target_train,
                                                             y_target_train=y_target_train,
                                                             batch_size=64,
                                                             attack_percentage=50,
                                                             input_shape=(600,))

attackobj = ml_privacy_meter.attack.meminf.initialize(
    target_train_model=model,
    target_attack_model=model,
    learning_rate=0.0001, optimizer='adam',
    train_datahandler=datahandlerA,
    attack_datahandler=datahandlerA,
    layers_to_exploit=[5],
    gradients_to_exploit=[3, 4, 5])
attackobj.train_attack()
