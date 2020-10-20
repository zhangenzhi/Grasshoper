import tensorflow as tf
import numpy as np

class dnn(tf.keras.Model):

    def __init__(self, hidden_dims=[64, 64]):
        super(dnn, self).__init__()
        self.hidden_dims = hidden_dims
        self.deep_layers = self.get_deep_layers()

    def get_deep_layers(self):
        layer_weights = []
        input_shape = 1
        for units in self.hidden_dims:
            w = self.add_weight(name="w_{}".format(units),
                                shape=[input_shape, units],
                                initializer='random_normal',
                                trainable=True)
            b = self.add_weight(name="b_{}".format(units),
                                shape=[units],
                                initializer='random_normal',
                                trainable=True)
            input_shape = units
            layer_weights.append([w,b])
        return layer_weights


    def call(self, inputs):

        result = inputs
        for var in self.deep_layers:
            result = tf.keras.activations.relu(tf.matmul(result, var[0]) + var[1])
        return result

if __name__ == "__main__":
    x = tf.constant([1.0, 2.0],shape=[2,1])
    model = dnn()
    trainable_count = [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    print(trainable_count)
    # print(model(x))
        
        
