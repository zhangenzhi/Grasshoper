import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from tensorflow import keras
from dataloader import FuncLoader

class DNN(keras.Model):
    def __init__(self):
        super(DNN,self).__init__()

        self.l1 = keras.layers.Dense(units=32,activation=keras.activations.relu)
        self.l2 = keras.layers.Dense(units=16,activation=keras.activations.relu)
        self.outputs = keras.layers.Dense(units=1,activation=keras.activations.tanh)

    def call(self,inputs):
        x = inputs
        x = self.l1(x)
        x = self.l2(x)
        x = self.outputs(x)
        return x

def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='tanh')
    ])

    model.compile(optimizer='adam',
                    loss='mse',
                    metrics=['mae'])
    return model

if __name__ == "__main__":
    dl = FuncLoader(name='sin')
    train_dataset,validation_dataset = dl.load_data()
    model = DNN()
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])
    model.fit(train_dataset, epochs=150,validation_data=validation_dataset)
    test_x = np.linspace(-20,20,100)
    res = model.predict(test_x)
    res = tf.squeeze(res,1)
    ground_truth = np.sin(test_x)

    fig = go.Figure(data=go.Scatter(x=test_x, y=res,mode="markers"))
    fig.show()  

    # dnn = DNN()
    # dnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #             loss=keras.losses.mse,
    #             metrics=[keras.metrics.mae])
    # history = dnn.fit(dataset,epochs=3)


        