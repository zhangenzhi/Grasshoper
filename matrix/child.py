import tensorflow as tf

class Child(keras.Model):

    def __init__(self, matrix=None):

        #bind
        self.bind = matrix

        #prefix
        self.l1 = keras.layers.Dense(units=32,activation=keras.activations.relu)
        self.l2 = keras.layers.Dense(units=16,activation=keras.activations.relu)
        self.outputs = keras.layers.Dense(units=1,activation=keras.activations.tanh)

    def _build_refine(self):
        self.var = self.bind.cast()

    def call(self,inputs):
        x = inputs
        x = self.l1(x)
        x = self.l2(x)
        x = self.refine(inputs=x) 
        x = self.outputs(x)
        return x

    def refine(self, inputs):
        res = tf.matmul(inputs,self.var["w"]) + self.var["b"]
        return inputs

    def update(self):
        pass

if __name__ == "__main__":

    child = Child()
    child.forward(inputs)