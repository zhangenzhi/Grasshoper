import tensorflow as tf
import pandas as pd

class DataLoader(object):
    def __init__(self, config):
        pass

    def _build_pipe(self):
        pass

    def load_data(self):
        pass

    def __getitem__(self):
        pass

    def __len__():
        pass

class FuncLoader(DataLoader):
    def __init__(self, name):
        super(DataLoader,self).__init__
        self.filename = "./dataset/function/" + name + ".csv"
        

    def load_data(self):
        df = pd.read_csv(self.filename)
        label = df.pop('Y')
        dataset = tf.data.Dataset.from_tensor_slices((df.values,label.values))

        for feat,targ in dataset.take(5):
            print ('Features: {}, Target: {}'.format(feat, targ)) 

if __name__ == "__main__":
    funcloader = FuncLoader(name="sin")
    funcloader.load_data()
