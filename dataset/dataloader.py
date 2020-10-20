import tensorflow as tf
import pandas as pd

class DataLoader(object):
    def __init__(self, config):
        self.size = 0 

    def _build_pipe(self):
        pass

    def load_data(self):
        pass

    def __len__(self):
        return self.size

class FuncLoader(DataLoader):
    def __init__(self, name, validation_split=0.2,batch_size = 128):
        super(DataLoader,self).__init__
        self.filename = "./dataset/function/" + name + ".csv"
        self.validation_split = validation_split
        self.batch_size = batch_size

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        

    def load_data(self):
        df = pd.read_csv(self.filename)
        self.size = len(df['Y'])
        label = df.pop('Y')

        split_idx = int(self.size*self.validation_split)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((df.values[split_idx:],label.values[split_idx:]))
        self.train_dataset = self.train_dataset.shuffle(self.size-split_idx).batch(self.batch_size)
        self.validation_dataset = tf.data.Dataset.from_tensor_slices((df.values[:split_idx],label.values[:split_idx]))
        self.validation_dataset = self.validation_dataset.shuffle(split_idx).batch(self.batch_size)
        return self.train_dataset,self.validation_dataset
    


if __name__ == "__main__":
    funcloader = FuncLoader(name="sin")
    funcloader.load_data()
