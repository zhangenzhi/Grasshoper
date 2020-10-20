import tensorflow as tf
import numpy as np
import sys
sys.path.append('/home/ubuntu_wsl/zhangenzhi/work/Grasshoper')
print(sys.path)
from model.dnn import dnn
from dataset.dataloader import FuncLoader


class Trainer:

    def __init__(self):
        self.dataloader = FuncLoader(name='sin')
        self.model = dnn()

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.loss = tf.keras.losses.mse
    
    def train_step(self):
        pass

    def run(self):
        train_dataset, validation_dataset = self.dataloader.load_data()
        print(train_dataset)

    def save_model(self):
        pass

    def load_model(self):
        pass

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()