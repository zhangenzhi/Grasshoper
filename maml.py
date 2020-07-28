import tensorflow as tf
from tensorflow import keras
from taskloader import TaskLoader
from dataloader import FuncLoader

class MAML():
    def __init__(self):
        # 
        # task and data
        sin_space = {"Amplitude":[0,3],"Frequency":[0,6.28],"phase":[0,1.57]}
        sin_wrap = SinWrapper()
        self.tl = TaskLoader(sin_space)
        self.num_tasks = 2

        # train
        self.sub_train_lr = 0.0025
        self.sub_train_epochs = 100
        self.all_train_lr = 0.001
        self.all_train_epochs = 10

    def _build_meta_model(self):
        pass

    def _build_sub_model(self):
        pass
    def call(self,inputs):
        pass

    def all_train(self):
        pass

    def sub_train(self):
        pass
    
    def run(self):
        loaders = self.tl.task_sampler(num_tasks=self.num_tasks)

if __name__ == "__main__":
    dl = FuncLoader(name='sin',batch_size=10)
    train_dataset,validation_dataset = dl.load_data()
    bit_ds = train_dataset.take(1)
    for data,label in bit_ds:
        print(data)
        print(label)


