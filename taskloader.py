import tensorflow as tf
import pandas as pd
import numpy as np
from dataloader import FuncLoader
from dataset.generator import FunctionGenerator

class FuncWrapper(object):
    def __init__(self, func=None):
        self.func = func
    def wrap(self, params):
        # do somthings you like
        pass
    def __call__(self, params):
        return self.wrap(params)

class SinWrapper(FuncWrapper):
    def __init__(self):
        super(SinWrapper,self).__init__(func=np.sin)
    def wrap(self, params):
        A = params['Amplitude']
        w = params['Frequency']
        b = params['phase']
        return lambda x: A*self.func(w*x+b)
    def __call__(self, params):
        return self.wrap(params)

class TaskLoader(object):

    def __init__(self, param_space):
        # sample n tasks from param space
        self.param_space = param_space
        self.num_tasks = 0
        self.sampled_tasks = []
        self.loaders = []

    def task_sampler(self, num_tasks=10):
        self.num_tasks = num_tasks
        for _ in range(num_tasks):
            tmp_task = {}
            for (param_name,r) in self.param_space.items():
                param = np.random.uniform(r[0],r[1],1)[0]
                tmp_task[param_name] = param
            self.sampled_tasks.append(tmp_task)

    def get_funcloader(self, func_w, f_name):
        # func_w is function wraper： build f by params
        for i in range(self.num_tasks):
            dname = f_name + "_" + str(i)
            task = self.sampled_tasks[i]
            f = func_w(task)
            funcgen = FunctionGenerator(f, radius=[-10,10], name=dname, num_samples= 100)
            funcgen.save()
            funcloader = FuncLoader(name=dname, validation_split=0.2)
            self.loaders.append(funcloader)
        return self.loaders


if __name__ == "__main__":
    sin_space = {"Amplitude":[0,3],"Frequency":[0,6.28],"phase":[0,1.57]}
    # sin_params = {"Amplitude":1,"Frequency":6.28,"phase":0}
    sin_wrap = SinWrapper()
    # f = sin_wrap(sin_params)
    # print(f(3))

    tl = TaskLoader(sin_space)
    tl.task_sampler()
    loaders = tl.get_funcloader(func_w=sin_wrap, f_name="sin")