import numpy as np
import csv
import plotly.graph_objects as go
# generate csv dataset

class Generator(object):
    def __init__(self):
        self.num_samples = 100000
        self.type = None #function or others
    
    def _build():
        pass

    def save(self):
        pass

    def draw(self):
        pass

class FunctionGenerator(Generator):
    #x-y function
    def __init__(self, f, radius, name):
        super(FunctionGenerator,self).__init__()
        self.type = "function"
        self.num_samples = 10000

        self.f = f
        self.radius = radius
        self.filename = name + '.csv'
        self.path = "./dataset/function/"

        self.x = []
        self.y = []

        self._build()

    def _build(self):
        #range
        low=self.radius[0]
        high=self.radius[1]
        size=self.num_samples

        self.x = np.random.uniform(low,high,size)
        self.y = f(self.x)
            
    def save(self):
        with open(self.path+self.filename,mode='w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["X","Y"])
            for i in range(self.num_samples):
                writer.writerow([self.x[i],self.y[i]])

    def draw(self):
        # if can
        fig = go.Figure(data=go.Scatter(x=self.x, y=self.y,mode="markers"))
        fig.show()  
        
if __name__ == "__main__":
    f = np.sin
    gen = FunctionGenerator(f=f,radius=[-10,10],name="sin")
    gen.draw()
    # gen.save()