import numpy as np
from abc import ABC, abstractmethod
from functools import reduce

class NeuralNetwork():
    
    def __init__(self,input_size,loss):
        self.layers = [Input(input_size)]
        self.loss = loss
        
    def add(self,layer):
        """Add layers"""
        layer.comp(self.layers[-1])
        self.layers.append(layer)
    
    def get_hyperparameters(self):
        w,b = zip(*[layer.get_hyperparameters() for layer in self.layers[1:]])
        return w,b
    
    def forward(self,x):
        """Feed forward through the network"""
        h = []
        z = []
        h_l = x
        for layer in self.layers:
            h_l,z_l = layer.forward(h_l)
            h.append(h_l)
            z.append(z_l)
        return h,z
    
    def loss(self,y_true,y_pred):
        return self.loss(y_true,y_pred)
    
    def backward(self,y_true,y_pred,z,x):
        """Implements the back propogation"""
        dws,dbs = [], []
        loss_gradient = self.loss.derivative(y_true,y_pred)
        loss_gradient *= np.eye(len(y_true))
        for l in range(len(self.layers)-1,0,-1):
            layer,z_l,x_l = self.layers[l],z[l-1],x[l-1]
            dw,db = layer.backward(z_l,x_l)
            W,b = layer.get_hyperparameters()
            dbs.append(loss_gradient @ db)
            dws.append(loss_gradient @ dw)
            loss_gradient = (loss_gradient * db.T) @ W
        return dws,dbs
    
    def gradient_update(self,dws,dbs,lr):
        """Calculate the gradients of all layers"""
        for i,(dw,db) in enumerate(zip(dws,dbs)):
            self.layers[-i-1].gradient_update(dw,db,lr)
    
    def predict(self,x):
        """Feed forward through the network"""
        return reduce(lambda x,y: y.forward(x)[0], [x]+self.layers)
    
    def __call__(self,x):
        return self.predict(x)