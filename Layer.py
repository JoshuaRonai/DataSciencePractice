import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    
    def __init__(self,num_neurons,activation,weights,biases):
        self.num_neurons = num_neurons
        self.activation = activation
        self.weights = weights
        self.biases = biases
        
    def get_num_neurons(self):
        return self.num_neurons
    
    def get_hyperparameters(self):
        return self.weights,self.biases
        
    def forward(self,x):
        z = self.weights @ x + self.biases
        h = self.activation(z)
        return h,z
    
    @abstractmethod
    def backward(self,z,x):
        pass
    
    def gradient_update(self,dw,db,lr):
        self.weights -= lr * dw
        self.biases -= lr * db
        
    def __call__(self,x):
        assert type(x) == np.ndarray
        return self.forward(x)
    

class Input(Layer):
    
    def __init__(self,num_neurons):
        activation = LinearActivation()
        weights = np.eye(num_neurons)
        biases = np.zeros((num_neurons,1))
        super().__init__(num_neurons,activation,weights,biases)
        
    def backward(self,z,x):
        dw = np.zeros((self.num_neurons)*2)
        db = np.zeros((self.num_neurons,1))
        return dw, db


class Dense(Layer):
    
    def __init__(self,num_neurons,activation):
        biases = np.random.uniform(-0.1,0.1,(num_neurons,1))
        super().__init__(num_neurons,activation,None,biases)
        
    def comp(self,other):
        self.weights = np.random.uniform(-0.1,0.1,(self.num_neurons,other.get_num_neurons()))
    
    def backward(self,z,x):
        db = self.activation.derivative(z)
        dw = db @ x.T
        return dw, db