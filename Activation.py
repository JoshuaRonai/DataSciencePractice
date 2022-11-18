from abc import ABC, abstractmethod

class Activation(ABC):
    
    @abstractmethod
    def function(self,x):
        pass
    
    @abstractmethod
    def derivative(self,x):
        pass
    
    def __call__(self,x):
        return self.function(x)
    
    
class LinearActivation(Activation):
    
    def function(self,x):
        return x
    
    def derivative(self,x):
        return np.repeat(1,x.shape)
    
    
class Sigmoid(Activation):
    
    def function(self,x):
        return np.reciprocal(1+np.exp(-1*x))
    
    def derivative(self,x):
        val = self.function(x)
        return val * (1-val)

    
class Tanh(Sigmoid):
    
    def function(self,x):
        return 2*super().function(2*x)-1
    
    def derivative(self,x):
        return 1-self.function(x)**2

    
class ReLU(Activation):
    
    def function(self,x):
        return np.max(x,0)
    
    def derivative(self,x):
        return np.where(x>0,1,0)