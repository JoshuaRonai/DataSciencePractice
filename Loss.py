from abc import ABC, abstractmethod

class Loss(ABC):
    
    @abstractmethod
    def function(self,y_true,y_pred):
        pass
    
    @abstractmethod
    def derivative(self,y_true,y_pred):
        pass
    
    def __call__(self,y_true,y_pred):
        return self.function(y_true,y_pred)

    
class BinaryCrossEntropy(Loss):
    
    def function(self,y_true,y_pred,eps=1e-9):
        y_pred = np.clip(y_pred, a_min=eps, a_max=1-eps)
        aff = y_true*np.log(y_pred)
        neg = (1-y_true)*np.log(1-y_pred)
        return -1*np.mean(aff+neg)
    
    def derivative(self,y_true,y_pred):
        num = y_true - y_pred
        den = y_pred * (1-y_pred)
        return -1*np.mean(num*np.reciprocal(den))

    
class MeanSquaredError(Loss):
    
    def function(self,y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    
    def derivative(self,y_true,y_pred):
        return 2*np.mean(y_pred-y_true)