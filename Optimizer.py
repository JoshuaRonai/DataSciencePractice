import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    
    def __init__(self,model,lr,epochs):
        self.model = model
        self.lr = lr
        self.epochs = epochs
        
    def get_model(self):
        return self.model
    
    @abstractmethod
    def make_gradients(self,dws,dbs,t):
        pass
    
    @abstractmethod
    def make_lr(self,t):
        pass
    
    @abstractmethod
    def train_step(self,X,y):
        pass
    
    @abstractmethod
    def train(self,X,y):
        pass
    

class MiniBatchGD(Optimizer):
    
    def __init__(self,model,lr,epochs,batch_size):
        super().__init__(model,lr,epochs)
        assert batch_size > 0 and type(batch_size) == int
        self.batch_size = batch_size
        
    def make_gradients(self,dws,dbs,t):
        return dws,dbs
    
    def make_lr(self,t):
        return self.lr
    
    def train_step(self,x,y,t):
        h,z = self.model.forward(x)
        h,y_pred = [h_l.sum(axis=1).reshape((h_l.shape[0],1)) for h_l in h[:-1]], h[-1]
        z = [z_l.sum(axis=1).reshape((z_l.shape[0],1)) for z_l in z]
        y = y.reshape(y_pred.shape)
        loss = self.model.loss(y,y_pred)
        dws,dbs = self.model.backward(y,y_pred,z[1:],h)
        dws,dbs = self.make_gradients(dws,dbs,t)
        lr = self.make_lr(t)
        self.model.gradient_update(dws,dbs,lr)
        return loss
    
    def train(self,X_train,Y_train,X_test,Y_test,print_by=20):
        train_losses = []
        test_losses = []
        N = len(Y_train)
        for epoch in range(1,self.epochs+1):
            inds = np.arange(N)
            np.random.shuffle(inds)
            train_loss, sub_n = 0, 0
            num_eqs = 0
            preamble = "Epoch "+str(epoch)+"/"+str(self.epochs)
            num_rows = " ("+str(sub_n)+"/"+str(N)+") "
            load_bar = "[" + ("="*num_eqs) + (">" if num_eqs<print_by else "") + " "*(print_by-1-num_eqs) + "] "
            loss_print = "train_loss - "+"{:.4f}".format(train_loss)
            print("\r"+preamble+num_rows+load_bar+loss_print,end="")
            while len(inds) > 0:
                train_inds, inds = inds[:self.batch_size], inds[self.batch_size:]
                x = X_train[train_inds,:]
                y = Y_train[train_inds]
                loss = self.train_step(x.T,y,epoch)
                n = len(train_inds)
                train_loss = (train_loss*sub_n + loss*n)/(sub_n+n)
                sub_n += n
                if (sub_n*print_by)//N - num_eqs > 0:
                    num_eqs += 1
                    num_rows = " ("+str(sub_n)+"/"+str(N)+") "
                    load_bar = "[" + ("="*num_eqs) + (">" if num_eqs<print_by else "") + " "*(print_by-num_eqs) + "] "
                    loss_print = "train_loss - "+"{:.4f}".format(train_loss)
                    print("\r"+preamble+num_rows+load_bar+loss_print,end="")
                    
            test_pred = model.forward(X_test.T)
            y_pred = test_pred[0][-1]
            y_test = Y_test.reshape(y_pred.shape)
            test_loss = self.model.loss(y_test,y_pred)
            test_print = " test_loss - "+"{:.4f}".format(test_loss)
            print("\r"+preamble+num_rows+load_bar+loss_print+test_print)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        return {"Train Loss": train_losses, "Test Loss": test_losses}

        
class StochasticGD(MiniBatchGD):
    
    def __init__(self,model,lr,epochs):
        super().__init__(model,lr,epochs,1)