from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
from .basicmodel import BasicModel


__all__ = ['train','evaluate','train_dann','train_dan','evaluate_bc']
def __basic_train(model,dataloader,optimzer,loss_fn):
    """
    just train the model one run
    """
    model.train()
    loss_total = 0.0
    for i,(x,y) in enumerate(dataloader):
        #inside the  forward function, model will put the x into model.device
        y_pred,_ = model(x)
        loss = loss_fn(y_pred,y.to(model.device))
        loss_total += loss.item()
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
    loss_total /= (i+1)
    return loss_total


def train(model,dataloader,optimzer,loss_fn,outter_time,T=5,print_flag=True,logger=None):
    """
    train the model with the dataloader, optimiizer and loss_fn with T iterations

    loss_fn: function, loss_fn(y_pred,y_true)
    print_flag: whether print the loss information to the screen
    logger: only useful when print_flag=False and then provide a logger instance with a info function
    outter_time: int, start 0, indicates the time you call train() function
    """
    model.train()
    for i in range(T):
        loss = __basic_train(model,dataloader,optimzer,loss_fn)
        if print_flag:
            print(f"{i+T*outter_time} times train loop, with loss {loss}")
        else:
            logger.info(f"{i+T*outter_time} times train loop, with loss {loss}")
        

def evaluate(model,X,y):
    """
    model: instance of BasicModel
    X: np array
    y: np array
    return accuracy_score of model on (X,y)
    """
    y_pred = model.predict_numpy(X)
    y_pred = np.argmax(y_pred,axis=1)
    return accuracy_score(y,y_pred)


def evaluate_bc(backbone,classifier,X,y):
    """
    evaluate the accuracy of backbone and classifier on (X,y)
    make sure the backbone and classifier are in the same device
    """
    assert backbone.device == classifier.device
    model = BasicModel(combine=True,backbone=backbone,classifier=classifier,device=backbone.device)
    return evaluate(model,X,y)



def _train_feature_based(model,loss_module,source_loader,target_loader,optimzer,outter_time,iter_times=5,print_flag=True,logger=None):
    """
    train the model using specify divergence
    """
    model.train()
    loss_module.train()
    loss_module.to(model.device)
    loss_total = 0.0
    for i in range(iter_times):
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)            
        inner_loop_time = 0
        loss_inner_total = 0.0
        while(True):
            try:
                x_s,y_s = next(source_iter)
                x_t,_ = next(target_iter)
                inner_loop_time +=1 
            except StopIteration:
                loss_inner_total /= inner_loop_time
                break
            y_s = y_s.to(model.device)
            y_pred,f_s = model(x_s)
            loss1 = F.cross_entropy(y_pred,y_s)
            _,f_t = model(x_t)
            loss2 = loss_module(f_s,f_t)
            loss = loss1 + loss2
            loss_inner_total += loss.item()  
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        loss_total += loss_inner_total
        if print_flag:
            print(f'{i+iter_times*outter_time} times train loop, with loss {loss_inner_total}')
        else:
            logger.info(f'{i+iter_times*outter_time} times train loop, with loss {loss_inner_total}')

    return (loss_total/i+1)

def train_dann(model,dann_loss,source_loader,target_loader,optimzer,outter_time,iter_times=5,print_flag=True,logger=None):
    """
    trainning the model with js divergence
    """

    _train_feature_based(model,dann_loss,source_loader,target_loader,optimzer,outter_time,iter_times=iter_times,print_flag=print_flag,logger=logger)


def train_dan(model,dan_loss,source_loader,target_loader,optimzer,outter_time,iter_times=5,print_flag=True,logger=None):
    """
    trainning the model with mmd divergence
    """
    _train_feature_based(model,dan_loss,source_loader,target_loader,optimzer,outter_time,iter_times=iter_times,print_flag=print_flag,logger=logger)


def train_mcd():
    pass
            