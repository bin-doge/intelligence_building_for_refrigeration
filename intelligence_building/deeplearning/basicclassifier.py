import torch
import torch.nn as nn
import torch.utils.data as tud


__all__ = ['BasicClassifier']
class BasicClassifier(nn.Module):
    """
    get the basic classifier of the basic model
    the default classifier will be a three-layer classifier with three same block

    block:
        nn.Linear(hidden_size,hidden_size)
        nn.BatchNorm1d(hidden_size)
        nn.ReLU()
    """
    def __init__(self,output_size,hidden_size,device='cuda'):
        super(BasicClassifier,self).__init__()
        self.device = device
        self.__init_classifier(output_size,hidden_size)
        self.init_parameters()
        self.to(self.device)

    
    def __init_classifier(self,output_size,hidden_size):
        classifier = []
        for i in range(2):
            classifier.append(nn.Linear(hidden_size,hidden_size))
            classifier.append(nn.BatchNorm1d(hidden_size))
            classifier.append(nn.ReLU())
        classifier.append(nn.Linear(hidden_size,output_size))
        self.classifier = nn.Sequential(*classifier)


    def init_parameters(self):
        """
        init the parameters of the classifier
        """
        for m in self.classifier:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,nn.init.calculate_gain('relu'))

    
    def set_device(self,device):
        """
        set the device of the classifier, the classifier will automaticlly move to the new device
        """
        self.device = device
        self.to(self.device)


    def set_classifier(self,classifier):
        """
        set the new inner classifier of the classifier class, the new classifier will automaticlly move to the original device
        classifier: nn.Module
        """
        self.classifier = classifier
        self.to(self.device)


    def forward(self,x):
        """
        pass the input x through the classifier, the input x will automaticlly move to the device of the classifier
        """
        x = x.to(self.device)
        output = self.classifier(x)
        return output


    def reset_batchnorm(self,dataset,T=10):
        """
        reset the running stats of the batchnorm layer with dataset T times
        """
        self.train()
        dataloader = tud.DataLoader(dataset,batch_size=256,shuffle=True,drop_last=True)
        for m in self.classifier:
            if isinstance(m,nn.BatchNorm1d):
                m.reset_running_stats()
        with torch.no_grad():
            for i in range(T):
                for x,y in dataloader:
                    self.forward(x)

    
