import torch.nn as nn
import torch.utils.data as tud
import torch


__all__ = ['BasicBackbone']
class BasicBackbone(nn.Module):
    """
    the basicbackbone of the basicmodel
    the default backbone will be a five layer backbone with the same block

    block:
        nn.Linear(input_size,hidden_size)
        nn.BatchNorm1d(hidden_size)
        nn.ReLU()
    

    """
    def __init__(self,input_size,hidden_size=512,device='cuda'):
        super(BasicBackbone,self).__init__()
        self.device = device
        self.__init_backbone(input_size,hidden_size)
        self.init_parameters()
        self.to(self.device)

    
    def __init_backbone(self,input_size,hidden_size):
        backbone = []
        for i in range(5):
            if i == 0:
                backbone.append(nn.Linear(input_size,hidden_size))
                backbone.append(nn.BatchNorm1d(hidden_size))
                backbone.append(nn.ReLU())
            else:
                backbone.append(nn.Linear(hidden_size,hidden_size))
                backbone.append(nn.BatchNorm1d(hidden_size))
                backbone.append(nn.ReLU())
        self.backbone = nn.Sequential(*backbone)

    
    def init_parameters(self):
        """
        init the parameters of the backbone
        """
        for m in self.backbone:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))


    def set_backbone(self,backbone):
        """
        set the inner backbone of the backbone class, it will automatic move the backbone to the original device
        backbone: nn.Module
        """
        self.backbone = backbone
        self.to(self.device)


    def set_device(self,device):
        """
        set the device of the backbone, it will automatic move the backbone to the new device
        """
        self.device = device
        self.to(self.device)


    def forward(self,x):
        """
        pass the input through the backbone, the x will automaticlly move to the device of the backbone
        """
        x = x.to(self.device)
        f = self.backbone(x)
        return f


    def reset_batchnorm(self,dataset,T=10):
        """
        reset the running stats of the batchnorm layer, it will forward the dataset T times to reset the running stats
        """
        self.train()
        dataloader = tud.DataLoader(dataset,batch_size=256,shuffle=True,drop_last=True)
        for m in self.backbone:
            if isinstance(m,nn.BatchNorm1d):
                m.reset_running_stats()
        with torch.no_grad():
            for i in range(T):
                for x,y in dataloader:
                    self.forward(x)

