
import torch
import torch.nn as nn
import torch.utils.data as tud

__all__ = ['BasicModel']
class BasicModel(nn.Module):
    """
    BasicModel
    the default model contains three parts, the default state of model is "train" state
    backbone:
        Seven layers, hidden unit is 512.
        input size * 512,
        512 * 512,
        ...,
        512*512
    classifier:
        one layers, 512 * output_size
    
    Function:

    forward(X)
    reset_batchnorm(X)
    set_backbone(backbone)
    set_classifier(classifier)
    init_parameters()
    set_device(device)


    """
    def __init__(self,input_size=None,output_size=None,hidden_size=512,combine=False,backbone=None,classifier=None,device='cuda'):
        """
        init the network, 7 layers feature exactors and 1 layer classifier and theirs parameters
        
        Args:
            input_size: size of the input
            output_size: size of the output
            hidden_size: size of the hidden unit

            combine: bool
                if combine is True
                the model will combine the provided backbone and classifier
                otherwise 
                the model will be a default state(seven layer backbone and one layer classifier)
            
            device: the device of the model, default 'cuda'
            
        """
        super(BasicModel,self).__init__()
        self.device = device
        if combine:
            self.device = device
            self.set_backbone(backbone)
            self.set_classifier(classifier)
        else:
            self.__init_network(input_size,output_size,hidden_size)
            self.init_parameters()
            self.to(self.device)

    
    def set_backbone(self,backbone):
        """
        set the new inner backbone, the new backbone will automaticlly move to the original device
        backbone: type: nn.Module
        """
        self.backbone = backbone
        self.to(self.device)

    
    def set_classifier(self,classifier):
        """
        set the new inner classifier, the new classifier will automaticlly move to the original device
        classifier: type: nn.Module
        """
        self.classifier = classifier
        self.to(self.device)


    def __init_network(self,input_size,output_size,hidden_size):
        backbone = []
        classifier = []
        for i in range(7):
            if i == 0:
                backbone.append(nn.Linear(input_size,hidden_size))
                backbone.append(nn.BatchNorm1d(hidden_size))
                backbone.append(nn.ReLU())
            else:
                backbone.append(nn.Linear(hidden_size,hidden_size))
                backbone.append(nn.BatchNorm1d(hidden_size))
                backbone.append(nn.ReLU())
        classifier.append(nn.Linear(hidden_size,output_size))
        self.backbone = nn.Sequential(*backbone)
        self.classifier = nn.Sequential(*classifier)


    def init_parameters(self):
        """
        use xavier_normal_ to init the parameters of Linear layer with 'relu' gain
        """
        for m in self.backbone:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))
        for m in self.classifier:
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))


    def set_device(self,device):
        """
        set the new device of the model, the model will automaticlly move to the new device
        """
        self.device = device
        self.to(self.device)


    def forward(self,X):
        """
        pass the X through the model, the input X will automaticlly move to the device of the model
        return output,feature
        feature is the input of classifier layer
        """
        X = X.to(self.device)
        feature = self.backbone(X)
        output = self.classifier(feature)
        return output,feature
    

    def reset_batchnorm(self,dataset,T=10):
        """
        reset the running mean and var of batchnorm layer with dataset T times(batch_size=256)
        dataset: Dataset object 
        T: indicates the times running the whole dataloader, default 3
        this operation will set the state of model to "train" state
        """
        self.train()
        dataloader = tud.DataLoader(dataset,batch_size=256,shuffle=True,drop_last=True)
        for m in self.backbone:
            if isinstance(m,nn.BatchNorm1d):
                m.reset_running_stats()
        for m in self.classifier:
            if isinstance(m,nn.BatchNorm1d):
                m.reset_running_stats()
        with torch.no_grad():
            for i in range(T):
                for x,y in dataloader:
                    self.forward(x)


    def predict_numpy(self,X):
        """
        X: input, type: np.array
        be careful of whether reset the running mean and var of batchnorm layer
        this operation will set the state of the model into "eval" state
        """
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X,dtype=torch.float32)
            return self.forward(X)[0].cpu().numpy()
        

    def get_numpy_feature(self,X):
        """
        X: numpy array
        return feature, type: numpy array
        """
        self.eval()
        X = torch.tensor(X,dtype=torch.float32)
        with torch.no_grad():
            return self.forward(X)[1].cpu().numpy()



        