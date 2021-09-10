import torch
import torch.utils.data as tud

__all__ = ['TorchDataset','torch_dataset_to_dataloader']
class TorchDataset:
    """
    transform the numpy array into dataset object
    """
    def __init__(self,X,y):
        self.X = torch.tensor(X,dtype=torch.float32)
        self.y = torch.LongTensor(y)

    def __getitem__(self,index):
        return self.X[index],self.y[index]

    def __len__(self):
        return len(self.y)


def torch_dataset_to_dataloader(dataset,batch_size=32):
    """
    get the dataloader of dataset, default batch_size=32
    """
    return tud.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)