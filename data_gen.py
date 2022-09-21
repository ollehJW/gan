import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2

transform_dict = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([480, 480]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([480, 480]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
}

class ConstructDataset(Dataset):
    """
    Construct pytorch Dataset from file list.
    Parameters
    ----------
    file_list : list
        image file list
    phase : str
        train phase. (Default: 'train')
    Returns
    --------
    pytorch Dataset
    """

    def __init__(self, file_list, phase = 'train'):
        self.file_list = file_list
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        image = cv2.imread(file_name, 1)
        image = transform_dict[self.phase](image)
        return image, 1

class dataset_generator(object):
    """
    Construct pytorch DataLoader from file list.
    Parameters
    ----------
    file_list : list
        image file list
    batch_size : int
        batch size. (Default: 16) 
    Returns
    --------
    pytorch DataLoader
    """
    def __init__(self, file_list, batch_size = 16):
        self.file_list = file_list
        self.batch_size = batch_size

    def dataloader(self):
        train_dataset = ConstructDataset(self.file_list, phase = 'train')
        return dict({'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)})
        
