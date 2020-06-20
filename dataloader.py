import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
# import pandas as pd

def get_transform(phase, resize=0, method=Image.BILINEAR):
    transform_list = []
    if resize > 0:
        size = [resize, resize]
        transform_list.append(transforms.Resize(size, method))
    # transform_list.append(transforms.ToPILImage())    
    transform_list.append(transforms.Grayscale(num_output_channels=1))
    if phase == 'train' :
        transform_list.append(transforms.RandomHorizontalFlip())
        #transform_list.append(transforms.RandomVerticalFlip())
        transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0))
        transform_list.append(transforms.RandomAffine(10, translate=None, scale=None, shear=None, resample=False, fillcolor=0))
        transform_list.append(transforms.RandomResizedCrop(384, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), ))
        #transform_list.append(transforms.RandomCrop(346, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'))
        transform_list.append(transforms.RandomRotation(10, resample=False, expand=False, center=None, fill=None))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.485,), (0.229,)))
    return transforms.Compose(transform_list)

class CustomDataset(data.Dataset):
    def __init__(self, root, phase='train'):
        self.root = root
        self.phase = phase
        self.labels = {}

        self.label_path = os.path.join(root, self.phase, self.phase+'_label_COVID.txt')
        with open(self.label_path, 'r') as f:
            file_list = []
            label_list = []
            for line in f.readlines()[0:]:
                v = line.strip().split()
                file_list.append(v[0])
                if self.phase != 'test' :
                    label_list.append(v[1])                

        self.labels['file'] = list(file_list)
        self.labels['label'] = list(label_list)

    def __getitem__(self, index):
        #if self.phase == 'train':
        image_path = os.path.join(self.root, self.phase, self.labels['file'][index])
        
        if self.phase != 'test' :
            is_label = self.labels['label'][index]
            is_label = torch.tensor(int(is_label))

        transform = get_transform(self.phase)
        image = Image.open(image_path).convert('RGB')
        image = transform(image)

        if self.phase != 'test' :
            return (self.labels['file'][index], image, is_label)
        elif self.phase == 'test' :
            dummy = ""
            return (self.labels['file'][index], image, dummy)


    def __len__(self):
        return len(self.labels['file'])

    def get_label_file(self):
        return self.label_path

def data_loader(root, phase='train', batch_size = 16):
    if phase == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = CustomDataset(root, phase)
    dataloader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.get_label_file()