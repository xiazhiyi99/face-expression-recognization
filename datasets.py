import torch
import torchvision.transforms as trans
import torch.utils.data as datautils
import json
import pathlib
import cv2
import PIL
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

transform = trans.Compose([
    trans.RandomHorizontalFlip(0.5),
    trans.RandomResizedCrop((224,224),scale=(0.9,1.0)),
    #trans.Resize((224,224)),
    trans.ColorJitter(brightness=10),
    trans.ColorJitter(contrast=10),
    trans.ColorJitter(saturation=10),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

transform_test = trans.Compose([
    trans.Resize((224,224)),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

class AffectNetDataset(datautils.Dataset):
    def __init__(self):
        f = open('./data/AffectNet/output/labels.json')
        self.labels = json.load(f)
        f.close()

        self.datapath = pathlib.Path('./data/AffectNet/output/data')
        self.datalist = [x for x in self.datapath.glob('./*jpg')]
    
    def __getitem__(self, idx):
        img = PIL.Image.open(str(self.datalist[idx]))
        img = transform(img)
        label = self.labels[self.datalist[idx].name]
        return img, label
    
    def __len__(self):
        return len(self.datalist)

class RAFDBDataset(datautils.Dataset):
    def __init__(self, split='train'):
        self.split = split
        f = open('./data/RAF-DB/EmoLabel/list_patition_label.txt')
        self.labels = [line.strip('\n').split(' ') for line in f.readlines()]
        self.labels = {k:int(v) for [k,v] in self.labels}
        f.close()
        self.datapath = pathlib.Path('/data/xzy/face-emotion/data/RAF-DB/Image/original')
        self.datalist = [x for x in self.datapath.glob('./%s*jpg'%split)]
    
    def __getitem__(self, idx):
        img = PIL.Image.open(str(self.datalist[idx]))
        if self.split=="train":
            img = transform(img)
        else:
            img = transform_test(img)
        label = self.labels[self.datalist[idx].name]
        return img, label
    
    def __len__(self):
        return len(self.datalist)

    def __get_sample_weight(self):
        cnt = {}
        weight = []
        for d in self.datalist:
            v = self.labels[d.name]
            if cnt.get(v) is None:
                cnt[v] = 0
            cnt[v] += 1
        for d in self.datalist:
            v = self.labels[d.name]
            weight.append(len(self.datalist) / cnt[v])
        return weight

    def get_sampler(self):
        sample_weights = self.__get_sample_weight()
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            True
        )
        return sampler


def get_loader(setname="affectnet", traindata_ratio=0.5, use_sampler=True, **kwargs):
    dataset_book = {"affectnet":AffectNetDataset, "raf-db":RAFDBDataset}
    if setname == "raf-db":
        trainset = RAFDBDataset('train')
        trainsampler = trainset.get_sampler() if use_sampler else None
        testset = RAFDBDataset('test')
        testsampler = testset.get_sampler() if use_sampler else None
        trainloader = datautils.DataLoader(trainset, sampler=trainsampler, **kwargs) 
        testloader = datautils.DataLoader(testset,  sampler=testsampler, **kwargs)
        return trainloader, testloader
    elif setname == "affectnet":
        dataset = dataset_book[setname]()
        train_size = int(len(dataset) * traindata_ratio)
        val_size = len(dataset) -  train_size
        trainset, testset = datautils.random_split(dataset, [train_size, val_size])
        trainloader, testloader = datautils.DataLoader(trainset, **kwargs), datautils.DataLoader(testset, **kwargs)
        return trainloader, testloader

if __name__=="__main__":
    import tqdm
    train_loader, test_loader = get_loader(setname='raf-db',batch_size=1, num_workers=4)
    cnt = np.zeros(10)
    pbar = tqdm.tqdm(train_loader)
    for data, label in pbar:
        cnt[label] += 1
    print(cnt)