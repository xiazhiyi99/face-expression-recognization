import torch
import torchvision.transforms as trans
import torch.utils.data as datautils
import json
import pathlib
import cv2
import PIL
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler
import random

transform_ = trans.Compose([
    trans.RandomHorizontalFlip(0.5),
    #trans.RandomResizedCrop((224,224),scale=(0.98,1.0)),
    trans.Resize((224,224)),
    #trans.ColorJitter(brightness=1),
    #trans.ColorJitter(contrast=1),
    #trans.ColorJitter(saturation=1),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
])

transform = trans.Compose([
    trans.RandomHorizontalFlip(0.5),
    #trans.RandomResizedCrop((224,224),scale=(0.98,1.0)),
    trans.Resize((224,224)),
    trans.ColorJitter(brightness=0.3),
    trans.ColorJitter(contrast=0.3),
    trans.ColorJitter(saturation=0.1),
    trans.RandomRotation(10, resample=False, expand=False, center=None),
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
        f = open('/data/xzy/face-expression-recognization/data/AffectNet/output/labels.json')
        self.labels = json.load(f)
        f.close()

        self.datapath = pathlib.Path('/data/xzy/face-expression-recognization/data/AffectNet/output/data')
        self.datalist = [x for x in self.datapath.glob('./*jpg')]
    
    def __getitem__(self, idx):
        img = PIL.Image.open(str(self.datalist[idx]))
        img = transform(img)
        label = self.labels[self.datalist[idx].name]
        return img, label
    
    def __len__(self):
        return len(self.datalist)

class AffectNetDataset7(datautils.Dataset):
    def __init__(self, split="train"):
        f = open('/data/xzy/face-expression-recognization/data/AffectNet7/labels.json')
        self.labels = json.load(f)
        f.close()

        self.datapath = pathlib.Path('/data/xzy/face-expression-recognization/data/AffectNet7/output/data')
        self.datalist = [x for x in self.datapath.glob('./%s*jpg'%split)]
    
    def __getitem__(self, idx):
        img = PIL.Image.open(str(self.datalist[idx]))
        img = transform(img)
        label = self.labels[self.datalist[idx].name]
        return img, label
    
    def __len__(self):
        return len(self.datalist)

class AffectNetDataset7Balanced(datautils.Dataset):
    def __init__(self, split="train"):
        f = open('/data/xzy/face-expression-recognization/data/AffectNet7/labels.json')
        self.labels = json.load(f)
        f.close()

        self.datapath = pathlib.Path('/data/xzy/face-expression-recognization/data/AffectNet7/output/data')
        self.datalist = [x for x in self.datapath.glob('./%s*jpg'%split)]

        balanced_list = []
        cnt = np.zeros((8))
        random.shuffle(self.datalist)
        for d in self.datalist:
            if cnt[self.labels[d.name]]<4500:
                cnt[self.labels[d.name]] += 1
                balanced_list.append(d)
        #print(cnt)
        self.datalist = balanced_list
    
    def __getitem__(self, idx):
        img = PIL.Image.open(str(self.datalist[idx]))
        img = transform(img)
        label = self.labels[self.datalist[idx].name]
        return img, label
    
    def __len__(self):
        return len(self.datalist)

class AffectNetDataset5Balanced(datautils.Dataset):
    """
    7: {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger"}
            ||
            ||
            \/
    5: {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Panic(3:Surprise, 4:Fear)", 4:"Hatred(5:Disgust, 6:Anger)"}
    
    """
    def __init__(self, split="train"):
        f = open('/data/xzy/face-expression-recognization/data/AffectNet7/labels.json')
        self.labels = json.load(f)
        f.close()
        for (k,v) in [(k,v) for k,v in self.labels.items()]:
            if v == 4:
                self.labels[k] = 3
            elif v == 5 or v == 6:
                self.labels[k] = 4

        self.datapath = pathlib.Path('/data/xzy/face-expression-recognization/data/AffectNet7/output/data')
        self.datalist = [x for x in self.datapath.glob('./%s*jpg'%split)]

        balanced_list = []
        cnt = np.zeros((8))
        random.shuffle(self.datalist)
        for d in self.datalist:
            if cnt[self.labels[d.name]]<19600:
                cnt[self.labels[d.name]] += 1
                balanced_list.append(d)
        self.datalist = balanced_list
    
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
        self.datapath = pathlib.Path('./data/RAF-DB/Image/original')
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

class AffectNetDataset7RAFDBDataset7Balanced(datautils.Dataset):
    def __init__(self, split="train"):
        self.split = split
        f = open('/data/xzy/face-expression-recognization/data/AffectNet7/labels.json')
        self.labels = json.load(f)
        f.close()

        self.datapath = pathlib.Path('/data/xzy/face-expression-recognization/data/AffectNet7/output/data')
        self.datalist = [x for x in self.datapath.glob('./%s*jpg'%split)]
        
        transfer = {1:3, 2:4, 3:5, 4:1, 5:2, 6:6, 7:0}
        f = open('/data/xzy/face-expression-recognization/data/RAF-DB/EmoLabel/list_patition_label.txt')
        label = [line.strip('\n').split(' ') for line in f.readlines()]
        for [k,v] in label:
            self.labels[k] = transfer[int(v)]
        f.close()

        self.datapath = pathlib.Path('/data/xzy/face-expression-recognization/data/RAF-DB/Image/original')
        if split=="val":
            rafsplit="test"
        else:
            rafsplit="train"
        for x in self.datapath.glob('./%s*jpg'%rafsplit):
            self.datalist.append(x)

        if split == "train":
            num_restrict =  4396
        else:
            num_restrict = 555
        balanced_list = []
        cnt = np.zeros((8))
        random.shuffle(self.datalist)
        for d in self.datalist:
            if cnt[self.labels[d.name]]<num_restrict:
                cnt[self.labels[d.name]] += 1
                balanced_list.append(d)
        print(cnt)
        self.datalist = balanced_list 

    
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

def get_loader(setname="affectnet", traindata_ratio=0.5, use_sampler=True, **kwargs):
    dataset_book = {"affectnet":AffectNetDataset, "raf-db":RAFDBDataset,
                    "affectnet7":AffectNetDataset7, "affectnet7balanced":AffectNetDataset7Balanced}
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
    elif setname == "affectnet7":
        trainset = AffectNetDataset7('train')
        testset = AffectNetDataset7('val')
        trainloader = datautils.DataLoader(trainset,**kwargs) 
        testloader = datautils.DataLoader(testset, **kwargs)
        return trainloader, testloader
    elif setname == "affectnet5balanced":
        trainset = AffectNetDataset5Balanced('train')
        testset = AffectNetDataset5Balanced('val')
        trainloader = datautils.DataLoader(trainset,**kwargs) 
        testloader = datautils.DataLoader(testset, **kwargs)
        return trainloader, testloader
    elif setname == "affectnet7balanced":
        trainset = AffectNetDataset7Balanced('train')
        testset = AffectNetDataset7Balanced('val')
        trainloader = datautils.DataLoader(trainset,**kwargs) 
        testloader = datautils.DataLoader(testset, **kwargs)
        return trainloader, testloader
    elif setname == "affectnet7_rafdb_balanced":
        trainset = AffectNetDataset7RAFDBDataset7Balanced('train')
        testset = AffectNetDataset7RAFDBDataset7Balanced('val')
        trainloader = datautils.DataLoader(trainset,**kwargs) 
        testloader = datautils.DataLoader(testset, **kwargs)
        return trainloader, testloader


if __name__=="__main__":
    trainset = AffectNetDataset7RAFDBDataset7Balanced('train')
    #trainset = AffectNetDataset7Balanced("train")