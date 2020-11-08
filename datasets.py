import torch
import torchvision.transforms as trans
import torch.utils.data as datautils
import json
import pathlib
import cv2
import PIL

transform = trans.Compose([
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
        f = open('./data/RAF-DB/EmoLabel/list_patition_label.txt')
        self.labels = [line.strip('\n').split(' ') for line in f.readlines()]
        self.labels = {k:int(v) for [k,v] in self.labels}
        f.close()

        self.datapath = pathlib.Path('/data/xzy/face-emotion/data/RAF-DB/Image/original')
        self.datalist = [x for x in self.datapath.glob('./%s*jpg'%split)]
    
    def __getitem__(self, idx):
        img = PIL.Image.open(str(self.datalist[idx]))
        img = transform(img)
        label = self.labels[self.datalist[idx].name]
        return img, label
    
    def __len__(self):
        return len(self.datalist)


def get_loader(setname="affectnet", traindata_ratio=0.5, **kwargs):
    dataset_book = {"affectnet":AffectNetDataset, "raf-db":RAFDBDataset}
    if setname == "raf-db":
        trainset = RAFDBDataset('train')
        testset = RAFDBDataset('test')
    elif setname == "affectnet":
        dataset = dataset_book[setname]()
        train_size = int(len(dataset) * traindata_ratio)
        val_size = len(dataset) -  train_size
        trainset, testset = datautils.random_split(dataset, [train_size, val_size])
    trainloader, testloader = datautils.DataLoader(trainset, **kwargs), datautils.DataLoader(testset, **kwargs)
    return trainloader, testloader

if __name__=="__main__":
    train_loader, test_loader = get_loader(setname='raf-db',batch_size=4, shuffle=True, num_workers=4)
    for data, label in train_loader:
        print(type(data),data.size())
        print(torch.nn.functional.one_hot(label, 1000).size())
        break