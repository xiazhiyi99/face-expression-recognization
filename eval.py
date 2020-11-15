import torch
import cv2
import pathlib
import numpy as np
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
import datasets
import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def eval(model, loader, label):
    TP = np.zeros((len(label)))
    FP = np.zeros((len(label)))
    FN = np.zeros((len(label)))
    TN = np.zeros((len(label)))
    CNT = np.zeros((len(label)))
    model.to(device).eval()
    pbar = tqdm.tqdm(loader)
    pbar.set_description("evaluating...")
    for _, (data, gt) in enumerate(pbar):
        data, gt = data.to(device), gt.to(device)
        pred = model(data)
        _, pred = torch.max(pred, 1)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        for i,(k,v) in enumerate(label.items()):
            TP[i] += np.logical_and(pred == k, gt == k).sum()
            FP[i] += np.logical_and(pred == k, gt != k).sum()
            FN[i] += np.logical_and(pred != k, gt == k).sum()
            TN[i] += np.logical_and(pred != k, gt != k).sum()
            CNT[i] += (gt == k).sum()
    label = [v for k,v in label.items()]
    for i in range(len(label)):
        print("%9s: precision:%.2f recall:%.2f (%.1f%%, %d)"%(
            label[i], TP[i]/(TP[i]+FP[i]), TP[i]/(TP[i]+FN[i]), 
            CNT[i]/CNT.sum()*100, CNT[i],
        ))
    print("%9s: precision:%.2f recall:%.2f (%.1f%%, %d)"%(
            "Average", (TP/(TP+FP)).mean(), (TP/(TP+FN)).mean(), 
            100, CNT.sum(),
        ))
    return

valLoader,_ = datasets.get_loader(setname="raf-db", batch_size=16,use_sampler=False, num_workers=4)
ckpt = torch.load("ckpt/raf-db_mobilenet_small_acc62.pth.tar", map_location='cpu')
model = MobileNetV3_Small()
model.load_state_dict(ckpt["state_dict"])
rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
eval(model, valLoader, rafdb_table)