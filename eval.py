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
    model.eval()
    pbar = tqdm.tqdm(loader)
    pbar.set_description("evaluating...")
    for _, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, pred = torch.max(pred, 1)
        for i in range(len(label)):
            TP[i] += (pred == i and label == i).sum().item()
            FP[i] += (pred == i and label != i).sum().item()
            TN[i] += (pred != i and label != i).sum().item()
            FN[i] += (pred != i and label = i).sum().item()
    avg_acc = Ttot / (Ttot + Ftot)
    print("Accuracy:", avg_acc)
    for i in range(len(label)):
        TP[i] += (pred == i and label == i).sum().item()
        FP[i] += (pred == i and label != i).sum().item()
        TN[i] += (pred != i and label != i).sum().item()
        FN[i] += (pred != i and label = i).sum().item()
    return avg_acc

_, valLoader = datasets.get_loader(setname="raf-db", batch_size=16, shuffle=True, num_workers=4)
ckpt = torch.load("ckpt/raf-db_mobilenetv3_small_acc72.pth.tar", map_location='cpu')
model = MobileNetV3_Small()
model.load_state_dict(ckpt["state_dict"])
rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
eval(model, valLoader, rafdb_table)