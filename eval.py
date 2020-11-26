import torch
import cv2
import pathlib
import numpy as np
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
import datasets
import tqdm
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')



def eval(model, loader, label, confusion_matrix="./conf_mat.jpg"):
    TP = np.zeros((len(label)))
    FP = np.zeros((len(label)))
    FN = np.zeros((len(label)))
    TN = np.zeros((len(label)))
    CNT = np.zeros((len(label)))
    TOT = 0
    NUM = 0
    if confusion_matrix:
        conf_mat = np.zeros((len(label), len(label)), dtype=int)
    model.to(device).eval()
    pbar = tqdm.tqdm(loader)
    pbar.set_description("evaluating...")
    for _, (data, gt) in enumerate(pbar):
        data, gt = data.to(device), gt.to(device)
        pred = model(data)
        _, pred = torch.max(pred, 1)
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        NUM += pred.shape[0] 
        TOT += (pred == gt).sum()
        for i,(k,v) in enumerate(label.items()):
            TP[i] += np.logical_and(pred == k, gt == k).sum()
            FP[i] += np.logical_and(pred == k, gt != k).sum()
            FN[i] += np.logical_and(pred != k, gt == k).sum()
            TN[i] += np.logical_and(pred != k, gt != k).sum()
            CNT[i] += (gt == k).sum()
            if confusion_matrix:
                for j, (p,q) in enumerate(label.items()):
                    conf_mat[i, j] += np.logical_and(pred == p, gt == k).sum()

    label = [v for k,v in label.items()]
    for i in range(len(label)):
        print("%9s: Precision:%.2f Recall:%.2f (%.1f%%, %d)"%(
            label[i], TP[i]/(TP[i]+FP[i]), TP[i]/(TP[i]+FN[i]), 
            CNT[i]/CNT.sum()*100, CNT[i],
        ))
    print("%9s: Precision:%.2f Recall:%.2f (%.1f%%, %d)"%(
            "Average", (TP/(TP+FP)).mean(), (TP/(TP+FN)).mean(), 
            100, CNT.sum(),
        ))
    
    print("%9s Accuracy:%.2f"%(
            "Top1", TOT / NUM
        ))
    if confusion_matrix:
        df_cm = pd.DataFrame(conf_mat,
                     index = [i for i in label],
                     columns = [i for i in label])
        fig = plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap="BuPu")
        fig.save(confusion_matrix)
    return

ckpt_name = "affectnet7_mobilenet_small_floss_alpha2.pth.tar"
print("Evaluating",ckpt_name)
_, valLoader = datasets.get_loader(setname="affectnet7", batch_size=8,use_sampler=False, num_workers=4)
ckpt = torch.load("ckpt/"+ckpt_name, map_location='cpu')
model = MobileNetV3_Small()
model.load_state_dict(ckpt["state_dict"])
#rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
affectnet7_table = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger"}
eval(model, valLoader, affectnet7_table)