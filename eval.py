import torch
import cv2
import pathlib
import numpy as np
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.resnet import ResNet50, ResNet101, ResNet152
from model.xception import Xception
from model.ghostnet import GhostNet
from model.loss import MultiFocalLoss
import datasets
import tqdm
import torch
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json

BASE_DIR = ROOT_DIR = pathlib.Path(__file__).absolute()
import sys
sys.path.append(ROOT_DIR.__str__())

import argparse
import yaml
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")
FLAGS = parser.parse_args()
f = open(FLAGS.config)
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()

def main():
    model = get_model(config)
    _, valloader = get_loader(config)
    tester = Tester(config, model, valloader)
    tester.eval()


def get_model(config):
    model_name = config["model"]["name"]
    config = config["tester"]
    path = config["resume_model"]
    print("Evaluating", path)
    ckpt = torch.load(path, map_location='cpu')
    model_dict = {"mobile_net_v3_small":MobileNetV3_Small,
                "mobile_net_v3_large":MobileNetV3_Large,
                "xception":Xception,
                "resnet50":ResNet50,
                "resnet101":ResNet101,
                "resnet152":ResNet152,
                "ghostnet":GhostNet}
    model = model_dict[model_name]()
    model.load_state_dict(ckpt["state_dict"])
    return model

def get_loader(config):
    # todo
    config = config["dataset"]
    DATASET = config["name"]
    BATCH_SIZE = config["batch_size"]
    workers = config["workers"]
    trainloader, valloader = datasets.get_loader(setname=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    return trainloader, valloader

class Tester:
    def __init__(self, config, model, testloader):
        config = config["tester"]
        self.confusion_matrix = config["confusion_matrix_path"]
        self.label = eval(config["label"])
        self.loader = testloader
        self.model = model

    @torch.no_grad()
    def eval(self): 
        model = self.model
        loader = self.loader
        label = self.label
        confusion_matrix = self.confusion_matrix
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        TP = np.zeros((len(label)))
        FP = np.zeros((len(label)))
        FN = np.zeros((len(label)))
        TN = np.zeros((len(label)))
        CNT = np.zeros((len(label)))
        TOT = 0
        TOT2 = 0
        TOT3 = 0
        NUM = 0
        if confusion_matrix:
            conf_mat = np.zeros((len(label), len(label)), dtype=int)
        model.to(device).eval()
        pbar = tqdm.tqdm(loader)
        pbar.set_description("evaluating...")
        for _, (data, gt) in enumerate(pbar):
            data, gt = data.to(device), gt.to(device)
            pred_score = model(data)
            _, pred = torch.max(pred_score, 1)
            pred = pred.cpu().numpy()
            pred_score[np.arange(pred_score.shape[0]),pred] = -99
            _, pred2 = torch.max(pred_score, 1)
            pred2 = pred2.cpu().numpy()
            pred_score[np.arange(pred_score.shape[0]),pred2] = -99
            _, pred3 = torch.max(pred_score, 1)
            pred3 = pred3.cpu().numpy()
            #print((pred == pred2).sum())
            #print(pred.shape, pred2.shape)

            gt = gt.cpu().numpy()
            NUM += pred.shape[0] 
            TOT += (pred == gt).sum()
            TOT2 += np.logical_or(pred == gt, pred2 == gt).sum()
            TOT3 += np.logical_or(np.logical_or(pred == gt, pred2 == gt), pred3 == gt).sum()
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
            print("%9s: Precision:%.2f Recall:%.2f F1:%.2f (%.1f%%, %d)"%(
                label[i], TP[i]/(TP[i]+FP[i]), TP[i]/(TP[i]+FN[i]), 
                2 / (1/(TP[i]/(TP[i]+FP[i])) + 1/(TP[i]/(TP[i]+FN[i]))),
                CNT[i]/CNT.sum()*100, CNT[i],
            ))
        print("%9s: mPrecision:%.2f mRecall:%.2f macroF1:%.2f (%.1f%%, %d)"%(
                "Average", (TP/(TP+FP)).mean(), (TP/(TP+FN)).mean(), 
                2 / (1/(TP/(TP+FP)) + 1/(TP/(TP+FN))).mean(),
                100, CNT.sum(),
            ))

        print("%9s Accuracy:%.2f"%(
                "Top1", TOT / NUM
            ))
        print("%9s Accuracy:%.2f"%(
                "Top2", TOT2 / NUM
            ))
        print("%9s Accuracy:%.2f"%(
                "Top3", TOT3 / NUM
            ))
        
        with open("./result.txt","w") as f:
            for i in range(len(label)):
                f.write("%9s: Precision:%.2f Recall:%.2f (%.1f%%, %d)\n"%(
                    label[i], TP[i]/(TP[i]+FP[i]), TP[i]/(TP[i]+FN[i]), 
                    CNT[i]/CNT.sum()*100, CNT[i],
                ))
            f.write("%9s: Precision:%.2f Recall:%.2f (%.1f%%, %d)\n"%(
                "Average", (TP/(TP+FP)).mean(), (TP/(TP+FN)).mean(), 
                100, CNT.sum(),
            ))
            f.write("%9s Accuracy:%.2f\n"%(
                "Top1", TOT / NUM,
            ))
            f.write("%9s Accuracy:%.2f\n"%(
                "Top2", TOT2 / NUM,
            ))
            f.write("%9s Accuracy:%.2f\n"%(
                "Top3", TOT3 / NUM,
            ))
        if confusion_matrix:
            df_cm = pd.DataFrame(conf_mat,
                            index = [i for i in label],
                            columns = [i for i in label])
            #fig = plt.figure(figsize = (10,7))
            fig = sn.heatmap(df_cm, annot=True, cmap="BuPu", fmt='g')
            fig = fig.get_figure()
            fig.savefig(confusion_matrix, dpi=400)
        return

if __name__ == "__main__":
    main()

if __name__ == "__main__  not use":
    ckpt_name = "affectnet7_mobilenet_small_floss_alpha2.pth.tar"
    print("Evaluating",ckpt_name)
    _, valLoader = datasets.get_loader(setname="affectnet7", batch_size=8,use_sampler=False, num_workers=4)
    ckpt = torch.load("ckpt/"+ckpt_name, map_location='cpu')
    model = MobileNetV3_Small()
    model.load_state_dict(ckpt["state_dict"])
    #rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
    affectnet7_table = {0:"Neutral", 1:"Happy", 2:"Sad", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger"}
    eval(model, valLoader, affectnet7_table)