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

BASE_DIR = ROOT_DIR = pathlib.Path(__file__).absolute()
import sys
sys.path.append(ROOT_DIR.__str__())

import argparse
import yaml
import time

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")
parser.add_argument("--e", action="store_true", default=False)
FLAGS = parser.parse_args()
f = open(FLAGS.config)
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()

def main():
    # MAX_EPOCH = 45
    # LR = 1e-3
    # DATASET = "affectnet7"
    # BATCH_SIZE = 64
    # DECAY_STEP = [10,20,30,40]
    # FOCAL_LOSS = True
    # load model
    model = get_pretrained_model(config)
    model = get_pretrained_model(config)

    # optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["lr"])
    if config["optimizer"]["weight_decay"]:
        p_decay, p_nodecay = split_parameters(model)
        optimizer = torch.optim.Adam([
            {"params": p_decay, "weight_decay":config["optimizer"]["weight_decay"]},
            {"params": p_nodecay}],
            lr=config["optimizer"]["lr"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), config["optimizer"]["lr"])
        
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                    config["scheduler"]["lr_scheduler"]["decay_list"],
                    config["scheduler"]["lr_scheduler"]["decay_rate"])
    trainloader, valloader = get_loader(config)
    trainer = Trainer(config, 
                      model,
                      optimizer,
                      trainloader,
                      valloader,
                      scheduler)
    trainer.train()

def get_pretrained_model(config):
    config = config["model"]
    #backbone = MobileNetV3_Small if config["name"] == "mobile_net_v3_small" else MobileNetV3_Large
    model_dict = {"mobile_net_v3_small":MobileNetV3_Small,
                  "mobile_net_v3_large":MobileNetV3_Large,
                  "xception":Xception,
                  "resnet50":ResNet50,
                  "resnet101":ResNet101,
                  "resnet152":ResNet152,
                  "ghostnet":GhostNet}
    backbone = model_dict[config["name"]]
    if config["use_focal_loss"]:
        alpha = np.zeros((1000))
        alpha[0:7] = np.array(config["focal_loss"]["alpha"])
        floss = MultiFocalLoss(1000, alpha, config["focal_loss"]["gamma"])
        model = backbone(loss=floss)
    else:
        in_channel = 3 if "in_channel" not in config else config["in_channel"]
        model = backbone() if in_channel==3 else backbone(in_channel=in_channel)
    if config["init"]:
        ckpt = torch.load(config["init"], map_location='cpu')
        model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
    else:
        model.init_params()
    return model

def get_loader(config):
    # todo
    config = config["dataset"]
    DATASET = config["name"]
    BATCH_SIZE = config["batch_size"]
    workers = config["workers"]
    trainloader, valloader = datasets.get_loader(setname=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    return trainloader, valloader


class Timer:
    def __init__(self):
        self.last = time.time() * 1000
    
    def get(self):
        now = time.time() * 1000
        res = now - self.last
        self.last = now
        return res


class Trainer:
    def __init__(self, config, model, optimizer, trainloader, valloader, lr_scheduler):
        config = config["trainer"]
        self.MAX_EPOCH = config["max_epoch"]
        self.EVAL_FREQUENCY = config["eval_frequency"]
        self.ckpt_dir = pathlib.Path(config["ckpt_dir"])
        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir()
        self.model = model
        self.optimizer = optimizer 
        self.trainloader = trainloader
        self.valloader = valloader 
        self.lr_scheduler = lr_scheduler

    def train(self):
        best_acc = -1
        for epoch in range(self.MAX_EPOCH):
            print("----------- Epoch %d lr %.8f -----------"%(epoch, self.optimizer.state_dict()['param_groups'][0]['lr']))
            self.train_one_epoch(self.model, self.optimizer, self.trainloader)
            self.lr_scheduler.step()
            if epoch % self.EVAL_FREQUENCY == 0:
                acc = self.eval_one_epoch(self.model, self.valloader)
                if acc>best_acc:
                    best_acc = acc
                    self.save_model(epoch, "checkpoints/checkpoint_best.pth.tar")
                    #state = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
                    #model_path = 'ckpt/%s_mobilenet_small%s.pth.tar'%(DATASET, "_floss_alpha2" if FOCAL_LOSS else "")
                    #print("# Best acc:%f, saving model %s"%(acc, model_path))
                    #torch.save(state, model_path)
        print("Complete. Best acc:", best_acc)
    
    def train_one_epoch(self, model, optimizer, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        pbar = tqdm.tqdm(loader)
        loss_tot = 0.
        Ttot = 0
        Ftot = 0
        model.train()
        model.to(device)
        timer = Timer()
        for i, (data, label) in enumerate(pbar):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            loss, out = model.get_loss(data, label)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(out, 1)
            T = (pred == label).sum()
            F = (pred != label).sum()
            Ttot += T.item()
            Ftot += F.item()
            _, pred = torch.max(out, 1)
            loss_tot += loss.item()
            pbar.set_description("idx:%d/%d loss:%f time:%d"%(i, len(pbar), loss.item(), timer.get()))
        print("Average Loss:%.3f Average Acc:%.3f"%(loss_tot / len(loader), Ttot / (Ttot + Ftot)))
        return 

    @torch.no_grad()
    def eval_one_epoch(self, model, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        pbar = tqdm.tqdm(loader)
        Ttot = 0
        Ftot = 0
        totloss = 0.
        model.eval()
        model.to(device)
        pbar.set_description("Evaluating...")
        for i, (data, label) in enumerate(pbar):
            data, label = data.to(device), label.to(device)
            loss, pred = model.get_loss(data, label)
            _, pred = torch.max(pred, 1)
            T = (pred == label).sum()
            F = (pred != label).sum()
            Ttot += T.item()
            Ftot += F.item()
            totloss += loss.item()
        avg_acc = Ttot / (Ttot + Ftot)
        avg_loss = totloss / totloss
        print("Eval Loss:%.3f Accuracy:%.3f"%(avg_loss, avg_acc))
        return avg_acc
    
    def save_model(self, epoch, path):
        state = {'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
        model_path = path #'ckpt/%s_mobilenet_small%s.pth.tar'%(DATASET, "_floss_alpha2" if FOCAL_LOSS else "")
        print("# Saving model %s"%model_path)
        torch.save(state, model_path)
# -------------- train ------------------
"""best_acc = 0
for epoch in range(MAX_EPOCH):
    print("----------- Epoch %d lr %.8f -----------"%(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    avg_loss = train_one_epoch(model, optimizer, trainloader)
    scheduler.step()
    if epoch % 2 == 0:
        acc = eval_one_epoch(model, valloader)
        if acc>best_acc:
            best_acc = acc
            state = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            model_path = 'ckpt/%s_mobilenet_small%s.pth.tar'%(DATASET, "_floss_alpha2" if FOCAL_LOSS else "")
            print("# Best acc:%f, saving model %s"%(acc, model_path))
            torch.save(state, model_path)

print("Complete. Best acc:", best_acc)"""

if __name__ == "__main__":
    main()

    
