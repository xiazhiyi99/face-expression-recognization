import torch
import cv2
import pathlib
import numpy as np
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.loss import MultiFocalLoss
import datasets
import tqdm

MAX_EPOCH = 45
LR = 1e-3
DATASET = "affectnet7"
BATCH_SIZE = 32
DECAY_STEP = [10,20,30,40]
FOCAL_LOSS = True
# load model
ckpt = torch.load("ckpt/mbv3_small.pth.tar", map_location='cpu')
if FOCAL_LOSS:
    alpha = np.zeros((1000))
    alpha[0:7] = np.array([.25, .25, .75, .75, .75, .75, .75])
    alpha[0:7] = np.array([.5, .3, 1.3, 3, 6, 10, 1.3])
    floss = MultiFocalLoss(1000, alpha)
    model = MobileNetV3_Small(loss=floss)
else:
    model = MobileNetV3_Small()
model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['state_dict'].items()})
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, DECAY_STEP, 0.3)
trainloader, valloader = datasets.get_loader(setname=DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)


def train_one_epoch(model, optimizer, loader):
    pbar = tqdm.tqdm(loader)
    loss_tot = 0.
    model.train()
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        loss = model.get_loss(data, label)
        loss.backward()
        optimizer.step()
        loss_tot += loss.item()
        pbar.set_description("idx:%d/%d loss:%f"%(i, len(pbar), loss.item()))
    return loss_tot / len(loader)

def eval_one_epoch(model, loader):
    pbar = tqdm.tqdm(loader)
    Ttot = 0
    Ftot = 0
    model.eval()
    pbar.set_description("evaluating...")
    for i, (data, label) in enumerate(pbar):
        data, label = data.to(device), label.to(device)
        pred = model(data)
        _, pred = torch.max(pred, 1)
        T = (pred == label).sum()
        F = (pred != label).sum()
        Ttot += T.item()
        Ftot += F.item()
    avg_acc = Ttot / (Ttot + Ftot)
    print("Accuracy:", avg_acc)
    return avg_acc

# -------------- train ------------------
best_acc = 0
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

print("Complete. Best acc:", best_acc)


    