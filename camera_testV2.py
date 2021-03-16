import model.mobilenetv3 as mbnet
import cv2
from solver import *
import torch 
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.resnet import ResNet50, ResNet101, ResNet152
from model.xception import Xception
from model.ghostnet import GhostNet
import argparse
import yaml
import os

ROOT_DIR = os.path.realpath(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./config.yaml")
FLAGS = parser.parse_args()
f = open(FLAGS.config, encoding='utf-8')
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()

def get_trained_model(config, map_location="cpu"):
    config = config
    #backbone = MobileNetV3_Small if config["name"] == "mobile_net_v3_small" else MobileNetV3_Large
    model_dict = {"mobile_net_v3_small":MobileNetV3_Small,
                  "mobile_net_v3_large":MobileNetV3_Large,
                  "xception":Xception,
                  "resnet50":ResNet50,
                  "resnet101":ResNet101,
                  "resnet152":ResNet152,
                  "ghostnet":GhostNet}
    model = model_dict[config["model"]["name"]]()
    ckpt = torch.load(config["tester"]["resume_model"], map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

label_table = affectnet7_cn_table # 如果读取yaml里的中文label，需要改动其他py文件比较麻烦。所以这里直接读取固定的label，其它以后再改。 eval(config["tester"]["label"])
if config["tester"].get("color"):
    color_table = config["tester"]["color"]
else:
    color_table = [(0,0,100)]*len(label_table)

detector = CV2FaceDetector('../../ckpt/haarcascade_frontalface_default.xml')
model = get_trained_model(config, "cpu")
smoother = LinearExponentialSmoothing(1)
classifier = ExpressionClassifier(model, label_table, smoother)
cam_solver = CameraSolver(detector, classifier)
vizor = CV2Visualizer(label_table=classifier.express_table, color_table=color_table)

cam_id = 0 # 本机摄像头
# cam_id = 'rtsp://admin:bwton123@192.168.24.64:554/Streaming/Channels/101' # RTSP视频流

cam_solver.start(cam_id)
while True:
    rt = cam_solver.get_solved_frame()
    if rt:
        vizor.update(*rt)
        vizor.show()
    vizor.pause(10)
cam_solver.close()












''' old
import cv2
import mobilenetv3
import torch
import torchvision.transforms as trans
import PIL.Image as Image
import matplotlib.pyplot as plt
import time
import numpy as np

class FaceDetector:
    def __init__(self, model_para_dir='haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(model_para_dir)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, minSize=(100, 100))
        return faces

    def detect_and_draw(self, frame):
        faces = self.detect(frame)
        for box in faces:
            cv2.rectangle(frame, box, (0,0,255), 2)
        return frame

class ExpressionDetector:
    def __init__(self, face_detector, classifier):
        self.face_detector = face_detector
        self.classifier = classifier
        self.resizer = trans.Compose([
            trans.Resize((224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.express_table = ["Happy", "Sad", "Surprise",
                              "Fear", "Disgust", "Anger"]

    def detect(self, frame):
        boxes = self.face_detector.detect(frame)
        exp = []
        for box in boxes:
            x,y,w,h = box
            crop = frame[y:y+h, x:x+w, :]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = self.resizer(Image.fromarray(crop))
            crop = torch.unsqueeze(crop, 0)
            pred = self.classifier(crop).detach()[:, 0:6].relu()
            conf, res = torch.max(pred, 1)
            #print(conf, pred, pred.sum())
            conf = conf / pred.sum()
            #print(res,conf)
            exp.append({"result":res, "confidence":conf, "vector":pred, "box":box})
        return exp

    def draw(self, frame, data):
        for d in data:
            x,y,w,h = d["box"]
            res = d["result"]
            conf = d["confidence"]
            #print(res)
            cv2.rectangle(frame, d["box"], (0,0,255), 2)
            cv2.putText(frame, "%s: %f"%(self.express_table[res], conf), (x,y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,155), 1)

class Visualizer:
    def __init__(self, label_table, window_size=20):
        self.fig= plt.figure(figsize=(1600,1200))
        self.frame = None
        self.vector = None
        self.line = []
        self.window_size = window_size
        self.label_table = label_table
    def update(self, frame, vector):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if vector:
            self.vector = vector[0]["vector"].squeeze().numpy()
            self.line.append(self.vector/self.vector.sum())
        if len(self.line)>self.window_size:
            self.line = self.line[1:]
    def show(self):
        self.fig.clear()
        ax1 = self.fig.add_subplot(2, 2, 1)
        ax2 = self.fig.add_subplot(2, 2, 2)
        ax3 = self.fig.add_subplot(2, 1, 2)
        if self.frame is not None:
            ax1.imshow(self.frame)
        if self.vector is not None:
            ax2.bar(np.arange(0,len(self.vector)), self.vector/self.vector.sum())
            ax2.set_xticks(np.arange(0,len(self.vector)))
            ax2.set_xticklabels(["Happy", "Sad", "Surprise","Fear", "Disgust", "Anger"])
            xs = [np.arange(0,len(self.vector))]
            ax3.plot(self.line,LineWidth=2)
        plt.pause(0.01)


def iou(now, prev):
    x1,y1,w1,h1 = now["box"]
    x2,y2,w2,h2 = prev["box"]
    x,y = max(x1,x2), max(y1, y2)
    u,v = min(x1+w1,x2+w2), min(y1+h1, y2+h2)
    union = w1*h1 + w2*h2 - (u-x)*(v-y)
    cross = (u-x)*(v-y)
    return cross / union

def smooth(res, prev_res, alpha=0.3):
    for now in res:
        for prev in prev_res:
            if iou(now, prev)>0.5:
                now["vector"] = alpha * now["vector"] + (1 - alpha) * prev["vector"]
                now["confidence"], now["result"] = torch.max(now["vector"], 1)
                now["confidence"] = now["confidence"] / now["vector"].sum()
                break

vizor = Visualizer([])
plt.ion()
vizor.show()
detector = FaceDetector()
classifier = mobilenetv3.MobileNetV3_Small()
state = torch.load("./affectnet_mobilenet_small.pth.tar", map_location='cpu')
classifier.load_state_dict(state["state_dict"])
classifier.eval()
exp_detector = ExpressionDetector(detector, classifier)

cam = cv2.VideoCapture(0)
prev_res = []
prev_t = int(time.time()*1000)
while(cam.isOpened()):
    _, frame = cam.read()
    res = exp_detector.detect(frame)
    smooth(res, prev_res)
    prev_res = res
    exp_detector.draw(frame, res)

    now_t = int(time.time()*1000)
    fps = 1000 / (now_t - prev_t + 1)
    prev_t = now_t
    if fps<1000:
        cv2.putText(frame, "FPS:%.1f"%fps, (0,40),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,155), 1)
    vizor.update(frame, res)
    vizor.show()
'''