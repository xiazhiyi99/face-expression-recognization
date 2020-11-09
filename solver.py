import cv2
import model.mobilenetv3 as mbnet
import torch
import torchvision.transforms as trans
import PIL.Image as Image
import matplotlib.pyplot as plt
import time
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg")

rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
affectnet_table = {0:"Happy", 1:"Sad", 2:"Surprise", 3:"Fear", 4:"Disgust", 5:"Anger"}


def iou(now, prev):
    x1,y1,w1,h1 = now["box"]
    x2,y2,w2,h2 = prev["box"]
    x,y = max(x1,x2), max(y1, y2)
    u,v = min(x1+w1,x2+w2), min(y1+h1, y2+h2)
    union = w1*h1 + w2*h2 - (u-x)*(v-y)
    cross = (u-x)*(v-y)
    return cross / union


def get_trained_model(model, param_path, map_location="cpu"):
    ckpt = torch.load(param_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


class LinearExponentialSmoothing:
    def __init__(self, rate=0.1):
        self.rate = rate

    def smooth(self, res, prev_res):
        for now in res:
            for prev in prev_res:
                if iou(now, prev) > 0.5:
                    now["vector"] = alpha * now["vector"] + (1 - alpha) * prev["vector"]
                    now["probability"], now["result"] = torch.max(now["vector"], 0)
                    now["probability"] = now["probability"] / now["vector"].sum()
                    break


class CV2FaceDetector:
    def __init__(self, model_para_dir='ckpt/haarcascade_frontalface_default.xml'):
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

class ExpressionClassifier:
    def __init__(self, classifier, express_table, smoother=None):
        self.classifier = classifier
        self.resizer = trans.Compose([
            trans.Resize((224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.output_range = express_table.keys()
        self.express_table = [v for k,v in express_table.items()]
        self.prev_exp = []
        self.smoother = smoother

    def detect(self, frame, boxes):
        exp = []
        for box in boxes:
            x,y,w,h = box
            crop = frame[y:y+h, x:x+w, :]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = self.resizer(Image.fromarray(crop))
            crop = torch.unsqueeze(crop, 0)
            pred = self.classifier(crop).detach().relu()
            pred = torch.squeeze(pred, 0)
            pred = pred[list(self.output_range)]
            pred = pred / pred.sum()
            prob, res = torch.max(pred, 0)
            exp.append({"result":res, "probability":prob, "vector":pred, "box":box})

        if self.smoother:
            self.smoother.smooth(exp, self.prev_exp, self.smooth_rate)
        self.prev_exp = exp
        return exp

    def __draw(self, frame, data):
        # no more use, move to visualizer
        for d in data:
            x,y,w,h = d["box"]
            res = d["result"]
            prob = d["probability"]
            #print(res)
            cv2.rectangle(frame, d["box"], (0,0,255), 2)
            cv2.putText(frame, "%s: %f"%(self.express_table[res], prob), (x,y-10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,155), 1)


class CameraSolver:
    def __init__(self, face_detector, classifier, fps=1):
        self.fps = fps
        self.detector = face_detector
        self.classifier = classifier
        self.cam = None

    def start(self, cam_id=0):
        self.cam = cv2.cv2.VideoCapture(cam_id)

    def close(self):
        if self.cam:
            self.cam.release()

    def get_frame(self):
        if self.cam and self.cam.isOpened():
            _, frame = self.cam.read()
            return frame
        else:
            raise Exception("Camera Error")

    def get_solved_frame(self):
        frame = self.get_frame()
        faces = self.detector.detect(frame)
        expressions = self.classifier.detect(frame, faces)
        #self.classifier.draw(frame, expressions)
        return expressions, frame


class VideoSolver:
    def __init__(self, face_detector, classifier, fps=1):
        #self.fps = fps
        self.detector = face_detector
        self.classifier = classifier
        self.video = None
        self.video_fps = 0

    def start(self, path):
        self.video = cv2.VideoCapture(path)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)

    def close(self):
        if self.video:
            self.video.release()
            self.video_fps = 0

    def get_frame(self):
        if self.video and self.video.isOpened():
            _, frame = self.video.read()
            return frame
        else:
            raise Exception("Video Error")

    def get_solved_frame(self):
        frame = self.get_frame()
        faces = self.detector.detect(frame)
        expressions = self.classifier.detect(frame, faces)
        #self.classifier.draw(frame, expressions)
        return expressions, frame


class Visualizer:
    def __init__(self, label_table, window_size=20):
        self.fig= plt.figure(figsize=(80,60))
        self.frame = None
        self.vector = None
        self.line = []
        self.window_size = window_size
        self.label_table = label_table

        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 1, 2)

    def update(self, data, frame):
        self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if data:
            # find the biggest face
            area = 0
            max_d = None
            for d in data:
                x, y, w, h = d["box"]
                if w * h > area:
                    self.vector = d["vector"].squeeze().numpy()
                    area = w * h
                    max_d = d
            self.frame = self.draw_frame(max_d, self.frame)
            self.line.append(self.vector)
        if len(self.line)>self.window_size:
            self.line = self.line[1:]

    def draw_frame(self, data, frame):
        x,y,w,h = data["box"]
        res = data["result"]
        prob = data["probability"]
        #print(res)
        cv2.rectangle(frame, data["box"], (0,0,255), 2)
        cv2.putText(frame, "%s: %f"%(self.label_table[res], prob), (x,y-10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,155), 1)
        return frame

    def show(self):
        plt.ion()
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        if self.frame is not None:
            self.ax1.imshow(self.frame)
        if self.vector is not None:
            self.ax2.bar(np.arange(0,len(self.vector)), self.vector/self.vector.sum())
            self.ax2.set_xticks(np.arange(0,len(self.vector)))
            self.ax2.set_xticklabels(self.label_table)
            #xs = [np.arange(0,len(self.vector))]
            self.ax3.plot(self.line,LineWidth=2)
            self.ax3.legend(self.label_table, loc=4)

if __name__=="__main__":
    detector = CV2FaceDetector()
    mbn = get_trained_model(mbnet.MobileNetV3_Small(), "ckpt/affectnet_mobilenetv3_small_acc83.pth.tar")
    classifier = ExpressionClassifier(mbn, affectnet_table)

    cam_solver = CameraSolver(detector, classifier)
    cam_solver.start(0)
    _,frame = cam_solver.get_solved_frame()
    #cv2.imshow("", frame)
    #cv2.waitKey(0)
    #cam_solver.close()

    vid_solver = VideoSolver(detector, classifier)
    vid_solver.start("test/WIN_20200707_18_04_40_Pro.mp4")
    _,frame = vid_solver.get_solved_frame()
    #cv2.imshow("", frame)
    #cv2.waitKey(0)
    vizor = Visualizer(label_table=classifier.express_table)
    while True:
        vizor.update(*cam_solver.get_solved_frame())
        vizor.show()
        plt.pause(0.01)
