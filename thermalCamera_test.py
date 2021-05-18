import model.mobilenetv3 as mbnet
import cv2
from solver import *
import torch 
from model.mobilenetv3 import MobileNetV3_Small, MobileNetV3_Large
from model.resnet import ResNet50, ResNet101, ResNet152
from model.xception import Xception
from model.ghostnet import GhostNet
import torchvision.transforms as trans
import PIL.Image as Image
import argparse
import yaml
import os

ROOT_DIR = os.path.realpath(__file__)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="/data/xzy/face-expression-recognization/experiment/xception_oulu7/config.yaml")
FLAGS = parser.parse_args()
f = open(FLAGS.config, encoding='utf-8')
config = yaml.load(f, Loader=yaml.FullLoader)
f.close()


transform_oulu_test = trans.Compose([
    #trans.RandomHorizontalFlip(0.5),
    #trans.RandomResizedCrop((224,224),scale=(0.75,1.0),ratio=(0.75,1.33),interpolation=2),
    trans.Resize((224,224)),
    #trans.ColorJitter(brightness=0.3),
    #trans.ColorJitter(contrast=0.3),
    #trans.ColorJitter(saturation=0.1),
    #trans.RandomRotation(20, resample=False, expand=False, center=None),
    trans.Grayscale(),
    trans.ToTensor(),
    trans.Normalize(mean=[0.456],
                    std=[0.226]),
])

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
    model = model_dict[config["model"]["name"]](in_channel=1)
    #ckpt = torch.load("./experiment/xception_aff7rafdb_balanced_transform/" + config["tester"]["resume_model"], map_location=map_location)
    ckpt = torch.load("/data/xzy/face-expression-recognization/experiment/xception_oulu7/checkpoints/checkpoint_best.pth.tar")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

class CV2FaceDetector:
    def __init__(self, model_para_dir='./ckpt/haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(model_para_dir)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
        faces = self.detector.detectMultiScale(gray, minSize=(50, 50))
        return faces

class DlibFaceDetector:
    def __init__(self, model_para_dir='./ckpt/mmod_human_face_detector.dat', zoom_factor=1.4):
        import dlib
        self.detector = dlib.cnn_face_detection_model_v1(model_para_dir)
        self.zoom = zoom_factor
        print("Dlib Use CUDA:",dlib.DLIB_USE_CUDA)

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector(rgb, 0)
        ret = []
        for face in faces:
            face = face.rect
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            h = round((bottom - top) * self.zoom)
            w = round((right - left) * self.zoom)
            x = (bottom + top) // 2
            y = (right + left) // 2
            left = max(0, y - w // 2)
            top = max(0, x - h // 2)
            right = min(frame.shape[1], y + w // 2)
            bottom = min(frame.shape[0], x + h // 2)
            ret.append([left, top, w, h])
        return ret

    def draw_frame(self, data, frame, zoom_factor=1.4):
        # 将检测框画到实际画面里
        for box in data:
            x,y,w,h = box
            #print(res)
            color = (0,0,155)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            break
        return frame

label_table = affectnet7_table # 如果读取yaml里的中文label，需要改动其他py文件比较麻烦。所以这里直接读取固定的label，其它以后再改。 eval(config["tester"]["label"])
if config["tester"].get("color"):
    color_table = config["tester"]["color"]
else:
    color_table = [(0,0,100)]*len(label_table)
    
class CameraSolver:
    def __init__(self, face_detector, classifier, fps=1):
        self.fps = fps
        self.detector = face_detector
        self.classifier = classifier
        self.cam = None

    def start(self, cam_id=0):
        self.cam = cv2.VideoCapture(cam_id)
        # self.cam = RTSCapture.create(cam_id)
        #self.cam.start_read()
    
    def isOpened(self):
        return self.cam.isOpened()

    def close(self):
        if self.cam:
            self.cam.release()

    def get_frame(self):
        if self.cam:
            _, frame = self.cam.read()
            return frame
        else:
            raise Exception("Camera Error")

    def get_solved_frame(self):
        # 从相机得到一帧画面，并返回检测结果和画面
        frame = self.get_frame()
        if frame is None:
            return None
        faces = self.detector.detect(frame)
        #frame = Image.fromarray(frame)
        #frame = transform_oulu_test(frame)
        expressions = self.classifier.detect(frame, faces)
        #self.classifier.draw(frame, expressions)
        return expressions, frame

detector = CV2FaceDetector()
detector = DlibFaceDetector()
model = get_trained_model(config, "cuda:1")
smoother = LinearExponentialSmoothing(1)
classifier = ExpressionClassifier(model, label_table, smoother, transform_oulu_test, torch.device("cuda:1"))
cam_solver = CameraSolver(detector, classifier)
vizor = CV2Visualizer(label_table=classifier.express_table, color_table=color_table)

cam_id = "/data/xzy/data/bus/2/R20210511-150927NMD00P0F.mp4"
# cam_id = 0 # 本机摄像头
# cam_id = 'rtsp://admin:bwton123@192.168.24.64:554/Streaming/Channels/101' # RTSP视频流

cam_solver.start(cam_id)
print(cam_solver.cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam_solver.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('testwrite2.mp4', fourcc, 20.0, (704,576),True)

cnt = 0
while cam_solver.isOpened():
    #rt = detector.read()
    rt = cam_solver.get_solved_frame()
    #print(rt[1])
    if rt:
        vizor.update(*rt)
        frame = vizor.draw()
    out.write(frame)
    cnt += 1
    if cnt % 24 == 0:
        print(cnt //24)
    if cnt//24==15:
        break

cam_solver.close()
out.release()