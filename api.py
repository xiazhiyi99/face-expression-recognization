import torch
import numpy as np
import cv2
import torchvision.transforms as trans
import PIL.Image as Image
import json

rafdb_table = {1:"Surprise", 2:"Fear", 3:"Disgust", 4:"Happiness", 5:"Sadness", 6:"Anger", 7:"Neutral"}
affectnet_table = {0:"Happy", 1:"Sad", 2:"Surprise", 3:"Fear", 4:"Disgust", 5:"Anger"}

class CV2FaceDetector:
    def __init__(self, model_para_dir='ckpt/haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(model_para_dir)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, minSize=(100, 100))
        return faces

def get_trained_model(model, param_path, map_location="cpu"):
    ckpt = torch.load(param_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

class BatchSolver:
    def __init__(self, detector, classifier, device, label, batch_size=128):
        self.detector = detector
        self.classifier = classifier
        self.device = device
        #self.solver = solver
        self.batch_size = batch_size
        self.resizer = trans.Compose([
            trans.Resize((224)),
            trans.ToTensor(),
            trans.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        self.label_table = [v for k,v in label.items()]
        self.label_idx = [k for k,v in label.items()]

        self.classifier.to(device)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def solve(self, imgs):
        if len(imgs) != self.batch_size:
            raise Exception("batch size not match.")
        input = torch.zeros((self.batch_size, 3, 224, 224))
        boxes_vector = np.zeros((self.batch_size, 4))
        exist_vector = np.zeros((self.batch_size), dtype=np.bool)
        index_vector = np.zeros((self.batch_size))
        for i,(idx,img) in enumerate(imgs):
            faces = self.detector.detect(img)
            index_vector[i] = idx
            if len(faces) != 0:
                x,y,w,h = faces[0]
                crop = img[y:y+h, x:x+w, :]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = self.resizer(Image.fromarray(crop))
                input[i, ...] = crop
                boxes_vector[i, :] = np.array([x,y,w,h])
                exist_vector[i] = True
            else:
                exist_vector[i] = False
        #input = torch.Tensor(input)
        input.to(self.device)
        output = self.classifier(input).relu()[:, self.label_idx, ...]
        output = output.softmax(1)
        _, cls = torch.max(output, 1)
        cls = [self.label_table[x] for x in cls.detach().numpy().tolist()]
        prob_vector = output.detach().numpy().tolist()
        prob_vector = np.round(prob_vector, 3)
        result = {
            "Vector":prob_vector.tolist(),
            "Class":cls,
            "Boxes":boxes_vector.astype(int).tolist(),
            "Exist":exist_vector.tolist(),
            "Index":index_vector.astype(int).tolist()
        }
        return result

    def solve_in_json(self, imgs):
        return json.dumps(self.solve(imgs))

if __name__=="__main__":
    from solver import *

    detector = CV2FaceDetector()
    mbn = get_trained_model(mbnet.MobileNetV3_Small(), "ckpt/affectnet_mobilenetv3_small_acc83.pth.tar")
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    bsolver = BatchSolver(detector, mbn, device, affectnet_table, batch_size=32)
    def img_catcher(path, num):
        imgs = []
        vid = cv2.VideoCapture(path)
        for i in range(num):
            _, img = vid.read()
            imgs.append((i, img))
        return imgs
    string = bsolver.solve_in_json(img_catcher("./test/WeChat_20201120140410.mp4", 32))
    print(len(string))
    f = open("test/outputtest.json", "w")
    f.write(string)
    f.close()
