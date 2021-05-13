import csv
import numpy as np
import cv2
import tqdm
import json
import pathlib
train_file = open("training.csv", "r")
val_file = open("validation.csv", "r")
label_file = open("labels.json", "w")
origin_dir = pathlib.Path("Manually_Annotated_Images")
output_dir = pathlib.Path("output/data")
labels = {}
if output_dir.exists():
    for f in output_dir.glob("./*"):
        f.unlink()
    output_dir.rmdir()
output_dir.mkdir(parents=True)
#subDirectory_filePath,face_x,face_y,face_width,face_height,facial_landmarks,expression,valence,arousal
class CV2FaceDetector:
    def __init__(self, model_para_dir='ckpt/haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(model_para_dir)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, minSize=(100, 100))
        return faces
face_detector = CV2FaceDetector("/data/xzy/face-expression-recognization/ckpt/haarcascade_frontalface_default.xml")
def solve(file, split):
    reader = csv.reader(file)
    reader_iter = reader.__iter__()
    reader_iter.__next__()
    train_data = []
    cnt = np.zeros((15))
    pbar = tqdm.tqdm(reader_iter)
    for i,row in enumerate(pbar):
        pbar.set_description("%d/%d"%(i, 414800))
        
        
        if "NULL" in set(row):
            continue
        path = pathlib.Path(row[0])
        x, y, w, h, c = int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[6])
        if c>=7:
            continue
        new_name = "_".join([split, path.name]).split(".")
        new_name = new_name[0] + ".jpg"
        img = cv2.imread((origin_dir / path).absolute().__str__())
        faces = face_detector.detect(img)
        if len(faces)>0:
            x,y,w,h = faces[0]
            img = img[y:y+h, x:x+w]
            cv2.imwrite("output/data/"+new_name, img)
            labels[new_name] = c
            cnt[int(row[6])] += 1
        #except:
        #    print(path.absolute().__str__())
    print(cnt)

solve(train_file, "train")
solve(val_file, "val")
json.dump(labels, label_file)
    