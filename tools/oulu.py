import pathlib
import cv2
import shutil
import random
from threading import Thread
import pickle

class CV2FaceDetector:
    def __init__(self, model_para_dir='/data/xzy/face-expression-recognization/ckpt/haarcascade_frontalface_default.xml'):
        self.detector = cv2.CascadeClassifier(model_para_dir)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, minSize=(50, 50))
        return faces

    def draw_frame(self, data, frame):
        # 将检测框画到实际画面里
        for box in data:
            x,y,w,h = box
            #print(res)
            color = (0,0,155)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            break
        return frame

class DlibFaceDetector:
    def __init__(self, model_para_dir='/data/xzy/face-expression-recognization/ckpt/mmod_human_face_detector.dat', zoom_factor=1.4):
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
        """
        for face in data:
            face = face.rect
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            h = bottom - top
            w = right - left
            x = (bottom + top) // 2
            y = (right + left) // 2
            left = max(0, y - round(w*zoom_factor) // 2)
            top = max(0, x - round(h*zoom_factor) // 2)
            right = min(frame.shape[1], y + round(w*zoom_factor) // 2)
            bottom = min(frame.shape[0], x + round(h*zoom_factor) // 2)
            #print(frame.shape, right)
            #print(res)
            color = (0,0,155)
            cv2.rectangle(frame,  (left, top), (right, bottom), color, 2)
            break
        """
        return frame

def async_cp(src, dst, detector=None):
    if detector==None:
        Thread(target=shutil.copy, args=[src, dst]).start()
        return None
    else:
        frame = cv2.imread(src)
        faces = detector.detect(frame)
        # frame = detector.draw_frame(faces, frame)
        if len(faces) != 1:
            print(src, len(faces))
        Thread(target=shutil.copy, args=[src, dst]).start()
        return faces

def generate_random_str(randomlength=16):
  """
  生成一个指定长度的随机字符串
  """
  random_str = ''
  base_str = 'ABCDEFGHIGKLMNOPQRSTUVWXYZabcdefghigklmnopqrstuvwxyz0123456789'
  length = len(base_str) - 1
  for i in range(randomlength):
    random_str += base_str[random.randint(0, length)]
  return random_str

def prepare_original():
    CLEAR_ALL = True
    original_img = pathlib.Path("./OriginalImg/NI")
    NI = original_img / "NI"
    output = pathlib.Path("./Oulu7")
    output.mkdir(exist_ok=True)
    if CLEAR_ALL:
        [x.unlink() for x in output.glob("./*")]

    affectnet7_table = {0:"Neutral", 1:"Happiness", 2:"Sadness", 3:"Surprise", 4:"Fear", 5:"Disgust", 6:"Anger"}
    affectnet7_table_re = {v:k for k,v in affectnet7_table.items()}

    #with open()
    detector = DlibFaceDetector()
    label = {"label":{}, "bbox":{}}
    for idx in range(1, 81):
        print(idx)
        for sub in ["Dark", "Strong", "Weak"]:
            for emotion in ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]:
                data_dir = original_img / sub / ("P%03d"%idx) / emotion
                img_list = [x for x in data_dir.glob("./*jpeg")]
                img_list.sort()
                
                # first one is neutral
                output_name = "P%03d_%s_%s_%s.jpeg"%(idx,sub,"Neutral", generate_random_str(8))
                label["label"][output_name] = 0
                label["bbox"][output_name] = async_cp(img_list[0].absolute().__str__(), output / output_name, detector=detector)

                # last seven is this emotion
                for i in range(1,8):
                    output_name = "P%03d_%s_%s_%s.jpeg"%(idx,sub,emotion, generate_random_str(8))
                    label["label"][output_name] = affectnet7_table_re[emotion]
                    label["bbox"][output_name] = async_cp(img_list[-i].absolute().__str__(), output / output_name, detector=detector)
    with open("./Oulu7label.pickle", "wb") as f:
        pickle.dump(label["label"], f)
        pickle.dump(label["bbox"], f)

if __name__=="__main__":
    prepare_original()
    #prepare_crop





            

        


