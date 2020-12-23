# face-expression-recognization

基于mobilenetv3的人脸表情检测器

## 要求
pytorch1.0+  
torchvision  
opencv-python  
pillow  
tqdm  

数据集：affectnet、RAF-DB或其它

## 本地摄像头测试

```shell
cd root/experiment/任意训练文件夹
python ../../camera_testV2.py --config config.yaml
```
为了方便测试，camera_testV2读入不同的config文件可进行不同模型的测试。需要确保config.yaml中“model.name”、“tester.resume_model”、“tester.label”的正确性。具体样例可以看已有的config文件。

## API

api.BatchSolver  
输出格式 json  

| 标签      | 内容      |  
|----------|----------|
|Vector    |浮点数list，7个表情类对映概率，大小为[batch_size, 7]|
|Class     |字符串list，预测的标签，大小为[batch_size]|
|Boxes     |整数list，按照xywh存储，若没有目标存储为[0,0,0,0]，大小为[batch_size, 4]|
|Exist     |布尔型list，图片中是否检测出人脸，大小为[batch_size]|
|Index     |整数list，对映图片帧序号cur_frame，大小为[batch_size]|

类别编号：  
rafdb_table = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]  
affectnet_table = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]
       
## 训练模块

首先需要准备数据集。建议自备数据集并自己写一个torch.data.Dataset类。也可以参考我们的数据集格式。如有需要下载请联系我们。

我们的使用了预处理过的affectnet和rafdb。路径如下：
```
root/data
├── AffectNet7
└── RAF-DB
```

RAF-DB数据集与官网直接下载的文件路径一致。  
AffectNet数据集经过人脸识别器剪裁并重命名。名命格式为"train_xxxxx.jpg""val_xxxxx.jpg"。label存在labels.json，格式为"文件名:label_id"
```
AffectNet7
├── labels.json
└── output
```

开始训练，以下以建立一个mobilenet训练为例。

```shell
cd root/experiment
mkdir mobilenet_train
cd mobilenet_train
cp ../其它训练文件夹/config.yaml ./config.yaml
vim config.yaml
```
之后修改config.yaml中的参数。开始训练。
```shell 
python ../../train.py --config ./config.yaml
``` 
测试每个类别对应准确率和召回率，并生成混淆矩阵等。
```shell
python ../../eval.py --config config.yaml
```


## 预测模块

1. 加载模型  
```python
import model.mobilenetv3 as mbnet
model = get_trained_model(mbnet.MobileNetV3_Small(), "ckpt/affectnet_mobilenetv3_small_acc83.pth.tar")
```

2. 人脸检测器 solver.CV2FaceDetector  
cv2自带cascade检测器，需下载参数至[root/ckpt]。  
```python
from solver import *
detector = CV2FaceDetector('ckpt/haarcascade_frontalface_default.xml')
results = detector.detect(frame)
```

3. 表情分类器 solver.ExpressionClassifier
表情分类器的封装，实例化时需要加载好模型，label表，平滑器（默认None）。
```python
from solver import *
smoother = LinearExponentialSmoothing(0.3)
affectnet_table = {0:"Happy", 1:"Sad", 2:"Surprise", 3:"Fear", 4:"Disgust", 5:"Anger"}
classifier = ExpressionClassifier(model, affectnet_table, smoother)
```

4. 相机检测器 solver.CameraSolver  
```python
cam_solver = CameraSolver(detector, classifier)
# 打开相机，0代表本机相机，其他数字为外接相机
cam_solver.start(0)
for i in range(100):
    results, frame = cam_solver.get_solved_frame()
    plt.pause(0.01)
cam_solver.close()
```

5. 视频检测器 solver.VideoSolver  
```python
vid_solver = VideoSolver(detector, classifier)
vid_solver.start("视频路径")
for i in range(100):
    results, frame = vid_solver.get_solved_frame()
vid_solver.close()
```

## 可视化模块

1. 可视化类 solver.Visualizer
```python
# 实例化时需要传入label列表
vizor = Visualizer(label_table=classifier.express_table)

cam_solver.start(0)
while True:
    # 绘制前需要更新绘制器内信息
    vizor.update(*cam_solver.get_solved_frame())
    vizor.show()
    plt.pause(0.01)
cam_solver.close()
```

## 参考

 MobileNetV3的PyTorch实现：https://github.com/xiaolai-sqlai/mobilenetv3