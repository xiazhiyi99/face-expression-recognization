# face-expression-recognization

基于mobilenetv3的人脸表情检测器

## 要求
pytorch1.0+  
torchvision  
opencv-python  
pillow  
tqdm  

数据集：affectnet、RAF-DB或其它

## Demo

```shell
cd face-expression-recognization
python ./camera_test.py
```

## 训练模块

1. 准备数据集并修改 datasets.py

2. 从https://github.com/xiaolai-sqlai/mobilenetv3 下载预训练模型mbn_small.pth.tar至[root/ckpt]

3. 修改train.py内的超参数 

4. 执行

```shell
python train.py
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

## 可视化模块

1. 相机检测器 solver.CameraSolver  
```python
cam_solver = CameraSolver(detector, classifier)
# 打开相机，0代表本机相机，其他数字为外接相机
cam_solver.start(0)
for i in range(100):
    results, frame = cam_solver.get_solved_frame()
    plt.pause(0.01)
cam_solver.close()
```

2. 视频检测器 solver.VideoSolver  
```python
vid_solver = VideoSolver(detector, classifier)
vid_solver.start("视频路径")
for i in range(100):
    results, frame = vid_solver.get_solved_frame()
vid_solver.close()
```

3. 可视化类 solver.Visualizer
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