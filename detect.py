import torch
import cv2
import numpy as np
from torchvision import transforms
from train_new import CNN  # 假定你的模型代码保存为 cnn_pytorch.py

def detect_license_plate(image_path, model_path):
    """
    检测图像中的车牌，并返回车牌类型和识别出的字符。

    :param image_path: 图像文件路径
    :param model_path: 训练好的模型权重路径
    :return: 车牌类型和识别出的字符
    """
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 预处理图像
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img_resized = cv2.resize(img, (128, 48))  # 确保图像大小为 128x48
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img_resized).unsqueeze(0).to(device)  # 添加 batch 维度

    # 前向传播
    with torch.no_grad():
        plate_type_output, outputs_7, outputs_8 = model(img_tensor)

    # 车牌类型
    plate_type = "普通蓝牌" if plate_type_output.argmax(dim=1).item() == 0 else "新能源小型车"

    # 字符预测
    if plate_type == "普通蓝牌":
        outputs = outputs_7
        char_count = 7
    else:
        outputs = outputs_8
        char_count = 8

    char_indices = [output.argmax(dim=1).item() for output in outputs[:char_count]]

    # 映射字符索引到实际字符
    char_map = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    recognized_chars = ''.join([char_map[idx] for idx in char_indices])

    return plate_type, recognized_chars

if __name__ == "__main__":
    # 示例使用
    image_path = "./test_images/000000004.jpg"
    # image_path = "./DataSet/CBLPRD-330k/000345606.jpg"  # 替换为测试图像路径
    model_path = "./train_model/cnn_epoch_35.pth"  # 替换为模型权重路径

    plate_type, recognized_chars = detect_license_plate(image_path, model_path)
    print(f"车牌类型: {plate_type}")
    print(f"识别出的字符: {recognized_chars}")
