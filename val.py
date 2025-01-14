import torch
from torch.utils.data import DataLoader
from train_new import CNN, LicensePlateDataset
from torchvision import transforms


def validate_model(val_txt_path, dataset_path, model_path):
    """
    验证模型在验证集上的准确率。

    :param val_txt_path: 验证集的描述文件路径 (val.txt)
    :param dataset_path: 数据集根目录路径
    :param model_path: 训练好的模型权重路径
    :return: None
    """
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载验证数据
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    val_dataset = LicensePlateDataset(dataset_path, val_txt_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 初始化计数器
    total_samples = 0
    correct_plate_type = 0
    correct_characters = 0
    total_characters = 0

    char_map = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # 验证过程
    with torch.no_grad():
        for images, labels, type_labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            type_labels = type_labels.to(device)

            # 前向传播
            plate_type_output, outputs_7, outputs_8 = model(images)

            # 类型分类准确率
            predicted_types = plate_type_output.argmax(dim=1)
            correct_plate_type += (predicted_types == type_labels).sum().item()

            # 字符分类准确率
            for i, predicted_type in enumerate(predicted_types):
                if predicted_type == 0:  # 普通蓝牌
                    outputs = outputs_7
                    char_count = 7
                else:  # 新能源小型车
                    outputs = outputs_8
                    char_count = 8

                predicted_chars = [output[i].argmax(dim=0).item() for output in outputs[:char_count]]
                true_chars = labels[i, :char_count].tolist()

                for pred_char, true_char in zip(predicted_chars, true_chars):
                    if true_char != -1:  # 跳过填充值
                        total_characters += 1
                        if pred_char == true_char:
                            correct_characters += 1

            total_samples += images.size(0)

    # 计算准确率
    plate_type_accuracy = correct_plate_type / total_samples * 100
    character_accuracy = correct_characters / total_characters * 100

    print(f"验证集车牌类型分类准确率: {plate_type_accuracy:.2f}%")
    print(f"验证集字符分类准确率: {character_accuracy:.2f}%")

if __name__ == "__main__":
    val_txt_path = "./DataSet/filtered_val.txt"  # 替换为验证集描述文件路径
    dataset_path = "./DataSet"  # 数据集根目录路径
    model_path = "./train_model/cnn_epoch_35.pth"  # 替换为模型权重路径

    validate_model(val_txt_path, dataset_path, model_path)
