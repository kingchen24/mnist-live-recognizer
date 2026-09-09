import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout层
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 经过3次池化后尺寸为3x3
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # 展平
        x = x.view(-1, 128 * 3 * 3)
        
        # 全连接层
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

class ImprovedCNN(nn.Module):
    """改进的CNN模型（带残差连接）"""
    
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # 第一个卷积块
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 第二个卷积块
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 第三个卷积块
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.classifier(x)
        
        return x

def create_model(model_name='simple_cnn', num_classes=10, device='cpu'):
    """创建模型工厂函数"""
    model_dict = {
        'simple_cnn': SimpleCNN,
        'improved_cnn': ImprovedCNN
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not found. Available: {list(model_dict.keys())}")
    
    model = model_dict[model_name](num_classes=num_classes)
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"=== Model: {model_name} ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model