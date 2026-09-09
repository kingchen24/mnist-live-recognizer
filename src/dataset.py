import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class MNISTDataLoader:
    """MNIST数据加载器"""
    
    def __init__(self, batch_size=64, download=True):
        """
        初始化数据加载器
        
        Args:
            batch_size: 批次大小
            download: 是否下载数据集
        """
        self.batch_size = batch_size
        self.download = download
        
        # 数据增强和预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
        ])
        
        # 加载训练集和测试集
        self.train_dataset, self.test_dataset = self.load_datasets()
        
    def load_datasets(self):
        """加载MNIST数据集"""
        train_dataset = datasets.MNIST(
            root='./data',
            train=True,
            download=self.download,
            transform=self.transform
        )
        
        test_dataset = datasets.MNIST(
            root='./data',
            train=False,
            download=self.download,
            transform=self.transform
        )
        
        return train_dataset, test_dataset
    
    def get_dataloaders(self):
        """获取数据加载器"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, test_loader
    
    def visualize_samples(self, num_samples=9):
        """可视化样本"""
        images, labels = next(iter(DataLoader(
            self.train_dataset,
            batch_size=num_samples,
            shuffle=True
        )))
        
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
        axes = axes.ravel()
        
        for idx in range(num_samples):
            img = images[idx].squeeze().numpy()
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(f'Label: {labels[idx].item()}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_dataset_info(self):
        """获取数据集信息"""
        print("=== MNIST Dataset Info ===")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Image shape: {self.train_dataset[0][0].shape}")
        print(f"Number of classes: {10}")
        
        # 统计类别分布
        train_labels = [label for _, label in self.train_dataset]
        test_labels = [label for _, label in self.test_dataset]
        
        print("\nClass distribution in training set:")
        for i in range(10):
            count = train_labels.count(i)
            print(f"  Class {i}: {count} samples ({count/len(train_labels)*100:.1f}%)")
        
        return {
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'shape': self.train_dataset[0][0].shape
        }