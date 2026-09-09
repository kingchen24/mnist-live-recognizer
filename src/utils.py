import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from datetime import datetime
import os

class ModelCheckpoint:
    """模型检查点保存"""
    
    def __init__(self, save_dir='./models', model_name='model'):
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(save_dir, exist_ok=True)
        
    def save(self, model, optimizer, epoch, accuracy, loss, is_best=False):
        """保存模型"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'loss': loss
        }
        
        # 常规保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{self.model_name}_{timestamp}.pth'
        path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, path)
        
        # 如果是最好模型，额外保存
        if is_best:
            best_path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
        
        print(f"Model saved to {path}")
        
    def load(self, model, optimizer=None, path=None):
        """加载模型"""
        if path is None:
            # 加载最好模型
            path = os.path.join(self.save_dir, f'{self.model_name}_best.pth')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Model loaded from {path}")
        print(f"Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['accuracy']:.2f}%")
        
        return checkpoint['epoch'], checkpoint['accuracy']

def visualize_predictions(model, data_loader, device='cpu', num_samples=9):
    """可视化模型预测结果"""
    model.eval()
    images, labels = next(iter(data_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # 创建可视化
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    
    for idx in range(min(num_samples, len(images))):
        img = images[idx].cpu().squeeze().numpy()
        true_label = labels[idx].item()
        pred_label = predictions[idx].item()
        
        axes[idx].imshow(img, cmap='gray')
        axes[idx].set_title(f'True: {true_label}, Pred: {pred_label}', 
                           color='green' if true_label == pred_label else 'red')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(train_losses, label='Training Loss')
    axes[0].plot(val_losses, label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(train_accs, label='Training Accuracy')
    axes[1].plot(val_accs, label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, data_loader, device='cpu'):
    """绘制混淆矩阵"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, 
                                target_names=[str(i) for i in range(10)]))
    
    return cm