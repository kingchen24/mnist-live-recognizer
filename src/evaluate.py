import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from model import create_model
from dataset import MNISTDataLoader
from utils import plot_confusion_matrix, visualize_predictions

class Evaluator:
    """评估器类"""
    
    def __init__(self, model_path, model_name='simple_cnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, model_name)
        self.data_loader = MNISTDataLoader(batch_size=64)
        
    def load_model(self, model_path, model_name):
        """加载训练好的模型"""
        model = create_model(model_name, num_classes=10, device=self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Training accuracy: {checkpoint['accuracy']:.2f}%")
        
        return model
    
    def compute_accuracy(self, data_loader):
        """计算准确率"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Accuracy on dataset: {accuracy:.2f}%")
        
        return accuracy
    
    def analyze_errors(self, data_loader, num_samples=20):
        """分析错误样本"""
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                
                # 找出错误预测
                mask = predictions != labels
                error_indices = torch.where(mask)[0]
                
                for idx in error_indices:
                    errors.append({
                        'image': images[idx].cpu(),
                        'true_label': labels[idx].item(),
                        'pred_label': predictions[idx].item(),
                        'probabilities': torch.softmax(outputs[idx], dim=0).cpu().numpy()
                    })
                    
                    if len(errors) >= num_samples:
                        break
                
                if len(errors) >= num_samples:
                    break
        
        # 显示错误样本
        self.display_errors(errors)
        
        return errors
    
    def display_errors(self, errors):
        """显示错误预测的样本"""
        num_errors = len(errors)
        rows = int(np.ceil(num_errors / 5))
        
        fig, axes = plt.subplots(rows, 5, figsize=(15, rows * 3))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, error in enumerate(errors):
            row = idx // 5
            col = idx % 5
            
            ax = axes[row, col]
            img = error['image'].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            
            title = f"True: {error['true_label']}\nPred: {error['pred_label']}"
            ax.set_title(title, color='red', fontsize=10)
            ax.axis('off')
            
            # 显示概率分布
            probs = error['probabilities']
            top3_idx = np.argsort(probs)[-3:][::-1]
            for i, prob_idx in enumerate(top3_idx):
                ax.text(28, 5 + i*4, f"{prob_idx}: {probs[prob_idx]:.2f}", 
                       fontsize=8, verticalalignment='top')
        
        # 隐藏多余的子图
        for idx in range(num_errors, rows * 5):
            row = idx // 5
            col = idx % 5
            axes[row, col].axis('off')
        
        plt.suptitle('Misclassified Samples', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def get_per_class_accuracy(self, data_loader):
        """获取每个类别的准确率"""
        self.model.eval()
        class_correct = [0] * 10
        class_total = [0] * 10
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predictions = torch.max(outputs, 1)
                
                for i in range(10):
                    mask = labels == i
                    if mask.any():
                        class_total[i] += mask.sum().item()
                        class_correct[i] += (predictions[mask] == labels[mask]).sum().item()
        
        # 打印每个类别的准确率
        print("\n=== Per-class Accuracy ===")
        for i in range(10):
            accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"Class {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # 绘制条形图
        plt.figure(figsize=(10, 6))
        accuracies = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 
                     for i in range(10)]
        
        bars = plt.bar(range(10), accuracies)
        plt.xlabel('Digit Class')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy per Class')
        plt.xticks(range(10))
        plt.ylim([0, 100])
        
        # 在条形上显示数值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return class_correct, class_total
    
    def run_full_evaluation(self):
        """运行完整的评估流程"""
        print("=== Starting Full Evaluation ===")
        
        # 获取数据加载器
        _, test_loader = self.data_loader.get_dataloaders()
        
        # 1. 计算总体准确率
        print("\n1. Overall Accuracy:")
        accuracy = self.compute_accuracy(test_loader)
        
        # 2. 每个类别的准确率
        self.get_per_class_accuracy(test_loader)
        
        # 3. 混淆矩阵
        print("\n3. Confusion Matrix:")
        plot_confusion_matrix(self.model, test_loader, self.device)
        
        # 4. 分析错误样本
        print("\n4. Error Analysis:")
        errors = self.analyze_errors(test_loader, num_samples=15)
        
        # 5. 可视化一些预测样本
        print("\n5. Sample Predictions:")
        visualize_predictions(self.model, test_loader, self.device, num_samples=9)
        
        return accuracy

def main():
    """主函数"""
    # 模型路径
    model_path = "./models/simple_cnn_best.pth"
    
    # 创建评估器
    evaluator = Evaluator(model_path, model_name='simple_cnn')
    
    # 运行评估
    accuracy = evaluator.run_full_evaluation()
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Final Model Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()