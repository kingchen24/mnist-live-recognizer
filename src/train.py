import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import os
from tqdm import tqdm

from model import create_model
from dataset import MNISTDataLoader
from utils import ModelCheckpoint

class Trainer:
    """训练器类"""
    
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 设置随机种子
        self.set_seed(config.get('seed', 42))
        
        # 初始化数据加载器
        self.data_loader = MNISTDataLoader(
            batch_size=config['batch_size'],
            download=config.get('download', True)
        )
        
        self.train_loader, self.test_loader = self.data_loader.get_dataloaders()
        
        # 初始化模型
        self.model = create_model(
            model_name=config['model_name'],
            num_classes=10,
            device=self.device
        )
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
        #    verbose=True
        )
        
        # 模型检查点
        self.checkpoint = ModelCheckpoint(
            save_dir=config.get('save_dir', './models'),
            model_name=config['model_name']
        )
        
        # TensorBoard记录器
        self.writer = SummaryWriter(
            log_dir=f"runs/{config['model_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        self.best_val_acc = 0.0
        
    def set_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
            
            # TensorBoard记录（每100个batch）
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/Loss_batch', loss.item(), step)
                self.writer.add_scalar('Train/Acc_batch', 100 * correct / total, step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """主训练循环"""
        print(f"Starting training with {self.config['model_name']}...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 学习率调度
            self.scheduler.step(val_acc)
            
            # 保存历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # TensorBoard记录
            self.writer.add_scalar('Train/Loss_epoch', train_loss, epoch)
            self.writer.add_scalar('Train/Acc_epoch', train_acc, epoch)
            self.writer.add_scalar('Val/Loss_epoch', val_loss, epoch)
            self.writer.add_scalar('Val/Acc_epoch', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 打印进度
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            # 保存检查点
            if (epoch + 1) % self.config.get('save_interval', 5) == 0 or is_best:
                self.checkpoint.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    accuracy=val_acc,
                    loss=val_loss,
                    is_best=is_best
                )
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # 关闭TensorBoard写入器
        self.writer.close()
        
        return self.history
    
    def evaluate(self):
        """最终评估"""
        print("\n=== Final Evaluation ===")
        val_loss, val_acc = self.validate()
        print(f"Final Validation Accuracy: {val_acc:.2f}%")
        print(f"Final Validation Loss: {val_loss:.4f}")
        
        return val_loss, val_acc

def main():
    """主函数"""
    # 配置参数
    config = {
        'model_name': 'simple_cnn',  # 可选: 'simple_cnn', 'improved_cnn'
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 20,
        'weight_decay': 1e-5,
        'save_dir': './models',
        'download': True,
        'seed': 42
    }
    
    # 创建训练器并训练
    trainer = Trainer(config)
    history = trainer.train()
    
    # 最终评估
    trainer.evaluate()
    
    return trainer, history

if __name__ == '__main__':
    trainer, history = main()