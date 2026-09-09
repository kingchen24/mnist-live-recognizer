"""
使用 OpenCV 实现的 MNIST 实时手写数字识别演示 - 最终修复版
运行: python src/realtime_opencv.py
"""

import sys
import os

# ====== 关键：动态设置模块导入路径 ======
# 获取当前脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（假设脚本在 src/ 目录下，上一级就是项目根目录）
project_root = os.path.dirname(script_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"[信息] 已将项目根目录添加到路径: {project_root}")

try:
    from src.model import create_model
    print("[信息] 成功导入模型模块")
except ImportError as e:
    print(f"[错误] 导入失败: {e}")
    print("请检查：1. 项目结构是否正确 2. src/model.py 文件是否存在")
    sys.exit(1)
# ====== 路径设置结束 ======

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class OpenCVDemo:
    def __init__(self, model_path='models/simple_cnn_best.pth', width=400, height=550):
        """
        初始化实时识别演示 (修复版)
        Args:
            model_path: 模型文件路径，相对于项目根目录
            width: 绘图窗口基础宽度
            height: 绘图窗口基础高度
        """
        # ====== 1. 动态路径设置 ======
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)

        if not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        print(f"[信息] 项目根目录: {project_root}")
        print(f"[信息] 模型文件路径: {model_path}")

        # ====== 2. 设置计算设备 ======
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[信息] 使用设备: {self.device}")
        
        # ====== 3. 加载模型 ======
        try:
            self.model = create_model('simple_cnn', num_classes=10, device=self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            accuracy = checkpoint.get('accuracy', 'N/A')
            print(f"[信息] 模型加载成功")
            print(f"[信息] 训练准确率: {accuracy}%")
        except FileNotFoundError:
            print(f"[错误] 找不到模型文件: {model_path}")
            print("请先运行 python src/train.py 训练模型")
            raise
        except Exception as e:
            print(f"[错误] 模型加载失败: {e}")
            raise

        # ====== 4. 初始化画布和窗口尺寸参数 ======
        if width <= 0 or height <= 0:
            print(f"[警告] 传入的基础尺寸无效({width}x{height})，使用默认值(400x550)")
            width, height = 400, 550
        
        self.base_width = width
        self.base_height = height
        self.current_width = max(1, self.base_width)
        self.current_height = max(1, self.base_height)
        
        # 创建基础画布（白色背景）
        self.canvas = np.ones((self.base_height, self.base_width, 3), dtype=np.uint8) * 255
        
        # 绘图相关参数
        self.drawing = False
        self.last_point = None
        self.brush_radius = 15
        
        # 标记是否需要更新显示
        self.need_update = True
        
        # 修复：初始化 manual_trigger 属性
        self.manual_trigger = False
        
        # 空白画布检测阈值
        self.empty_canvas_threshold = 50  # 黑色像素少于50个认为是空白画布
        
        # 键盘事件处理标志
        self.s_key_pressed = False
        
        print(f"[信息] 基础画布尺寸: {self.base_width}x{self.base_height}")
        print(f"[信息] 初始窗口尺寸: {self.current_width}x{self.current_height}")
        print(f"[信息] 空白画布检测阈值: {self.empty_canvas_threshold} 像素")
        print("[信息] OpenCVDemo 初始化完成")

    def mouse_callback(self, event, x, y, flags, param):
        """
        鼠标回调函数 (修复版：正确转换坐标并绘制右侧百分比)
        """
        # 计算右侧面板宽度（动态）
        panel_width_ratio = 0.45
        panel_width = int(self.current_width * panel_width_ratio)
        drawing_area_width = self.current_width - panel_width
        
        # 转换窗口坐标到基础画布坐标
        scale_x = drawing_area_width / self.base_width
        scale_y = self.current_height / self.base_height
        
        # 只有当鼠标在左侧绘图区内才处理
        if x < drawing_area_width and y < self.current_height:
            # 将窗口坐标转换回基础画布坐标
            canvas_x = int(x / scale_x) if scale_x > 0 else x
            canvas_y = int(y / scale_y) if scale_y > 0 else y
            
            # 确保坐标在有效范围内
            canvas_x = max(0, min(canvas_x, self.base_width - 1))
            canvas_y = max(0, min(canvas_y, self.base_height - 1))
            
            # 动态计算画笔半径
            current_brush_radius = int(self.brush_radius * min(scale_x, scale_y))
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.last_point = (canvas_x, canvas_y)
                # 在按下位置画一个点
                cv2.circle(self.canvas, (canvas_x, canvas_y), 
                          current_brush_radius, (0, 0, 0), -1)
                self.need_update = True
                self.manual_trigger = False  # 鼠标操作不是手动触发
                
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing and self.last_point:
                    cv2.line(self.canvas, self.last_point, (canvas_x, canvas_y),
                            (0, 0, 0), current_brush_radius * 2, cv2.LINE_AA)
                    self.last_point = (canvas_x, canvas_y)
                    self.need_update = True
                    
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.last_point = None
                # 松开鼠标时立即预测一次
                self.need_update = True
                self.manual_trigger = False  # 鼠标操作不是手动触发

    def draw_line(self, img, start, end):
        """在图像上画线"""
        cv2.line(img, start, end, (0, 0, 0), self.brush_radius * 2, cv2.LINE_AA)
        
    def preprocess_canvas(self):
        """预处理画布图像，匹配MNIST输入格式"""
        # 1. 转为灰度图并缩放到28x28
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 2. 反转颜色（MNIST是黑底白字，我们画的是白底黑字）
        inverted = 255 - resized
        
        # 3. 归一化并转换为Tensor
        normalized = inverted / 255.0
        normalized = (normalized - 0.1307) / 0.3081  # MNIST的标准化参数
        
        tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device), inverted
    
    def predict_and_display(self):
        """进行预测并显示结果 (修复版：改进窗口适应布局)"""
        # ====== 0. 确保窗口尺寸有效 ======
        if self.current_width <= 10 or self.current_height <= 10:
            self.current_width = self.base_width
            self.current_height = self.base_height

        # 检查是否为空白画布
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 50)  # 阈值50，小于50的像素视为黑色
        is_empty_canvas = black_pixels < self.empty_canvas_threshold
        
        if is_empty_canvas:
            # 空白画布：强制预测为0
            predicted = 0
            probabilities = torch.zeros(10)
            probabilities[0] = 1.0  # 数字0的置信度为100%
            processed_img = np.zeros((28, 28), dtype=np.uint8)  # 全黑的28x28图像
        else:
            # 非空白画布：正常预测
            with torch.no_grad():
                input_tensor, processed_img = self.preprocess_canvas()
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                predicted = torch.argmax(probabilities).item()

        # 如果是手动触发，打印详细信息
        if self.manual_trigger:
            print(f"[手动识别] 预测数字: {predicted}")
            print(f"[手动识别] 画布黑色像素数: {black_pixels}")
            print(f"[手动识别] 空白画布判断: {'是' if is_empty_canvas else '否'}")
            
            if is_empty_canvas:
                print(f"[手动识别] 检测到空白画布，强制预测为0")
            else:
                for i, prob in enumerate(probabilities):
                    print(f"  {i}: {prob.item()*100:.2f}%")
            print("-" * 40)

        # ====== 1. 计算布局参数 ======
        # 面板宽度比例
        PANEL_WIDTH_RATIO = 0.45
        PANEL_WIDTH = int(self.current_width * PANEL_WIDTH_RATIO)
        PANEL_WIDTH = max(250, PANEL_WIDTH)  # 最小宽度确保有空间显示百分比
        
        # 绘图区域宽度
        drawing_area_width = self.current_width - PANEL_WIDTH
        
        # ====== 2. 画布缩放 ======
        # 缩放比例：将基础画布缩放到绘图区域大小
        scale_x = drawing_area_width / self.base_width
        scale_y = self.current_height / self.base_height
        
        # 确保缩放比例有效
        if scale_x <= 0 or scale_y <= 0:
            scale_x = scale_y = 1.0
            drawing_area_width = self.base_width
        
        # 缩放画布到绘图区域大小
        try:
            display_canvas = cv2.resize(self.canvas, (drawing_area_width, self.current_height))
        except:
            # 如果缩放失败，使用原始画布
            display_canvas = self.canvas.copy()
            scale_x = scale_y = 1.0

        # ====== 3. 创建信息面板 ======
        info_panel = np.ones((self.current_height, PANEL_WIDTH, 3), dtype=np.uint8) * 240
        
        # 动态计算区域高度（基于当前窗口高度）
        # 使用固定的像素值而不是比例，确保在窗口缩小时仍然可读
        TITLE_HEIGHT = max(70, int(70 * (self.current_height / self.base_height)))
        CONFIDENCE_HEIGHT = max(300, int(300 * (self.current_height / self.base_height)))
        PREVIEW_HEIGHT = max(150, int(150 * (self.current_height / self.base_height)))
        
        # 确保高度不超过面板高度
        total_needed_height = TITLE_HEIGHT + CONFIDENCE_HEIGHT + PREVIEW_HEIGHT
        if total_needed_height > self.current_height:
            # 按比例缩小各个区域
            scale_factor = self.current_height / total_needed_height
            TITLE_HEIGHT = int(TITLE_HEIGHT * scale_factor)
            CONFIDENCE_HEIGHT = int(CONFIDENCE_HEIGHT * scale_factor)
            PREVIEW_HEIGHT = self.current_height - TITLE_HEIGHT - CONFIDENCE_HEIGHT

        # ====== 4. 区域 A: 动态标题 ======
        title_font_scale = 1.2 * min(scale_x, scale_y, 1.5)  # 根据缩放调整字体大小
        title_thickness = max(2, int(2 * min(scale_x, scale_y, 1.5)))
        
        # 修复：使用英文避免乱码问题
        if is_empty_canvas:
            title_text = f"Prediction: {predicted} (Empty)"
        elif self.manual_trigger:
            title_text = f"Prediction: {predicted} (Manual)"
        else:
            title_text = f"Prediction: {predicted}"
        
        (text_width, text_height), _ = cv2.getTextSize(
            title_text, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_thickness
        )
        title_x = (PANEL_WIDTH - text_width) // 2
        title_y = TITLE_HEIGHT // 2 + text_height // 2
        
        # 确保标题在面板范围内
        if title_x < 0:
            title_x = 10
        if title_y >= TITLE_HEIGHT:
            title_y = TITLE_HEIGHT - 10
        
        # 修正颜色：OpenCV使用BGR顺序
        if is_empty_canvas:
            title_color = (0, 0, 255)  # 红色，表示空白画布
        elif self.manual_trigger:
            title_color = (255, 0, 0)  # 蓝色，表示手动触发
        else:
            title_color = (0, 100, 0)  # 深绿色，正常预测
        
        cv2.putText(info_panel, title_text, (title_x, title_y),
                    cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, title_color, title_thickness)

        # ====== 5. 区域 B: 置信度条和百分比 ======
        # 计算置信度区域起始Y坐标
        confidence_start_y = TITLE_HEIGHT + 20
        
        # 根据缩放调整条形尺寸
        bar_height = max(15, int(20 * min(scale_x, scale_y, 1.5)))
        bar_spacing = max(25, int(28 * min(scale_x, scale_y, 1.5)))
        bar_max_width = max(150, int(180 * min(scale_x, scale_y, 1.5)))
        font_scale = max(0.5, 0.7 * min(scale_x, scale_y, 1.5))
        
        # 绘制置信度条
        for i in range(10):
            prob = probabilities[i].item() * 100
            bar_width = int(prob * bar_max_width / 100.0)
            
            y = confidence_start_y + i * bar_spacing
            
            # 确保在面板范围内
            if y + bar_height > confidence_start_y + CONFIDENCE_HEIGHT:
                break
            
            # 1. 绘制条形背景
            bg_start_x = 20
            cv2.rectangle(info_panel,
                         (bg_start_x, y),
                         (bg_start_x + bar_max_width, y + bar_height),
                         (220, 220, 220), -1, cv2.LINE_AA)

            # 2. 绘制前景概率条
            color = (0, 180, 0) if i == predicted else (150, 150, 150)
            if bar_width > 0:
                cv2.rectangle(info_panel,
                             (bg_start_x, y),
                             (bg_start_x + bar_width, y + bar_height),
                             color, -1, cv2.LINE_AA)

            # 3. 绘制数字标签
            text_y = y + (bar_height // 2) + 5
            
            # a) 左侧数字标签
            cv2.putText(info_panel, f"{i}:",
                       (5, text_y),
                       cv2.FONT_HERSHEY_DUPLEX,
                       font_scale,
                       (0, 0, 0),
                       max(1, int(1.5 * min(scale_x, scale_y, 1.5))),
                       cv2.LINE_AA)

            # b) 右侧百分比
            percent_text = f"{prob:.1f}%"
            (text_width, _), _ = cv2.getTextSize(
                percent_text, 
                cv2.FONT_HERSHEY_DUPLEX,
                font_scale,
                max(1, int(1.5 * min(scale_x, scale_y, 1.5)))
            )
            
            # 文本位置：在条形区域右侧
            text_x = bg_start_x + bar_max_width + 10
            
            # 如果文本会超出面板，调整位置
            if text_x + text_width > PANEL_WIDTH - 10:
                text_x = PANEL_WIDTH - text_width - 10
            
            cv2.putText(info_panel, percent_text,
                       (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX,
                       font_scale,
                       (0, 0, 0),
                       max(1, int(1.5 * min(scale_x, scale_y, 1.5))),
                       cv2.LINE_AA)

        # ====== 6. 区域 C: 小图预览 ======
        # 计算预览区域起始Y坐标
        preview_start_y = TITLE_HEIGHT + CONFIDENCE_HEIGHT
        
        # 动态计算预览图大小
        preview_size = max(80, int(100 * min(scale_x, scale_y, 1.5)))
        preview_x = (PANEL_WIDTH - preview_size) // 2
        preview_y = preview_start_y + (PREVIEW_HEIGHT - preview_size) // 2
        
        # 确保预览图在面板范围内
        if (preview_y + preview_size <= self.current_height and
            preview_y >= 0 and
            preview_size > 10):

            # 绘制白色背景框
            border_thickness = max(2, int(4 * min(scale_x, scale_y, 1.5)))
            cv2.rectangle(info_panel,
                         (preview_x - border_thickness, preview_y - border_thickness),
                         (preview_x + preview_size + border_thickness,
                          preview_y + preview_size + border_thickness),
                         (255, 255, 255), -1, cv2.LINE_AA)

            # 处理小图
            small_display = cv2.resize(processed_img,
                                      (preview_size, preview_size),
                                      interpolation=cv2.INTER_NEAREST)
            small_display_color = cv2.cvtColor(small_display, cv2.COLOR_GRAY2BGR)

            # 将清晰的小图放置到面板
            info_panel[preview_y:preview_y + preview_size,
                      preview_x:preview_x + preview_size] = small_display_color

            # 添加小图标签
            label_font_scale = max(0.4, 0.5 * min(scale_x, scale_y, 1.5))
            label_y = preview_y + preview_size + int(20 * min(scale_x, scale_y, 1.5))
            if label_y < self.current_height - 5:
                # 修复：使用英文避免乱码问题
                label_text = "Empty Canvas" if is_empty_canvas else "Model Input"
                cv2.putText(info_panel, label_text,
                           (preview_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           label_font_scale,
                           (100, 100, 100) if not is_empty_canvas else (0, 0, 255),  # 空白画布时用红色
                           max(1, int(1.2 * min(scale_x, scale_y, 1.5))),
                           cv2.LINE_AA)

        # ====== 7. 合并并显示最终图像 ======
        combined = np.hstack([display_canvas, info_panel])
        cv2.imshow("MNIST Real-time Digit Recognition (OpenCV)", combined)
        
        # 标记为已更新
        self.need_update = False
    
    def run(self):
        """主运行循环 (修复版：改进键盘事件处理)"""
        window_name = "MNIST Real-time Digit Recognition (OpenCV)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        initial_width = self.base_width + int(self.base_width * 0.45)
        cv2.resizeWindow(window_name, initial_width, self.base_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("\n" + "="*60)
        print("MNIST 实时手写数字识别演示 (OpenCV版 - 修复版)")
        print("="*60)
        print("使用说明：")
        print("  1. 拖动窗口边框可调整窗口大小，界面元素会自动缩放")
        print("  2. 在左侧白色区域用鼠标绘制数字")
        print("  3. 按 'c' 键清除画布")
        print("  4. 按 's' 键手动触发识别 (现在有效!)")
        print("  5. 按 'ESC' 键或关闭窗口退出")
        print("  注意：空白画布将自动预测为0")
        print("="*60 + "\n")

        # 初始显示一次
        self.predict_and_display()

        # 用于跟踪上次按键状态
        last_key_state = {}
        
        while True:
            # 1. 获取窗口尺寸
            try:
                window_rect = cv2.getWindowImageRect(window_name)
                if window_rect and len(window_rect) == 4:
                    current_window_width = window_rect[2]
                    current_window_height = window_rect[3]
                    
                    if (current_window_width > 10 and current_window_height > 10 and 
                        (abs(current_window_width - self.current_width) > 2 or 
                         abs(current_window_height - self.current_height) > 2)):
                        self.current_width = current_window_width
                        self.current_height = current_window_height
                        self.need_update = True
            except:
                pass
            
            # 2. 只有在需要更新时才重绘界面
            if self.need_update:
                self.predict_and_display()
            
            # 3. 改进的键盘输入处理
            # 使用更长的等待时间以确保键盘事件被正确捕获
            key = cv2.waitKey(30) & 0xFF
            
            # 检测ESC键
            if key == 27:  # ESC 键
                print("[信息] 用户按下ESC键，退出程序")
                break
            
            # 检测'c'键：清除画布
            elif key == ord('c'):  # 'c' 键
                self.canvas = np.ones((self.base_height, self.base_width, 3), dtype=np.uint8) * 255
                print("[信息] 画布已清除")
                self.need_update = True
                self.manual_trigger = False  # 清除不是手动触发
            
            # 检测's'键：手动触发识别
            elif key == ord('s'):  # 's' 键
                # 添加防抖：只有当上次按键不是's'时才触发
                if not last_key_state.get('s', False):
                    print("[信息] *** 手动触发识别 ***")
                    # 设置需要更新，并标记为手动触发
                    self.need_update = True
                    self.manual_trigger = True
                last_key_state['s'] = True
            else:
                last_key_state['s'] = False
            
            # 4. 处理Shift键问题：明确忽略Shift键
            # Shift键的key值是16（左Shift）或225（右Shift）
            if key in [16, 225]:
                # 忽略Shift键，不执行任何操作
                pass

        cv2.destroyAllWindows()
        print("[信息] 程序已安全退出。")

if __name__ == "__main__":
    try:
        demo = OpenCVDemo(model_path='models/simple_cnn_best.pth')
        demo.run()
    except KeyboardInterrupt:
        print("\n[信息] 程序被用户中断")
    except Exception as e:
        print(f"\n[错误] 程序运行出错: {e}")
        import traceback
        traceback.print_exc()