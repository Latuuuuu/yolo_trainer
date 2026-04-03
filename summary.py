from ultralytics import YOLO
from models.scconv import SCConv
import ultralytics.nn.modules as modules

# 1. 註冊你的 SCConv 模組
setattr(modules, 'SCConv', SCConv)

if __name__ == "__main__":
    # 2. 載入模型 (確保 YAML 裡已經寫了 ch: 1)
    model_path = 'models/yolo11-obb-ch1.yaml'
    model = YOLO(model_path)
    
    # 3. 使用官方內建的 info 方法
    # imgsz=640 會自動計算在 640x640 下的 GFLOPs
    print("\n" + "="*40)
    print(f"正在分析模型: {model_path}")
    print("="*40)
    
    # detailed=True 會列出每一層的細節，方便你確認 SCConv 是否被正確載入
    model.info(detailed=True, imgsz=640)
    
    print("\n分析完成！請查看上方輸出中的 'GFLOPs' 與 'parameters' 數據。")