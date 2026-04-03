from ultralytics import YOLO

def main():
    model = YOLO('runs/hazelnut_for_d455cap_yatb/weights/best.pt')
    model.export(
        format='onnx', 
        imgsz=960, 
        device=0,
        simplify=True,
        opset=11
    )
    
if __name__ == '__main__':
    main()