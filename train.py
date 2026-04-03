from ultralytics import YOLO
# from models.scconv import SCConv
# import albumentations as A
# import torch.nn as nn
# import ultralytics.nn.modules as modules
# setattr(modules, 'SCConv', SCConv)

# custom_transforms = [
#     A.ToGray(p=1.0, num_output_channels=3), # 改回 3，內容依然是黑白
# ]

def main():
    model = YOLO('yolo11n.pt')

    model.train(
        data='/usr/src/datasets/table_tennis/data.yaml',
        epochs=300,
        imgsz=1280,
        device=0,
        project='/usr/src/runs',
        name='table_tennis_1280',
        batch=8,
        patience=150,

        close_mosaic=50,
    )
    # model = YOLO('yolo11n-obb.pt')

    # model.train(
    #     data='/usr/src/datasets/hazelnut_d455_gray_52/data.yaml',
    #     epochs=300,
    #     imgsz=1280,
    #     device=0,
    #     project='/usr/src/app/runs',
    #     name='hazelnut_for_d455cap_',
    #     pretrained='runs/hazelnut_rgb_best/weights/best.pt',
    #     optimizer='SGD',
    #     lr0=0.005,
    #     lrf=0.01,
    #     # cache=False,
    #     # amp=True,

    #     momentum=0.937,
    #     weight_decay=0.0005,
    #     warmup_epochs=3.0,
    #     cos_lr=True,
    #     box=10.0,

    #     batch=8,
    #     hsv_s=0.0,
    #     # hsv_v=0.6,
    #     mosaic=0.5,
    #     close_mosaic=10,
    #     degrees=180.0,
    #     scale=0.05,
    #     shear=0.0,
    #     perspective=0.0005,
    # )

if __name__ == '__main__':
    main()