from ultralytics import YOLO

def main():
    model = YOLO('yolo11n-obb.pt')

    results = model.train(
        data='/usr/src/datasets/hazelnut_aruco_obb/data.yaml', # check the path to yaml file
        epochs=300,
        imgsz=640,
        degrees=15.0,
        scale=0.5,
        device=0,
        perspective=0.001,
        project='runs',
        name='hazelnut_experiment' # model name
    )

if __name__ == '__main__':
    main()