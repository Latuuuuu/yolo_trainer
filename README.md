# yolo_trainer
This repository provides a tool for training **Ultralytics YOLOv11** models using Docker.

***This tool requires GPU***
## Run Container
```bash
cd docker
docker compose build
docker compose up -d
docker exec -it yolo_trainer bash
```
## Dataset Preparation
1. Place your dataset folder inside the datasets/ directory
2. Change the path to your yaml file
```bash
data='/usr/src/datasets/hazelnut_aruco_obb/data.yaml'
```
3. Maybe change some train args

## Run Training
```bash
python train.py
```
## Run Predict
```bash
python predict.py
```