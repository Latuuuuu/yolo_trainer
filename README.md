# yolo_trainer
This repository provides a tool for training **Ultralytics YOLOv11** models using Docker.

***This tool requires GPU***

## Dataset Preparation
1. Place your dataset folder inside the datasets/ directory
2. Change the path to your yaml file
```bash
data='/usr/src/datasets/hazelnut_aruco_obb/data.yaml'
```
3. Maybe change some train args
## Run Training
Navigate to the docker directory and start the container
```bash
cd docker
docker compose up --build
```
Results will be place inside runs/ directory