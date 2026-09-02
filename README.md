# yolo_trainer

A ready-to-use [Ultralytics YOLO](https://docs.ultralytics.com/) training environment in Docker.
Clone it, run one script, and start training — no CUDA, cuDNN or PyTorch installation on your machine.

繁體中文說明：[README.zh-TW.md](README.zh-TW.md)

- Detection and **OBB** (oriented bounding box) training, prediction and export
- All settings live in `configs/*.yaml` — you never edit the training code
- Results land in `runs/`, owned by your user (not root)

## Requirements

| | |
|---|---|
| GPU | An NVIDIA GPU. **CPU-only machines cannot train here.** |
| Driver | NVIDIA driver installed (`nvidia-smi` works) |
| Docker | [Docker Engine](https://docs.docker.com/engine/install/) + the `docker compose` plugin |
| Runtime | [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so containers can use the GPU |
| Disk | ~20 GB for the image |

## Quick start

```bash
git clone <this-repo> && cd yolo_trainer
bash setup.sh                       # checks the host, builds the image, starts the container
docker exec -it yolo_trainer bash   # you are now inside, in /usr/src
```

`setup.sh` verifies your driver and the container runtime, writes `docker/.env` with your user id
(so files created by the container belong to you), allows X11 windows, then builds and starts
everything. It is safe to re-run; `bash setup.sh --no-build` skips the rebuild.

Everything below runs **inside the container**.

## 1. Prepare a dataset

Put your dataset folder in `datasets/`. The YOLO format (what Roboflow exports) looks like this:

```
datasets/my_dataset/
├── data.yaml
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}      # optional
```

`data.yaml` needs three keys — paths are relative to the folder holding the file:

```yaml
train: train/images
val: valid/images
test: test/images

names:
  0: player
  1: ball
```

For OBB the labels are 8 numbers per line (four corner points) instead of 4; export from Roboflow
as **YOLOv8 Oriented Bounding Boxes** and train with an `-obb` model.

## 2. Train

Copy a config, edit it, run it. You should not need to touch `train.py`.

```bash
cp configs/example_detect.yaml configs/my_experiment.yaml
# edit `data:` to point at your dataset, and `name:` to name the run
python train.py --config configs/my_experiment.yaml
```

A config has three parts:

```yaml
model: yolo11n.pt                      # weights (.pt) or architecture (models/*.yaml)
data: datasets/my_dataset/data.yaml    # relative to the repository root
args:                                  # anything from the Ultralytics train settings
  epochs: 300
  imgsz: 640
  batch: 8
  name: my_experiment
```

Shipped examples: [`configs/example_detect.yaml`](configs/example_detect.yaml) (plain detection) and
[`configs/example_obb.yaml`](configs/example_obb.yaml) (OBB fine-tuning with a hand-tuned SGD schedule).

Frequently changed arguments:

| Argument | What it does |
|---|---|
| `epochs` | How many passes over the dataset |
| `imgsz` | Training resolution. 1280 helps with small objects, costs ~4× the time and memory of 640 |
| `batch` | Images per step. **Lower this first when you run out of GPU memory** |
| `device` | `0`, `0,1`, or `cpu` |
| `patience` | Stop early after N epochs without improvement |
| `optimizer`, `lr0`, `cos_lr` | Learning-rate schedule; `auto` is fine to start with |
| `degrees`, `scale`, `hsv_*`, `mosaic` | Augmentation — match these to how your real images vary |
| `close_mosaic` | Turn mosaic off for the last N epochs so the model finishes on realistic images |

Full list: <https://docs.ultralytics.com/modes/train/#train-settings>

Override anything from the command line without editing the config:

```bash
python train.py --config configs/my_experiment.yaml --set epochs=1 --name smoke_test   # quick check
python train.py --config configs/my_experiment.yaml --set lr0=0.001 --set batch=4
python train.py --config configs/my_experiment.yaml --resume                           # continue runs/<name>
python train.py --config configs/my_experiment.yaml --resume runs/other_run/weights/last.pt
```

Results (weights, curves, confusion matrix, sample batches) go to `runs/<name>/`.
The two files to look at first are `runs/<name>/results.png` and `confusion_matrix.png`;
the trained model is `runs/<name>/weights/best.pt`.

To fine-tune from your own checkpoint, point `model:` at it:
`model: runs/my_previous_run/weights/best.pt`.

## 3. Predict

```bash
# save annotated images to runs/predict/ (works over SSH, no display needed)
python predict.py --weights runs/my_run/weights/best.pt --source imgs/my_photos

# or open a window per image (needs X11; any key = next, q = quit)
python predict.py --weights runs/my_run/weights/best.pt --source imgs/my_photos --show
```

Useful flags: `--conf 0.6` (confidence threshold), `--imgsz`, `--line-width`, `--hide-conf`, `--device cpu`.
`--source` takes a single image, a folder, or a camera index such as `0`.

## 4. Export

```bash
python export.py --weights runs/my_run/weights/best.pt                       # ONNX, imgsz 640
python export.py --weights runs/my_run/weights/best.pt --format engine --imgsz 960 --half
```

The exported file appears next to the `.pt`. Formats: <https://docs.ultralytics.com/modes/export/#export-formats>

## Troubleshooting

| Symptom | Fix |
|---|---|
| `setup.sh`: *Docker cannot see the NVIDIA runtime* | Install nvidia-container-toolkit, then `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| `permission denied` on the Docker socket | `sudo usermod -aG docker $USER`, then log out and back in |
| `torch.cuda.is_available()` is False inside the container | Re-run `bash setup.sh`; check `docker exec -it yolo_trainer nvidia-smi` |
| `CUDA out of memory` | Lower `batch` (8 → 4 → 2), then `imgsz` (1280 → 640), and set `workers: 4` |
| `--show` does nothing / `cannot open display` | Run `xhost +local:docker` on the **host** (it resets after reboot), or drop `--show` and use the saved images |
| `QFontDatabase: Cannot find font directory .../cv2/qt/fonts` | Harmless — it comes from the OpenCV Qt build and does not affect the window |
| Files in `runs/` are owned by root | `docker/.env` has the wrong ids — re-run `bash setup.sh`, then `cd docker && docker compose up -d --force-recreate` |
| `Dataset config not found` | The script lists the datasets it can see; check the path in your config is relative to the repository root |
| Downloading `yolo11n.pt` fails | Download it manually from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) into `weights/` |
| Training is very slow | Check `nvidia-smi` during training — if utilisation is low the dataloader is the bottleneck: raise `workers`, or use smaller images |
| Edits to `train.py` seem to have no effect | You are editing a different copy — the repository is mounted at `/usr/src` inside the container |

## Custom architectures (advanced)

`models/` is a local, git-ignored folder for your own model definitions: drop a modified
architecture YAML in there and point a config at it to train it from scratch.

```yaml
model: models/my_architecture.yaml
```

Start from an official Ultralytics YAML (e.g. `yolo11.yaml`, `yolo11-obb.yaml`) and edit a copy.

## Layout

```
yolo_trainer/
├── setup.sh            one-shot host setup + container start
├── configs/            training configs — this is what you edit
├── train.py            training entry point
├── predict.py          inference on images / folders / camera
├── export.py           ONNX / TensorRT export
├── common.py           shared path handling and error messages
├── docker/             Dockerfile, compose.yaml, .env (generated)
├── models/             your own architecture YAMLs (git-ignored)
├── datasets/           your datasets      (git-ignored)
├── imgs/               images to predict  (git-ignored)
├── runs/               training results   (git-ignored)
└── weights/            base weights downloaded automatically, plus checkpoints you keep (git-ignored)
```

The bottom five folders are ignored by git on purpose — datasets and weights are far too big for a
repository. Share them over a drive or storage service instead.
