# yolo_trainer

用 Docker 打包好的 [Ultralytics YOLO](https://docs.ultralytics.com/) 訓練環境。
clone 下來跑一個腳本就能開始訓練，不用在自己電腦上裝 CUDA、cuDNN、PyTorch。

English: [README.md](README.md)

- 支援一般偵測與 **OBB**（旋轉框）的訓練、推論、匯出
- 所有設定都在 `configs/*.yaml`，不用改程式碼
- 訓練結果放在 `runs/`，檔案擁有者是你自己（不是 root）

## 環境需求

| | |
|---|---|
| GPU | NVIDIA 顯示卡。**只有 CPU 的機器沒辦法訓練。** |
| 驅動 | 已安裝 NVIDIA driver（`nvidia-smi` 跑得起來） |
| Docker | [Docker Engine](https://docs.docker.com/engine/install/) 與 `docker compose` plugin |
| Runtime | [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，容器才看得到 GPU |
| 硬碟 | 映像檔約 20 GB |

## 快速開始

```bash
git clone <this-repo> && cd yolo_trainer
bash setup.sh                       # 檢查環境、build 映像檔、啟動容器
docker exec -it yolo_trainer bash   # 進到容器裡，位置是 /usr/src
```

`setup.sh` 會確認驅動與 container runtime、用你的 uid 寫出 `docker/.env`（這樣容器產生的檔案才不會變成 root）、
開 X11 權限，然後 build 並啟動容器。可以重複執行；`bash setup.sh --no-build` 會跳過重新 build。

以下指令都是在**容器裡面**執行。

## 1. 準備資料集

把資料集資料夾放進 `datasets/`。YOLO 格式（Roboflow 匯出的格式）長這樣：

```
datasets/my_dataset/
├── data.yaml
├── train/{images,labels}
├── valid/{images,labels}
└── test/{images,labels}      # 可有可無
```

`data.yaml` 至少要有三個欄位，路徑是相對於 `data.yaml` 所在的資料夾：

```yaml
train: train/images
val: valid/images
test: test/images

names:
  0: player
  1: ball
```

OBB 的標註每行是 8 個數字（四個角點）而不是 4 個；Roboflow 匯出時選
**YOLOv8 Oriented Bounding Boxes**，訓練時要用 `-obb` 結尾的模型。

## 2. 訓練

複製一份 config、改內容、執行。不需要動 `train.py`。

```bash
cp configs/example_detect.yaml configs/my_experiment.yaml
# 把 data: 改成你的資料集、name: 改成這次實驗的名字
python train.py --config configs/my_experiment.yaml
```

config 分成三部分：

```yaml
model: yolo11n.pt                      # 權重 (.pt) 或架構檔 (models/*.yaml)
data: datasets/my_dataset/data.yaml    # 相對於 repo 根目錄
args:                                  # Ultralytics 的訓練參數都可以寫在這
  epochs: 300
  imgsz: 640
  batch: 8
  name: my_experiment
```

內建範例：[`configs/example_detect.yaml`](configs/example_detect.yaml)（一般偵測）、
[`configs/example_obb.yaml`](configs/example_obb.yaml)（OBB 微調，附一組手調過的 SGD 設定）。

最常調的參數：

| 參數 | 作用 |
|---|---|
| `epochs` | 總共訓練幾輪 |
| `imgsz` | 訓練解析度。小物件用 1280 會好很多，但時間與記憶體約是 640 的 4 倍 |
| `batch` | 一次幾張圖。**顯示記憶體不夠時先降這個** |
| `device` | `0`、`0,1` 或 `cpu` |
| `patience` | 連續 N 輪沒進步就提早停止 |
| `optimizer`、`lr0`、`cos_lr` | 學習率設定；剛開始用預設的 `auto` 就好 |
| `degrees`、`scale`、`hsv_*`、`mosaic` | 資料增強——要對應你實際拍到的畫面變化 |
| `close_mosaic` | 最後 N 輪關掉 mosaic，讓模型收在正常畫面上 |

完整清單：<https://docs.ultralytics.com/modes/train/#train-settings>

不改 config 也可以直接在指令列覆寫：

```bash
python train.py --config configs/my_experiment.yaml --set epochs=1 --name smoke_test   # 先跑一輪確認沒問題
python train.py --config configs/my_experiment.yaml --set lr0=0.001 --set batch=4
python train.py --config configs/my_experiment.yaml --resume                           # 接續 runs/<name> 的訓練
python train.py --config configs/my_experiment.yaml --resume runs/other_run/weights/last.pt
```

訓練結果（權重、曲線、混淆矩陣、範例批次圖）在 `runs/<name>/`。
先看 `runs/<name>/results.png` 和 `confusion_matrix.png`，訓練好的模型是 `runs/<name>/weights/best.pt`。

要從自己之前的權重繼續微調，把 `model:` 指過去就好：
`model: runs/my_previous_run/weights/best.pt`。

## 3. 推論

```bash
# 存標註後的圖到 runs/predict/（不需要螢幕，SSH 也能用）
python predict.py --weights runs/my_run/weights/best.pt --source imgs/my_photos

# 或一張一張開視窗看（需要 X11；按任意鍵下一張、q 離開）
python predict.py --weights runs/my_run/weights/best.pt --source imgs/my_photos --show
```

常用參數：`--conf 0.6`（信心門檻）、`--imgsz`、`--line-width`、`--hide-conf`、`--device cpu`。
`--source` 可以是單張圖、資料夾，或攝影機編號（例如 `0`）。

## 4. 匯出

```bash
python export.py --weights runs/my_run/weights/best.pt                       # ONNX，imgsz 640
python export.py --weights runs/my_run/weights/best.pt --format engine --imgsz 960 --half
```

匯出的檔案會放在 `.pt` 旁邊。支援格式：<https://docs.ultralytics.com/modes/export/#export-formats>

## 疑難排解

| 狀況 | 解法 |
|---|---|
| `setup.sh` 說 *Docker cannot see the NVIDIA runtime* | 裝 nvidia-container-toolkit，然後 `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| Docker socket `permission denied` | `sudo usermod -aG docker $USER`，登出再登入 |
| 容器裡 `torch.cuda.is_available()` 是 False | 重跑 `bash setup.sh`；用 `docker exec -it yolo_trainer nvidia-smi` 確認 |
| `CUDA out of memory` | 先降 `batch`（8 → 4 → 2），再降 `imgsz`（1280 → 640），也可以把 `workers` 調到 4 |
| `--show` 沒反應／`cannot open display` | 在**主機**上執行 `xhost +local:docker`（重開機會失效），或不要用 `--show`，直接看存下來的圖 |
| 出現 `QFontDatabase: Cannot find font directory .../cv2/qt/fonts` | 無害，是 OpenCV 的 Qt 版本自己找字型，不影響視窗顯示 |
| `runs/` 裡的檔案變成 root 的 | `docker/.env` 的 uid 不對，重跑 `bash setup.sh`，再 `cd docker && docker compose up -d --force-recreate` |
| `Dataset config not found` | 錯誤訊息會列出目前找得到的資料集；確認 config 裡的路徑是相對於 repo 根目錄 |
| `yolo11n.pt` 下載失敗 | 從 [Ultralytics releases](https://github.com/ultralytics/assets/releases) 手動下載，放到 `weights/` |
| 訓練很慢 | 訓練時看 `nvidia-smi`，如果 GPU 使用率很低代表卡在讀資料：調高 `workers` 或縮小圖片 |
| 改了 `train.py` 卻沒作用 | 你改到別份檔案了——repo 是掛載到容器的 `/usr/src` |

## 自訂架構（進階）

`models/` 是本地資料夾（不進 git），放你自己改的模型定義：把修改過的架構 YAML 丟進去，
config 指向它就會從頭訓練。

```yaml
model: models/my_architecture.yaml
```

建議從官方的 Ultralytics YAML（例如 `yolo11.yaml`、`yolo11-obb.yaml`）複製一份再改。

## 目錄結構

```
yolo_trainer/
├── setup.sh            一鍵環境設定 + 啟動容器
├── configs/            訓練設定檔——要改的是這裡
├── train.py            訓練進入點
├── predict.py          對圖片／資料夾／攝影機做推論
├── export.py           ONNX / TensorRT 匯出
├── common.py           共用的路徑處理與錯誤訊息
├── docker/             Dockerfile、compose.yaml、.env（自動產生）
├── models/             自己改的架構 YAML（不進 git）
├── datasets/           你的資料集      （不進 git）
├── imgs/               要推論的圖片    （不進 git）
├── runs/               訓練結果        （不進 git）
└── weights/            自動下載的基礎權重與你保留的權重（不進 git）
```

下面這五個資料夾刻意不進 git——資料集和權重對 repo 來說太大了，請用雲端硬碟或檔案伺服器分享。
