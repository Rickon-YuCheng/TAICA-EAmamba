# TAICA-EAmamba

本專案為 **EAMamba (Efficient All-Around Vision State Space Model)** 的實作，專注於圖像修復與超解析度重建任務（Image Super-Resolution）。本程式碼基於 Mamba 狀態空間模型架構，旨在提供高效且高品質的圖像重建能力。

## 🛠 環境需求

- Linux 系統
- Python 3.10
- CUDA 12.x
- PyTorch 2.1

## 🚀 快速開始

#### 1. 安裝 Mamba 與相依套件

由於 Mamba 對環境版本要求較高，請依序執行以下指令進行安裝（建議在虛擬環境中執行）：


#### 2. 下載預編譯的 Mamba SSM wheel 檔 (適用於 CUDA 12, Torch 2.1, Python 3.10)
```
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```

#### 3. 安裝 wheel 檔
```pip install mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl```

#### 4. 安裝其他專案依賴
```pip install -r requirements.txt```

## 💻 使用方法
本專案使用 PYTHONPATH=. 來確保模組路徑正確，請參照以下指令執行。

### 訓練 (Training)
執行訓練腳本，需指定配置檔 (config) 與實驗名稱：
```
PYTHONPATH=. python3 src/EAMamba/train.py --config src/EAMamba/configs/realsrx2-eamamba.yaml --name result
```

### 測試 (Testing)
使用訓練好的模型權重進行測試（例如 RealSRx2 資料集）：
```
PYTHONPATH=. python3 src/EAMamba/test.py --model save/result3/current_iter-best.pth --dataset RealSRx2
```

### 視覺化 (Visualization)
使用 TensorBoard 查看訓練過程與指標：
```
echo "nameserver 8.8.8.8" > /etc/resolv.conf
tensorboard --logdir /workspace/save/result21/tensorboard --port 6006 --bind_all
```

## 📁 專案結構
```
.
├── src/
│   └── EAMamba/
│       ├── configs/     # 模型與訓練參數設定
│       ├── train.py     # 訓練腳本
│       └── test.py      # 測試腳本
├── requirements.txt     # Python 依賴列表
└── README.md
```