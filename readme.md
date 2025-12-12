```
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```
```
pip install mamba_ssm-2.2.4+cu12torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```
```
echo "nameserver 8.8.8.8" > /etc/resolv.conf
tensorboard --logdir /workspace/save/result21/tensorboard --port 6006 --bind_all
```

```
PYTHONPATH=. python3 src/EAMamba/train.py --config src/EAMamba/configs/realsrx2-eamamba.yaml --name result
```