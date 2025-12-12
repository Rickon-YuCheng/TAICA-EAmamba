```
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.1cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
```
```
mamba_ssm-2.2.4+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```
```
echo "nameserver 8.8.8.8" > /etc/resolv.conf
tensorboard --logdir /workspace/save/result21/tensorboard --port 6006 --bind_all
```

```
PYTHONPATH=. python3 src/EAMamba/train.py --config src/EAMamba/configs/realsrx2-eamamba.yaml --name result
PYTHONPATH=. python3 src/EAMamba/test.py --model save/_realsrx2-eamamba/current_iter-best.pth --dataset RealSRx2
```