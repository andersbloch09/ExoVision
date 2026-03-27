# ExoVision

# On Jetson: 
# Install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh

# Create Python 3.10 environment
conda create -n yolonet python=3.10
conda activate yolonet

### Build environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### Export variables:
#### Server side:
```bash
export ROLE=receiver
```
#### Sender side:
```bash
export ROLE=sender
```

### Run scripts:
```bash
python main.p
```

## .env format:
# Role of this machine: 'sender' (Spain) or 'receiver' (Denmark)
ROLE=sender

# For sender only: Tailscale hostname of the receiver machine (MagicDNS)
# Example: andersarch
DENMARK_HOST=andersarch

