# ExoVision
### Build environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
pip3.10 install -r requirements.txt
```
### Export variables:
Create a .env file in the outer directory and fill the information
#### Server side:
```bash
export ROLE=receiver
```
#### Sender side:
```bash
export ROLE=sender
```
#### HOST:
```bash
export DENMARK_HOST=andersarch
```


### Run scripts:
```bash
python main.p
```

# Python 3.10 from source for Jetson before starting: 

## 1. Install required build tools
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget \
libbz2-dev

## 2. Download Python 3.10 source
cd /tmp
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
tar -xf Python-3.10.12.tgz
cd Python-3.10.12

## 3. Configure and compile (optimized)
./configure --enable-optimizations --with-ensurepip=install
make -j4    # replace 4 with number of CPU cores
sudo make altinstall   # altinstall avoids overwriting system python3

## 4. Verify installation
python3.10 --version
pip3.10 --version

## 5. Create virtual environment for your project
cd ~/ExoVision
python3.10 -m venv venv
source venv/bin/activate

## 6. Upgrade pip inside the venv
pip install --upgrade pip

## 7. Install your project dependencies
pip install -r requirements.txt


## Model Weight Update Strategies

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **Atomic Swap** | Simple, fast | Doubles memory briefly | Jetson (small models) |
| **Versioning** | Robust, checksum verify | More code | Production systems |
| **Read-Write Lock** | Fine-grained control | Complex, can block reads | High-concurrency systems |

**Recommendation for Jetson:** Use Atomic Swap - loads new model completely before acquiring lock, then swaps in microseconds. Inference never pauses.