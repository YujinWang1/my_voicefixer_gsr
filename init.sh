# 安装系统依赖
sudo apt-get update
sudo apt-get install -y libsox-fmt-all libsox-dev sox

# 安装 pip 依赖
pip3 install -r requirements.txt

# 安装 PyTorch（如有版本冲突请根据实际环境调整）
pip3 install torch==1.8.1 torchaudio==0.8.1 pytorch_lightning==1.5.0

# 安装 GitPython
pip3 install GitPython

# 安装 speechmetrics（注意 numpy 版本）
pip3 install numpy==1.23.4
pip3 install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics

python3 -m pip install git+https://github.com/facebookresearch/WavAugment.git

