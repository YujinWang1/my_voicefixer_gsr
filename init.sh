# ��װϵͳ����
sudo apt-get update
sudo apt-get install -y libsox-fmt-all libsox-dev sox

# ��װ pip ����
pip3 install -r requirements.txt

# ��װ PyTorch�����а汾��ͻ�����ʵ�ʻ���������
pip3 install torch==1.8.1 torchaudio==0.8.1 pytorch_lightning==1.5.0

# ��װ GitPython
pip3 install GitPython

# ��װ speechmetrics��ע�� numpy �汾��
pip3 install numpy==1.23.4
pip3 install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics

python3 -m pip install git+https://github.com/facebookresearch/WavAugment.git

