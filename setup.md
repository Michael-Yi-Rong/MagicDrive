# MADIC-DRIVE
git clone https://github.com/cure-lab/MagicDrive.git

# environment
conda create --name mdrive python=3.8
conda activate mdrive
pip install https://download.pytorch.org/whl/cu113/torch-1.10.2%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchvision-0.11.3%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/mmcv_full-1.4.5-cp38-cp38-manylinux1_x86_64.whl

pip install -r requirements/dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements/dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

cd third_party/diffusers
pip install .

cd third_party
git clone https://github.com/mit-han-lab/bevfusion.git
cd third_party/bevfusion
git checkout db75150

cd third_party/bevfusion
python setup.py develop

pip install fastapi==0.112.2 --no-deps  #new version has bug

# configuration
accelerate config

# stable-diffusion-v1-5 download
cd /SSD_DISK/users/rongyi/projects/MagicDrive-main/pretrained
git clone https://gitee.com/hf-models/stable-diffusion-v1-5.git  # mirror
git lfs install
git lfs pull  # more than 40 gb

# SDv1.5mv-rawbox_2023-09-07_18-39_224x400 download
[SDv1.5mv-rawbox_2023-09-07_18-39_224x400](https://mycuhk-my.sharepoint.com/personal/1155157018_link_cuhk_edu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1155157018%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FMagicDrive%2Fgithub%5Frelease%2FSDv1%2E5mv%2Drawbox%5F2023%2D09%2D07%5F18%2D39%5F224x400%2Ezip&parent=%2Fpersonal%2F1155157018%5Flink%5Fcuhk%5Fedu%5Fhk%2FDocuments%2FMagicDrive%2Fgithub%5Frelease&ga=1)  # open in browser, put in /SSD_DISK/users/rongyi/projects/MagicDrive-main/pretrained

# magicdrive-log mkdir
mkdir -p /SSD_DISK/users/rongyi/projects/MagicDrive-main/magicdrive-log

# executation
cd /SSD_DISK/users/rongyi/projects/MagicDrive-main
python demo/interactive_gui.py

# inspection
lsof -i :7860
kill -9 208040  # PID changes

# local listening
ssh -L 7860:localhost:7860 rongyi@10.28.5.215 -p 15654