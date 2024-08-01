pip install mujoco_py==2.0.2.8
pip install numpy==1.21.4
pip install gym==0.21.0

pip install networkx
pip install opencv-python
pip install matplotlib

mamba create -n ng39 python=3.9 pytorch pytorch-cuda=11.8 cudnn networkx matplotlib opencv tensorboard gym=0.21 -c conda-forge -c pytorch -c nvidia
mamba install moviepy