## Installation
create conda environment
```
conda create -n ngte python=3.9
conda activate ngte
```

To run MuJoCo simulation, a license is required. We used MuJoCo200.
https://www.roboti.us/download.html

## Usage
### Training and Evaluation
./scripts/{ENV}.sh {GPU} {SEED} {SAVE_DIR}
```
./scripts/Reacher.sh 0 0 exp
./scripts/AntMazeSmall.sh 0 0 exp
./scripts/AntMaze.sh 0 0 exp
./scripts/AntMazeBottleneck.sh 0 0 exp
./scripts/AntMazeComplex.sh 0 0 exp
```

## Troubleshooting

protobuf error

```
pip install --upgrade protobuf==3.20.0
```

gym EntryPoint error

```
pip uninstall gym
pip install gym==0.22.0
```

Cython error
```
pip install "Cython<3.0"
```


Our code sourced and modified from official implementation of [DHRL](https://github.com/jayLEE0301/dhrl_official/tree/main/DHRL) Algorithm.
