# XuanCE框架测试工具

本目录包含用于测试XuanCE框架在不同深度学习后端下的性能和功能的工具。

## 测试文件说明

- `test_all_7.py`: 测试多智能体强化学习算法在不同框架下的性能
- `test_oneflow_marl.py`: 专门测试OneFlow框架下的多智能体强化学习算法
- `test_oneflow_sarl.py`: 专门测试OneFlow框架下的单智能体强化学习算法
- `run_oneflow_tests.py`: 运行所有OneFlow框架测试的脚本
- `compare_frameworks.py`: 比较不同框架性能的测试脚本

## 使用方法

### 运行OneFlow测试

运行所有OneFlow测试：

```bash
python run_oneflow_tests.py
```

只运行单智能体测试：

```bash
python run_oneflow_tests.py sarl
```

只运行多智能体测试：

```bash
python run_oneflow_tests.py marl
```

### 比较不同框架性能

比较DQN算法在不同框架下的性能：

```bash
python compare_frameworks.py --method dqn --env classic_control --env_id CartPole-v1 --frameworks torch oneflow --n_steps 10000 --n_runs 3
```

比较PPO算法在不同框架下的性能：

```bash
python compare_frameworks.py --method ppoclip --env classic_control --env_id CartPole-v1 --frameworks torch oneflow --n_steps 10000 --n_runs 3
```

比较MADDPG算法在不同框架下的性能：

```bash
python compare_frameworks.py --method maddpg --env mpe --env_id simple_spread_v3 --frameworks torch oneflow --n_steps 10000 --n_runs 3
```

## 参数说明

`compare_frameworks.py` 支持以下参数：

- `--method`: 强化学习算法名称
- `--env`: 环境类型
- `--env_id`: 环境ID
- `--frameworks`: 要比较的框架列表
- `--n_steps`: 训练步数
- `--device`: 设备类型 ('cpu' 或 'cuda')
- `--n_runs`: 每个框架运行的次数
- `--save_path`: 结果保存路径

## 支持的算法

### 单智能体算法

- DQN, DDQN, DuelingDQN, C51, QRDQN, PER-DQN, NoisyDQN, DRQN
- PG, A2C, PPO-Clip, PPO-KL, PPG, DDPG, TD3, SAC

### 多智能体算法

- IQL, VDN, QMIX, MADDPG, QTRAN, COMA, MAPPO

## 注意事项

1. 确保已安装所有需要的依赖包
2. 对于GPU测试，确保设备支持CUDA并已正确配置
3. 测试结果会保存在 `results` 目录下 