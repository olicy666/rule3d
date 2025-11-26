# Raven3D 三宫格点云数据集生成器

本项目实现了 `program.md` 描述的三步点云生成系统。支持四种基础几何体、40 条可插拔规则、按难度概率采样，输出 A/B/C 三张点云及 meta 信息。

## 快速开始
环境：Python 3.10+，依赖仅 `numpy`。
```bash
pip install numpy
python main.py --output output --num-samples 3 --points 4096 --seed 0
```
生成目录示例：
```
output/sample_000000/
    A.ply
    B.ply
    C.ply
    meta.json
```

## 常用参数
- `--output`：输出根目录，默认 `output`
- `--num-samples`：生成多少道题目（多少个样本目录）
- `--points`：每个点云的点数，默认 4096
- `--seed`：固定随机种子
- `--simple-prob` / `--medium-prob` / `--complex-prob`：三种难度的采样概率，默认 0.7 / 0.2 / 0.1

示例：只生成复杂题目 5 道
```bash
python main.py --num-samples 5 --simple-prob 0 --medium-prob 0 --complex-prob 1
```

## 规则体系（40 条）
- Simple（14）：S01~S14（尺度/单轴/统一缩放、固定轴旋转、三轴旋转、平移、二段平移、形状替换、固定形状比例变化、密度感变化、竖向递增、重心补偿缩放、绕固定点旋转、恒等）
- Medium（14）：M01~M14（分离→接触→相交、相交深度增长、包含、相对位置、重心偏移、平行/垂直、轴角度变化、镜像对称、距离线性、队形移动、附着层级、比例耦合、面对齐→边对齐→点对齐、形状家族）
- Complex（12）：C01~C12（尺度+平移联动、旋转+尺度联动、多步序列、布尔序列、穿孔拓扑、隧道拓扑、截面拓扑、对称+缩放、角色互换、相交角度渐增、接触面积递增、多物体构型）

## 目录结构与扩展
- `main.py`：CLI 入口
- `raven3d/geometry.py`：几何原语与表面采样
- `raven3d/scene.py`：场景点云合成
- `raven3d/rules/`：40 条规则，按难度拆分；`base.py` 为基类
- `raven3d/factory.py`：默认规则注册
- `raven3d/dataset.py`：数据集生成逻辑

扩展新规则：在 `raven3d/rules/` 相应难度文件中添加 Rule 子类，并在 `factory.py` 中注册即可。
