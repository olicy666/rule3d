# 三宫格点云数据集生成器

本项目实现了 `program.md` 描述的三步点云生成系统。支持四种基础几何体、40 条可插拔规则、按难度概率采样，输出六张点云（A/B 参考 + C/D/E/F 选项，其中仅一张正确）及 meta 信息。

## 快速开始
环境：Python 3.10+，依赖仅 `numpy`。
```bash
pip install numpy
python main.py --output output --num-samples 3 --points 4096 --seed 0
```
生成目录示例：
```
output/sample_000000/
    1.ply
    2.ply
    3.ply
    4.ply
    5.ply
    6.ply
    meta.json  # 当前题目的路径与答案（同下方 meta 列表格式）
output/meta.json  # 所有题目组成的列表
                 # 每个 meta 中包含 rule_id 以便回溯规则
```

## 常用参数
- `--output`：输出根目录，默认 `output`
- `--num-samples`：生成多少道题目（多少个样本目录）
- `--points`：每个点云的点数，默认 4096
- `--seed`：固定随机种子
- `--simple-prob` / `--medium-prob` / `--complex-prob`：三种难度的采样概率，默认 0.7 / 0.2 / 0.1
- `--mode`：规则采样模式，默认 `main`。可选 `r1-only`/`r2-only`/`r3-only`/`r4-only` 四大类，`r1-1`~`r3-2` 子类，以及消融的 `all-minus-rX`（详见 `program.md` 的四种模式说明）。

说明：A/B/C/D 选项的正确答案位置是均衡分布的（例如 100 题时四个选项各 25 次）。文件名映射：参考 1/2，对应候选 3/4/5/6，A/B/C/D 分别指向 3/4/5/6。

## 运行指令示例（覆盖四类模式）
- 主模式（40 条规则随机，答案均衡）：`python main.py --output output --num-samples 100 --mode main`

- 单大类模式：
  - R1 物体属性：`python main.py --num-samples 50 --mode r1-only`
  - R2 成对空间关系：`python main.py --num-samples 50 --mode r2-only`
  - R3 多物体构型：`python main.py --num-samples 50 --mode r3-only`
  - R4 结构与拓扑：`python main.py --num-samples 50 --mode r4-only`

- 大类内子类模式：
  - R1-1 尺度/比例：`python main.py --num-samples 20 --mode r1-1`
  - R1-2 位姿（旋转/平移）：`python main.py --num-samples 20 --mode r1-2`
  - R1-3 外观/密度/恒等：`python main.py --num-samples 20 --mode r1-3`
  - R1-4 复合联动序列：`python main.py --num-samples 20 --mode r1-4`
  - R2-1 拓扑强度：`python main.py --num-samples 20 --mode r2-1`
  - R2-2 度量/方向：`python main.py --num-samples 20 --mode r2-2`
  - R3-1 全局对称与身份：`python main.py --num-samples 20 --mode r3-1`
  - R3-2 群体组织高阶构型：`python main.py --num-samples 20 --mode r3-2`

- 消融模式（去掉一个大类）：
  - 去掉 R1：`python main.py --num-samples 30 --mode all-minus-r1`
  - 去掉 R2：`python main.py --num-samples 30 --mode all-minus-r2`
  - 去掉 R3：`python main.py --num-samples 30 --mode all-minus-r3`
  - 去掉 R4：`python main.py --num-samples 30 --mode all-minus-r4`

可选参数：`--points` 控制点数，`--seed` 固定随机性，`--simple-prob/--medium-prob/--complex-prob` 调整难度比例（在当前模式可用规则内按概率抽样）。

示例：只生成复杂题目 5 道
```bash
python main.py --num-samples 5 --simple-prob 0 --medium-prob 0 --complex-prob 1 --mode r4-only
```

## 规则体系（40 条,具体规则见rule.md文档）
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
