# Rule3D：基于规则的多几何体点云推理数据集生成器

本项目实现 `program.md` 中的“数学原型”版生成器：每题输出 6 个点云（1/2 为参考帧 A/B，3/4/5/6 为四个候选，其中仅 1 个正确）和 `meta.json`，同时根目录提供所有题目的 `meta.json` 列表。规则总数 25（移除 R4 拓扑类），每帧场景由 2~3 个几何体组成。

## 快速开始
环境：Python 3.10+，依赖仅 `numpy`。
```bash
pip install numpy
python main.py --output output --num-samples 3 --points 4096 --seed 0
## 模式与参数
- `--mode` 可选：
  - 主集合：`main`
  - 大类：`r1-only`, `r2-only`, `r3-only`
  - 消融：`all-minus-r1`, `all-minus-r2`, `all-minus-r3`
  - （已删除 `r4-only` 与 `all-minus-r4`）
- `--rules` 自定义规则列表（逗号分隔，如 `R1-1,R2-3,R3-2`，会覆盖 `--mode`）
  - 仅从给定规则编号中采样题目，编号不区分大小写，非法编号会直接报错提示可选列表。
  - 示例：`python main.py --num-samples 5 --rules R1-1,R2-3,R3-2 --points 4096`
- 规则采样：当前对可用规则等概率随机采样（`--simple-prob / --medium-prob / --complex-prob` 已弃用）。
- 其他：`--output`, `--num-samples`, `--points`, `--seed`。
示例：`python main.py --num-samples 10 --mode r2-only --points 4096`
- 默认每帧点数 `--points=4096`，写入 PLY 时固定颜色：1=红，2=绿，3=蓝，4=紫，5=白，6=橙（同一文件内所有点同色）。
```
生成目录示例：
```
output/sample_000000/
    1.ply  # A
    2.ply  # B
    3.ply  # 候选 A
    4.ply  # 候选 B
    5.ply  # 候选 C
    6.ply  # 候选 D
    meta.json  # 含文件路径、正确选项、rule_meta
output/meta.json  # 所有题目的 meta 列表
```
点云文件固定颜色（每个文件内所有点相同）：1=深海蓝(31,119,180)，2=鲜亮橙(255,127,14)，3=森林绿(44,160,44)，4=砖红(214,39,40)，5=柔和紫(148,103,189)，6=可可棕(140,86,75)。

## 数学对象定义
- 场景帧：$$X_t = \{O_{t,1}, \dots, O_{t,M_t}\},\quad M_t \in \{2,3\}$$
- 物体基础属性：$$s \in \{\texttt{cube},\texttt{sphere},\texttt{cylinder},\texttt{cone}\},\quad r=(r_x,r_y,r_z),\quad p\in\mathbb{R}^3,\quad R\in SO(3),\quad d\in\mathbb{R}_+$$
- 采样：按权重 $$w_i = d_i\cdot\text{vol}(r_i)$$ 分配总点数到各物体，合并后写入同一 ply；meta 中保留对象级参数。

## 派生函数（仅依赖 (s,r,p,R,d)）
- 单体：$$\text{size}(O)=r_x r_y r_z,\quad \text{ar}(O)=(\tfrac{r_x}{r_y},\tfrac{r_y}{r_z}),\quad \text{axis}_k(O)=R e_k,\quad \text{den}(O)=d$$
- 成对：$$\text{dist}(i,j)=\|p_i-p_j\|_2,\quad \text{dir}(i,j)=\tfrac{p_j-p_i}{\|p_j-p_i\|_2+\epsilon},\quad \text{ang}(i,j)=\arccos(\langle\text{axis}_1(i),\text{axis}_1(j)\rangle)$$
  $$\text{touch}(i,j)=\mathbf{1}[\text{dist}\le \rho_i+\rho_j+\tau],\quad \text{contain}(i,j)=\mathbf{1}[\text{AABB}(i)\supseteq\text{AABB}(j)]$$
  $$\text{contain\_ratio}(i,j)=\min_k \frac{\min(a^{max}_k-b^{max}_k,\,b^{min}_k-a^{min}_k)}{(a^{max}_k-a^{min}_k)/2-(b^{max}_k-b^{min}_k)/2}$$
- 多体：$$\text{cent}(S)=\tfrac{1}{|S|}\sum_{i\in S}p_i,\quad \text{area}(1,2,3)=\tfrac{1}{2}\|(p_2-p_1)\times(p_3-p_1)\|$$
  $$\text{ord}_x(S)=\text{argsort}([p_{i,x}]),\quad \text{sym}(S)=\mathbf{1}[\forall i\,\exists j:\|p_j-\pi_n(p_i)\|\le\delta]$$

## 模式变换（可枚举）
- 等差：$$v_{t+1}-v_t=\Delta\ \Rightarrow\ v_3=2v_2-v_1$$
- 等比：$$v_{t+1}=k\odot v_t\ \Rightarrow\ v_3=k\odot v_2$$
- 离散序列：固定符号序列 (ABA/ABC/等)。
- 刚体/仿射：$$p_{t+1}=Q p_t + t,\quad R_{t+1}=Q R_t$$
- 联动：多变量守恒/耦合（质心恒定、距离和恒定等）。

## 规则清单（25 条：9 Simple + 10 Medium + 6 Complex）
所有规则 meta 包含 `rule_id, rule_group(R1/R2/R3), difficulty, K_R, involved_indices, base_attrs_used, derived_funcs, pattern_type, pattern_params, v1/v2/v3`。

### R1 物体属性推理（R1-1–R1-11）
- **R1-1 等差统一缩放**：$$\text{size}_2-\text{size}_1=\Delta,\ \text{size}_3=2\text{size}_2-\text{size}_1$$
- **R1-2 各向异性等比拉伸**：$$r_{t+1}=r_t\odot s,\ s_{axis}=k,\ s_{\text{others}}=\tfrac{1}{\sqrt{k}}\ \Rightarrow\ \text{vol 保持}$$
- **R1-3 固定轴旋转**：$$R_{t+1}=R_t\cdot\text{Rot}(\hat{u},\theta)$$
- **R1-4 旋转离散循环**：$$(R_1,R_2,R_3)=(Q_0,Q_{90},Q_{180})$$
- **R1-5 固定向量平移**：$$p_{t+1}=p_t+\Delta p$$
- **R1-6 密度等差**：$$d_2-d_1=\Delta,\ d_3=2d_2-d_1$$
- **R1-7 形状变化继承**：A→B 哪些位置形状改变，B→C 在相同位置继续改变
- **R1-8 尺度-位置联动**：$$\text{cent}(S_t)=\text{const},\ r_{t+1}=k r_t,\ p_{t+1}=p_t+\delta p$$
- **R1-9 恒等**：$$(s,r,p,R,d)_1=(s,r,p,R,d)_2=(s,r,p,R,d)_3$$
- **R1-10 双对象守恒**：$$\text{size}(i)_t+\text{size}(j)_t=C,\ \text{size}(i)\ \text{递增},\ \text{size}(j)\ \text{递减}$$
- **R1-11 复合属性**：$$r_{t+1}=k r_t,\ R_{t+1}=R_t\text{Rot}(\hat{u},\theta)$$

### R2 成对空间关系推理（R2-1–R2-8）
- **R2-1 成对距离等比**：$$\text{dist}_2=k\text{dist}_1,\ \text{dist}_3=k\text{dist}_2$$
- **R2-2 方向保持**：$$\text{dir}_1=\text{dir}_2=\text{dir}_3,\ \text{dist}\ \text{线性/等比}$$
- **R2-3 方向旋转等差角**：$$\angle(\text{dir}_t,\text{dir}_{t+1})=\theta,\ \text{dist}\ \text{恒定}$$
- **R2-4 包含比例等差**：$$\rho_{t+1}-\rho_t=\Delta,\ \rho_t\in(0,1)$$
- **R2-5 夹角等差**：$$\text{ang}_3=2\text{ang}_2-\text{ang}_1$$
- **R2-6 距离差分守恒**：$$\text{dist}(i,j)-\text{dist}(k,l)=C,\ \forall t$$
- **R2-7 刚体一致变换**：集合施加相同刚体，任意成对距离保持不变。
- **R2-8 相对姿态保持**：共同旋转 $$R^{(i)}_{t+1}=Q R^{(i)}_t$$，夹角恒定。

### R3 多物体构型推理（R3-1–R3-6）
- **R3-1 三对象面积等差**：$$\text{area}_3=2\text{area}_2-\text{area}_1$$
- **R3-2 排序模式循环**：$$\text{ord}_x(S_1),\text{ord}_x(S_2),\text{ord}_x(S_3)\ \text{按固定置换}$$
- **R3-3 距离集合等比**：$$v_{t+1}=k v_t,\ v_t=[\text{dist}(1,2),\text{dist}(1,3),\text{dist}(2,3)]$$
- **R3-4 对称+刚体**：$\text{sym}(S_2)=1,\ X_3=Q X_2+t$
- **R3-5 组间质心距离等差**：$$u_t=\|\text{cent}(S_a)-\text{cent}(S_b)\|,\ u_3=2u_2-u_1$$
- **R3-6 面积-边长守恒**：$$\text{area}(1,2,3)\cdot \text{dist}(1,2)=C$$



## 目录结构
- `main.py`：CLI 入口与模式选择
- `raven3d/scene.py`：`ObjectState`/`Scene` 多物体采样
- `raven3d/rules/`：25 条规则（按 simple/medium/complex 划分）
- `raven3d/dataset.py`：样本生成与 meta 写出
- `program.md`：实施要求

扩展新规则时可复用 `rules/utils.py` 中的派生函数与 meta 构建工具。
