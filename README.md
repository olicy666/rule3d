# SPIRAL3D:Structured Perception to Intelligent Reasoning And Logic in 3D

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

顶层认知原型：
Level 1 (R1): 固有属性感知
- 认知视角： Intrinsic Attribute Perception（固有属性感知）。
  - 描述： 就像人类婴儿首先学会识别“红色”、“圆形”一样，模型首先必须具备高保真的特征解耦能力。它需要从杂乱的 3D 点云中准确提取出每个物体的独立物理属性（尺寸、密度、姿态），不受环境干扰。
- 图论视角： Node Embedding Refinement（节点嵌入精炼）。
  - 定义： 给定图  $$G=(V, E)$$，R1 考察模型近似一元函数  $$f(v_i)$$的能力。
  - 数学： 目标是最小化属性预测误差  $$\mathcal{L}_{R1} = \sum || \hat{y}_{attr} - f(v_i) ||$$
Insight：强调是从非结构化的 Raw Point Cloud 到结构化的 Graph Node 的特征提取过程，这本身就是 3D 视觉的一大难点。
Level 2 (R2): 显式拓扑推理
- 认知视角： Relational Reasoning（关系推理）。
  - 描述： 认知升级。模型从关注点扩展到关注线，必须理解物体间的空间拓扑（如包含、相邻）和度量关系（如距离、角度），建立场景的空间语境。
- 图论视角： Edge Message Passing（边消息传递）。
  - 定义： R2 考察模型近似二元函数  $$g(v_i, v_j)$$的能力。
  - 数学： 这是一个  $$O(N^2)$$的成对交互问题，模型必须准确编码边属性 $$e_{ij}$$。
Insight：强调 R2 使得模型不再是看一堆点，而是结合多个点判断点间的关系。
Level 3 (R3): 隐式结构推理
- 认知视角： Abstract Systemic Reasoning（抽象系统推理） 
  - 描述： 真正的智力飞跃。模型不仅要看清现有的点和线，还要想象出不存在的结构。比如看到三个点，脑子里要画出一个三角形并计算面积；看到一堆乱点，要能分出 Group A 和 Group B 并找到它们的重心。这是对潜在变量的推理。
- 图论视角： Subgraph / Hyper-edge Pattern Recognition（子图/超边模式识别）。
  - 定义： R3 考察模型在节点子集  $$V_{sub} \subset V$$ 上提取高阶特征  $$H(V_{sub})$$的能力。
  - 数学： 涉及超边的构建。比如面积是三个节点  $$(v_1, v_2, v_3)$$共同决定的属性；对称性是整个图 $$G$$ 的全局同构属性。
Insight：聚焦于如何高效处理高阶依赖，将题目包装为抽象认知
Level 4 (R4): 抽象逻辑演化——不再完全约束原图表的形式，会学会演化该图表
- 认知视角： Counterfactual & Planning（反事实与规划）。
  - 描述： 认知的最高阶——想象力，模型不再满足于静态观察，它开始在脑海中演化这个场景。它思考“如果……会怎样”（If... then...），这需要对物理规律和几何约束的深刻理解。
- 图论视角： Dynamic Graph Evolution（动态图演化）。
  - 定义： R4 考察模型预测图状态转移的能力： $$
G_{t+1} = \mathcal{T}(G_t, \text{Action})$$。
  - 实现（基于推理轨迹）： 推理轨迹就是图的演化步：
    1. Step 1: 识别当前子图结构（Constraint Identification）。
    2. Step 2: 施加扰动（Perturbation）。
    3. Step 3: 预测节点属性或边关系的更新（State Update）。
Insight：把静态的 3D 问答提升到了World Model的高度。

### R1 固有属性感知
- **R1-1 等差统一缩放**：$$\text{size}_2-\text{size}_1=\Delta,\ \text{size}_3=2\text{size}_2-\text{size}_1$$
- **R1-2 尺度轴置换循环**：$$(r_x,r_y,r_z)\to(r_y,r_z,r_x)\to(r_z,r_x,r_y)$$
- **R1-3 固定轴旋转**：$$R_{t+1}=R_t\cdot\text{Rot}(\hat{u},\theta)$$
- **R1-4 旋转离散循环**：$$(R_1,R_2,R_3)=(Q_0,Q_{90},Q_{180})$$
- **R1-5 固定向量平移**：$$p_{t+1}=p_t+\Delta p$$
- **R1-6 密度等差**：$$d_2-d_1=\Delta,\ d_3=2d_2-d_1$$
- **R1-7 形状变化继承**：A→B 哪些位置形状改变，B→C 在相同位置继续改变
- **R1-8 质心守恒缩放**：$$\text{cent}(S_t)=\text{const},\ r_{t+1}=k r_t,\ p_{t+1}=p_t+\delta p$$
- **R1-9 双对象守恒**：$$\text{size}(i)_t+\text{size}(j)_t=C,\ \text{size}(i)\ \text{递增},\ \text{size}(j)\ \text{递减}$$
- **R1-10 复合位姿缩放**：$$r_{t+1}=k r_t,\ R_{t+1}=R_t\text{Rot}(\hat{u},\theta)$$
- **R1-11 属性互换联动**：两物体形状/尺度交替互换（先形状后尺度或相反）
- **R1-12 主轴对齐位移**：移动方向与主轴一致
- **R1-13 距离驱动缩放**：远离锚点变大，靠近锚点变小
- **R1-14 镜像互补缩放**：镜像位置下尺寸一增一减
- **R1-15 距离尺寸倒数**：距离越近尺寸变化越剧烈
- **R1-16 距离密度倒数**：距离越近密度变化越剧烈
- **R1-17 几何体个数变化**：不同形状的数量按等差增加

### R2 显式拓扑推理
- **R2-1 成对距离等比**：$$\text{dist}_2=k\text{dist}_1,\ \text{dist}_3=k\text{dist}_2$$
- **R2-2 各向异性等比拉伸**：$$r_{t+1}=r_t\odot s,\ s_{axis}=k,\ s_{\text{others}}=\tfrac{1}{\sqrt{k}}\ \Rightarrow\ \text{vol 保持}$$
- **R2-3 方向旋转等差**：$$\angle(\text{dir}_t,\text{dir}_{t+1})=\theta,\ \text{dist}\ \text{恒定}$$
- **R2-4 包含比例等差**：$$\rho_{t+1}-\rho_t=\Delta,\ \rho_t\in(0,1)$$
- **R2-5 夹角等差**：$$\text{ang}_3=2\text{ang}_2-\text{ang}_1$$
- **R2-6 距离差分守恒**：$$\text{dist}(i,j)-\text{dist}(k,l)=C,\ \forall t$$
- **R2-7 刚体一致变换**：集合施加相同刚体，任意成对距离保持不变。
- **R2-8 相对姿态保持**：共同旋转 $$R^{(i)}_{t+1}=Q R^{(i)}_t$$，夹角恒定。
- **R2-9 加速旋转**：旋转幅度在第二帧与第三帧之间继续增大。
- **R2-10 几何融合**：多个对象以等差步伐向中心融合。
- **R2-11 行星公转**：多个对象绕中心球体旋转，角速度各不相同。
- **R2-12 易位姿态转换**：不同形状的姿态变化按形状延续。
- **R2-13 易位尺寸转换**：不同形状的尺寸变化按形状延续。
- **R2-14 易位密度转换**：不同形状的密度变化按形状延续。

### R3 隐式结构推理
- **R3-1 三对象面积等差**：$$\text{area}_3=2\text{area}_2-\text{area}_1$$（允许 $\Delta=0$）
- **R3-2 排序模式循环**：$$\text{ord}_x(S_1),\text{ord}_x(S_2),\text{ord}_x(S_3)\ \text{按固定置换}$$
- **R3-3 距离集合等比**：左右独立等比缩放（$k_L,k_R$ 允许为 1）
- **R3-4 刚体对称变换**：$\text{sym}(S_2)=1,\ X_3=Q X_2+t$
- **R3-5 质心距离等差**：$$u_t=\|\text{cent}(S_a)-\text{cent}(S_b)\|,\ u_3=2u_2-u_1$$
- **R3-6 面积-边长守恒**：$$\text{area}(1,2,3)\cdot \text{dist}(1,2)=C$$
- **R3-7 多对象位置轮换**：不同形状沿结构顺/逆时针相邻换位（直线/三角形/矩形/五角星）
- **R3-8 多对象密度变化**：多对象密度按位置延续增减（直线/三角形/矩形/五角星）
- **R3-9 多对象尺度变化**：多对象尺度按位置延续增减（直线/三角形/矩形/五角星）
- **R3-10 多对象形状变化**：多对象形状按位置延续转换（直线/三角形/矩形/五角星）
- **R3-11 正弦位置转换**：位置沿正弦采样点连续滑动（45°倍数/0°）

### R4 抽象逻辑演化
具体规则待定