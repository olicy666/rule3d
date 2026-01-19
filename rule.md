## 规则原型概要

- 场景：$$X_t=\{O_{t,1},\dots,O_{t,M_t}\},\ M_t\in\{2,3\}$$
- 物体属性：$$s,r,p,R,d$$（形状、尺度、位置、姿态、密度）
- 主要派生函数：`size, ar, axis, den, dist, dir, ang, touch, contain, cent, area, ord_x, sym`
- 模式变换：等差、等比、离散序列、刚体/仿射、联动守恒。
- meta 记录：`rule_id, rule_group, difficulty, K_R, involved_indices, base_attrs_used, derived_funcs, pattern_type, pattern_params, v1/v2/v3, M_t, frames`

## R1 物体属性推理（16 条，缺 R1-9）
- **R1-1**：$$\text{size}_2-\text{size}_1=\Delta,\ \text{size}_3=2\text{size}_2-\text{size}_1$$
- **R1-2**：$$r_{t+1}=r_t\odot s,\ s_{axis}=k,\ s_{\text{others}}=\tfrac{1}{\sqrt{k}}$$
- **R1-3**：$$R_{t+1}=R_t\cdot\text{Rot}(\hat{u},\theta)$$
- **R1-4**：$$(R_1,R_2,R_3)=(Q_0,Q_{90},Q_{180})$$
- **R1-5**：$$p_{t+1}=p_t+\Delta p$$
- **R1-6**：$$d_2-d_1=\Delta,\ d_3=2d_2-d_1$$
- **R1-7**：A→B 发生形状变化的位置在 B→C 继续变化
- **R1-8**：$$\text{cent}(S_t)=\text{const},\ r_{t+1}=k r_t$$（位置联动）
- **R1-10**：$$\text{size}(i)+\text{size}(j)=C$$，一增一减
- **R1-11**：$$r_{t+1}=k r_t,\ R_{t+1}=R_t\text{Rot}(\hat{u},\theta)$$
- **R1-12**：形状/尺度/姿态在两物体间互换
- **R1-13**：移动方向与主轴一致
- **R1-14**：远离锚点变大，靠近锚点变小
- **R1-15**：整体 AABB 尺寸守恒
- **R1-16**：三物体按固定顺序迁移属性
- **R1-17**：镜像位置下尺寸一增一减

## R2 成对空间关系（R2-1–R2-8）
- **R2-1**：$$\text{dist}_2=k\text{dist}_1,\ \text{dist}_3=k\text{dist}_2$$
- **R2-2**：方向保持，距离等差/等比
- **R2-3**：方向旋转等差角，距离恒定
- **R2-4**：$$\rho_{t+1}-\rho_t=\Delta,\ \rho_t\in(0,1)$$
- **R2-5**：$$\text{ang}_3=2\text{ang}_2-\text{ang}_1$$
- **R2-6**：$$\text{dist}(i,j)-\text{dist}(k,l)=C$$
- **R2-7**：刚体一致变换，所有 pairwise dist 不变
- **R2-8**：共同旋转，夹角恒定

## R3 多物体构型（R3-1–R3-6）
- **R3-1**：$$\text{area}_3=2\text{area}_2-\text{area}_1$$
- **R3-2**：$$\text{ord}_x(S_t)$$ 按固定置换循环
- **R3-3**：$$v_{t+1}=k v_t,\ v_t=[\text{dist}(1,2),\text{dist}(1,3),\text{dist}(2,3)]$$
- **R3-4**：$\text{sym}(S_2)=1,\ X_3=QX_2+t$
- **R3-5**：$$u_t=\|\text{cent}(S_a)-\text{cent}(S_b)\|,\ u_3=2u_2-u_1$$
- **R3-6**：$$\text{area}(1,2,3)\cdot \text{dist}(1,2)=C$$

## 删除的规则
R4 结构与拓扑类别及 `C04/C05/C06/C07` 全部移除，规则总数为 30；不再提供 `r4-only`、`all-minus-r4` 等模式。
