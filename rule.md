## 规则原型概要

- 场景：$$X_t=\{O_{t,1},\dots,O_{t,M_t}\},\ M_t\in\{2,3\}$$
- 物体属性：$$s,r,p,R,d$$（形状、尺度、位置、旋转、密度）
- 主要派生函数：`size, ar, axis, den, dist, dir, ang, touch, contain, cent, area, ord_x, sym`
- 模式变换：等差、等比、离散序列、刚体/仿射、联动守恒。
- meta 记录：`rule_id, rule_group, difficulty, K_R, involved_indices, base_attrs_used, derived_funcs, pattern_type, pattern_params, v1/v2/v3, M_t, frames`

## R1 物体属性推理（S02,S04–S07,S09,S12–S14 + M14 + C01）
- **S02**：$$\text{size}_2-\text{size}_1=\Delta,\ \text{size}_3=2\text{size}_2-\text{size}_1$$
- **S04**：$$r_{t+1}=r_t\odot s,\ s_{axis}=k,\ s_{\text{others}}=\tfrac{1}{\sqrt{k}}$$
- **S05**：$$R_{t+1}=R_t\cdot\text{Rot}(\hat{u},\theta)$$
- **S06**：$$(R_1,R_2,R_3)=(Q_0,Q_{90},Q_{180})$$
- **S07**：$$p_{t+1}=p_t+\Delta p$$
- **S09**：$$d_2-d_1=\Delta,\ d_3=2d_2-d_1$$
- **S12**：A→B 发生形状变化的位置在 B→C 继续变化
- **S13**：$$\text{cent}(S_t)=\text{const},\ r_{t+1}=k r_t$$（位置联动）
- **S14**：属性保持恒等
- **M14**：$$\text{size}(i)+\text{size}(j)=C$$，一增一减
- **C01**：$$r_{t+1}=k r_t,\ R_{t+1}=R_t\text{Rot}(\hat{u},\theta)$$

## R2 成对空间关系（M02–M04 + M06–M07 + M09 + C10–C11）
- **M02**：$$\text{dist}_2=k\text{dist}_1,\ \text{dist}_3=k\text{dist}_2$$
- **M03**：方向保持，距离等差/等比
- **M04**：方向旋转等差角，距离恒定
- **M06**：$$\rho_{t+1}-\rho_t=\Delta,\ \rho_t\in(0,1)$$
- **M07**：$$\text{ang}_3=2\text{ang}_2-\text{ang}_1$$
- **M09**：$$\text{dist}(i,j)-\text{dist}(k,l)=C$$
- **C10**：刚体一致变换，所有 pairwise dist 不变
- **C11**：共同旋转，夹角恒定

## R3 多物体构型（M08 + M10 + M12 + C08–C09 + C12）
- **M08**：$$\text{area}_3=2\text{area}_2-\text{area}_1$$
- **M10**：$$\text{ord}_x(S_t)$$ 按固定置换循环
- **M12**：$$v_{t+1}=k v_t,\ v_t=[\text{dist}(1,2),\text{dist}(1,3),\text{dist}(2,3)]$$
- **C08**：$\text{sym}(S_2)=1,\ X_3=QX_2+t$
- **C09**：$$u_t=\|\text{cent}(S_a)-\text{cent}(S_b)\|,\ u_3=2u_2-u_1$$
- **C12**：$$\text{area}(1,2,3)\cdot \text{dist}(1,2)=C$$

## 删除的规则
R4 结构与拓扑类别及 `C04/C05/C06/C07` 全部移除，规则总数为 25；不再提供 `r4-only`、`all-minus-r4` 等模式。
