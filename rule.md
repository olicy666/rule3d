## 规则原型概要

- 场景：$$X_t=\{O_{t,1},\dots,O_{t,M_t}\},\ M_t\in\{2,3\}$$
- 物体属性：$$s,r,p,R,d$$（形状、尺度、位置、旋转、密度）
- 主要派生函数：`size, ar, axis, den, dist, dir, ang, touch, contain, cent, area, ord_x, sym`
- 模式变换：等差、等比、离散序列、刚体/仿射、联动守恒。
- meta 记录：`rule_id, rule_group, difficulty, K_R, involved_indices, base_attrs_used, derived_funcs, pattern_type, pattern_params, v1/v2/v3, M_t, frames`

## R1 物体属性推理（S01–S14 + M14 + C01–C03）
- **S01**：$$r_2=k r_1,\ r_3=k r_2$$（等比 size）
- **S02**：$$\text{size}_2-\text{size}_1=\Delta,\ \text{size}_3=2\text{size}_2-\text{size}_1$$
- **S03**：$$r_{2,x}=k r_{1,x},\ r_{3,x}=k r_{2,x}$$
- **S04**：$$(\text{ar}_1,\text{ar}_2,\text{ar}_3)=(c_1,c_2,c_1)$$
- **S05**：$$R_{t+1}=R_t\cdot\text{Rot}(\hat{u},\theta)$$
- **S06**：$$(R_1,R_2,R_3)=(Q_0,Q_{90},Q_{180})$$
- **S07**：$$p_{t+1}=p_t+\Delta p$$
- **S08**：$$(p_1,p_2,p_3)=(c_1,c_2,c_1)$$
- **S09**：$$d_2-d_1=\Delta,\ d_3=2d_2-d_1$$
- **S10**：$$d_2=k d_1,\ d_3=k d_2$$
- **S11**：$$(s_1,s_2,s_3)=(a,b,a)$$
- **S12**：$$(s_1,s_2,s_3)=(a,b,c)$$
- **S13**：$$\text{cent}(S_t)=\text{const},\ r_{t+1}=k r_t$$（位置联动）
- **S14**：属性保持恒等
- **M14**：$$\text{size}(i)+\text{size}(j)=C$$，一增一减
- **C01**：$$r_{t+1}=k r_t,\ R_{t+1}=R_t\text{Rot}(\hat{u},\theta)$$
- **C02**：$$v_2=v_1+\Delta,\ v_3=v_2$$
- **C03**：$$s_2\ne s_1\Rightarrow \text{size}_2=k\text{size}_1,\ s_3=s_2$$

## R2 成对空间关系（M01–M07 + M09 + C10–C11）
- **M01**：$$\text{dist}_3=2\text{dist}_2-\text{dist}_1$$
- **M02**：$$\text{dist}_2=k\text{dist}_1,\ \text{dist}_3=k\text{dist}_2$$
- **M03**：方向保持，距离等差/等比
- **M04**：方向旋转等差角，距离恒定
- **M05**：$$(\text{touch}_1,\text{touch}_2,\text{touch}_3)=(0,1,1)$$
- **M06**：$$(\text{contain}_1,\text{contain}_2,\text{contain}_3)=(0,1,0)$$
- **M07**：$$\text{ang}_3=2\text{ang}_2-\text{ang}_1$$
- **M09**：$$\text{dist}(i,j)-\text{dist}(k,l)=C$$
- **C10**：刚体一致变换，所有 pairwise dist 不变
- **C11**：共同旋转，夹角恒定

## R3 多物体构型（M08 + M10–M13 + C08–C09 + C12）
- **M08**：$$\text{area}_3=2\text{area}_2-\text{area}_1$$
- **M10**：$$\text{ord}_x(S_t)$$ 按固定置换循环
- **M11**：$$\text{cent}_{t+1}=\text{cent}_t+\Delta c$$
- **M12**：$$v_{t+1}=k v_t,\ v_t=[\text{dist}(1,2),\text{dist}(1,3),\text{dist}(2,3)]$$
- **M13**：$$(\text{sym}_1,\text{sym}_2,\text{sym}_3)=(0,1,1)$$
- **C08**：$\text{sym}(S_2)=1,\ X_3=QX_2+t$
- **C09**：$$u_t=\|\text{cent}(S_a)-\text{cent}(S_b)\|,\ u_3=2u_2-u_1$$
- **C12**：$$\text{area}(1,2,3)\cdot \text{dist}(1,2)=C$$

## 删除的规则
R4 结构与拓扑类别及 `C04/C05/C06/C07` 全部移除，规则总数为 36；不再提供 `r4-only`、`all-minus-r4` 等模式。
