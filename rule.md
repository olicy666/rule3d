## 5.1 SIMPLE（简单规则，共 14 条）
设计目标：送分题，变化直观，某些规则可以极其简单。
### **S01 – 尺度线性变化（Scale Linear）**
* B 的大小 = k × A
* C 的大小 = k × B = k² × A
* k 为随机比例
### **S02 – 单轴缩放（Single-Axis Stretch）**
* 仅对一个维度 scaleX/scaleY/scaleZ 做变化
* 例如 width 扩大
### **S03 – 全局统一缩放（Uniform Scaling）**
* B = A 放大或缩小
* C = B 再放大/缩小
### **S04 – 固定轴旋转（Fixed Axis Rotation）**
* 绕 X/Y/Z 某一轴旋转固定角度 Δθ
* A→B→C 的角度累积增长
### **S05 – 三轴旋转（Full Euler Rotation）**
* 同时对三个欧拉角做增量变化
### **S06 – 平移（Translation）**
* 选定一个方向平移向量 d
* A 在原点附近
* B = A + d
* C = B + d
### **S07 – 二段平移（Two-step Translation）**
* A→B 采用 d1
* B→C 采用同方向但不同长度的 d2（比例保持）
### **S08 – 形状替换（Shape Substitution）**
* sphere → cylinder → cone → cube → sphere（循环）
### **S09 – 形状固定但比例变化（Shape Constant Scale Change）**
* 只改变同一形状的 scale（例如 cube 的 x/y/z 比例）
### **S10 – 颜色/密度变化（Point Density Change）**
* 点云密度变化（A 低密度 → B 中等密度 → C 高密度）
* 点云数量保持一致，通过“保留比例”实现密度感
### **S11 – 位姿保持 + 大小竖向递增**
* 位置和旋转不变，size 递增
### **S12 – 距离固定缩放（Distance-Scale Matched）**
* 缩放的同时按固定方式移动，使重心位置保持一致
### **S13 – 单原语旋转 + 不动点存在**
* A/B/C 都有一个固定点，不动点作为旋转中心
### **S14 – 规则不变（Identity Rule，送分）**
* A、B、C 完全相同
* 用于验证模型基本一致性
* 非常重要的最基础“送分题”

---

## 5.2 MEDIUM（中等规则，共 14 条）
目标：引入两个物体之间关系，清晰可视。
### **M01 – 接触模式链（Separate → Touch → Intersect）**
* A：分离
* B：轻微接触
* C：相交
### **M02 – 相交深度线性增长（Intersection Depth Growth）**
* 按固定方向逐步推进
### **M03 – 包含关系（Containment）**
* inner 在 outer 里移动但始终不越界
### **M04 – 空间相对位置（Above/Below/Left/Right）**
* A→B→C 在同一方向上移动
### **M05 – 重心偏移（Center-of-Mass Shift）**
* 两个物体的重心相对位置发生线性变化
### **M06 – 平行 / 垂直关系保持**
* 两个物体的主轴保持平行或垂直
### **M07 – 轴角度线性变化**
* 两个物体的主轴角度逐渐变化
### **M08 – 对称轴保持（Axes Symmetry）**
* 两物体保持镜像对称
### **M09 – 距离线性变化（Distance Rule）**
* 两物体间距离递增/递减
### **M10 – 排列模式变化（Pattern Movement）**
* 多个物体按相同方向同时偏移
* 保持队形
### **M11 – 层级保持（Hierarchy Preserved）**
* one object “owns” another（如小球附在圆柱上）
### **M12 – 组合比例关系（Composite Ratio Rule）**
* 大小/距离/角度按比例耦合变化（但不复杂）
### **M13 – 面对齐 / 边对齐 / 点对齐（Alignment）**
* A：面对齐
* B：边对齐
* C：顶点对齐
  （或其他三段式）
### **M14 – 形状家族变化（Shape Family Relation）**
* 球 → 椭球 → 球
* 或不同立方体的形变序列

---

## 5.3 COMPLEX（复杂规则，共 12 条）
目标：引入拓扑、布尔、联合变化、多属性联动。
### **C01 – 尺度 + 平移联动（Scale + Translation Coupled）**
* 缩放的同时沿比例偏移
### **C02 – 旋转 + 比例联动（Rotation + Scaling Coupled）**
* A：旋转小
* B：旋转+变大
* C：旋转更多 + 更大
### **C03 – 三步序列（Multi-step Sequence）**
* 每一步变化不同，但函数化重复
* 例如 A→B：scale 乘 1.2
* B→C：再乘 1.2
* 位置或朝向也同时按某函数变化
### **C04 – 布尔序列（Boolean Sequence）**
* 例如 sphere & cube：
  * A：分离
  * B：几乎相交
  * C：深度相交（可明显看到布尔形状）
### **C05 – 洞拓扑（Hole Topology）**
* A：立方体
* B：Cube – Cylinder（穿孔）
* C：Cylinder 孔变大或孔位置偏移
### **C06 – 隧道拓扑（Tunnel）**
* A：cube
* B：两个 cylinder 穿过 cube
* C：穿透位置或方向变化
### **C07 – 截面拓扑（Cross-section）**
* A：圆柱 & 立方体
* B：轻微相交
* C：形成更明显的截面形状
### **C08 – 对称 + 缩放联动（Symmetry + Scale）**
* 两物体镜像对称
* 镜像轴不变
* 同比例变大或变小
### **C09 – 角色互换（Role Exchange）**
* A：obj1 左 obj2 右
* B：中点开始互换
* C：obj1 和 obj2 完成位置、大小、方向交换
### **C10 – 相交角度渐增（Progressive Clash Angle）**
* 两个长形物体（圆柱）
* A：轻微角度
* B：更大角度
* C：接近垂直
### **C11 – 接触面积递增（Contact Area Growth）**
* 圆柱与立方体接触面积逐步增加
  * A：点接触
  * B：线接触
  * C：面接触
### **C12 – 多物体结构规则（Multi-Object Configuration）**
* 三个物体形成某种构型（如等边三角形）
* B/C 保持整体形态但整体旋转/缩放/偏移

根据上诉的40条规则可再次分化为四个大类：
R1 物体属性推理：对应的编号是：S01–S14 + M14 + C01–C03，R1-1 基础尺度/比例属性

规则编号：S01, S02, S03, S09, S12

对应推理能力：
单物体“连续度量属性变换”推理。

R2 成对空间关系推理： 对应编号：M01–M07 + M09 + C10–C11

R3 多物体构型推理：对应编号：M08 + M10–M13 + C08–C09 + C12

R4 结构与拓扑推理：对应编号：C04–C07