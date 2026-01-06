# Role
你是一个高级 Python 全栈开发工程师，擅长使用 Streamlit 构建数据科学应用，并精通 Three.js 的前端交互开发。

# Task
请基于我现有的 `raven3d` 核心库（用于生成 3D 逻辑推理题目），编写一个完整的 Streamlit Web 应用入口文件 `app.py`。
该应用需要包含：用户考试系统、原生 Three.js 点云可视化组件、以及管理员后台数据分析系统。

# Project Context
我的项目根目录结构如下：
- `raven3d/`: 核心算法包（不可修改）。
  - `dataset.py`: `DatasetGenerator` 类负责生成题目。
  - `io.py`: `write_ply` 和 `write_meta` 负责保存文件。
- `app.py`: 【待创建】Web 应用主入口。
- `exam_records.csv`: 【待创建】用于持久化存储用户成绩。

---

# Functional Requirements (功能需求)

## 1. 用户系统 (User & Session)
- **登录页**：用户输入“姓名/ID”。
- **Seed 机制**：将用户名转换为唯一的 `seed` (int)，传给 `raven3d` 生成器。确保同一用户每次生成的题目也是固定的（“千人千面”但“一人一卷”）。
- **Session 管理**：使用 `st.session_state` 维护登录状态、当前题目索引、当前得分、临时文件目录对象等。

## 2. 核心可视化 (Three.js Component) -必须遵守3draven/human这个文件夹下面点云可视化的方法 **重点**
用户明确要求**放弃 Plotly**，必须使用原生 **Three.js** 来获得更好的 3D 交互手感。请实现一个 Python 函数 `pl_component(ply_content_str, height=250)`，通过 `streamlit.components.v1.html` 嵌入 HTML。

**HTML/JS 内部逻辑必须严格遵守：**
1.  **依赖**：从 CDN (unpkg) 引入 `Three.js (v0.161.0)`, `OrbitControls`, `PLYLoader` (ES Modules 方式)。
2.  **材质 (Material)**：
    - 使用 `THREE.PointsMaterial`。
    - **必须开启 `vertexColors: true`**（因为题目中 Ref1 是红色，Ref2 是绿色，必须保留颜色以区分）。
    - 开启 `sizeAttenuation: true`。
3.  **交互手感 (核心复刻点)**：
    - 使用 `OrbitControls`。
    - **自动视角适配 (Fit Camera)**：加载完点云后，计算 Bounding Box。
    - **距离计算**：根据包围盒半径 `radius` 计算最佳距离 `dist = radius / Math.tan(fov/2) * 1.15`。
    - **缩放限制 (Lock Zoom)**：必须设置 `controls.minDistance = dist * 0.85` 和 `controls.maxDistance = dist * 1.35`。这是为了防止用户拉太远看不清或拉太近穿模，保持与 `1.html` 一致的“紧凑感”。
    - **点大小自动调整**：根据 radius 动态设置点大小，公式参考：`size = max(radius * 0.002, 0.001) * 3.0`。

## 3. 在线考试流程
- **侧边栏配置**：允许自由切换当前在哪一题、模式 (Mode)，档次考试只会设计一种模式，考试为一种模式也就是一种规则一场，一场考试20道题目
- **生成题目**：点击按钮后，使用 `tempfile` 调用 `raven3d.dataset.DatasetGenerator` 生成题目（包含 `.ply` 和 `meta.json`,注意是meta.json中包含答案，所以不能给考试者看，该文件用于最后与用户所提交上来的答案来比较是否答案正确）。
- **答题界面布局**：
  - **第一行 (参考图)**：2 列布局 (Ref1, Ref2)。
  - **第二行 (选项)**：4 列布局 (Cand1, Cand2, Cand3, Cand4)。
  - **交互**：显示 A/B/C/D 单选框，提交不立即判分，存储到后台并与前面生成的meta.json
  - **结果**：结果将在最后以一个新的result.json文件返回给用户，这个文件中包含了原先生成题目时候meta.json时给的正确标准答案和用户回答的答案，然后根据此来计算正确率，以及该题正确答案的解析，就像meta.json中的cand1_reason那样

## 4. 后台管理与数据持久化
- **数据存储**：
  - 在根目录维护 `exam_records.csv`。
  - 字段：`username` (用户), `score` (得分), `total` (总题数), `accuracy` (正确率)，`reason`（每种错误原因的占比，按meta.json的cand1_reason来）
  - **触发时机**：用户做完所有题目点击“结束”时
- **管理员入口**：
  - 在登录页增加逻辑：如果用户名为 `admin` (硬编码一个简单密码：123456)，进入“管理员后台”。
- **后台仪表盘**：
  - **数据表**：展示所有考试记录 (`st.dataframe`)。
  - **图表**：
    - 柱状图 (`st.bar_chart`)：各用户的正确率统计，一个规则一张
    - 折线图 (`st.line_chart`)：所有用户的平均正确率统计，一个规则为横坐标的一点，总共36点
  - **导出**：提供 `st.download_button` 下载最新 CSV。

---

# Implementation Notes (技术细节)
1.  **PLY 文件传递**：Python 读取 `.ply` 文件为字符串，嵌入到 HTML 模板的 JS 变量中。JS 端将其转为 `Blob`，生成 URL 供 `PLYLoader` 加载。
2.  **性能优化**：由于一个页面有 6 个 WebGL Canvas，请将 `renderer.setPixelRatio(1)` 设为 1，并控制 Canvas 高度（推荐 250px-300px）。
3.  **依赖包 (`requirements.txt`)**：需包含 `streamlit`, `pandas`, `numpy`, `scipy`, `h5py`。
4.  **异常处理**：确保 `exam_records.csv` 不存在时会自动创建 Header。

请直接生成 `app.py` 和 `requirements.txt` 的完整代码。