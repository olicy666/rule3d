from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from datetime import datetime
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from raven3d.dataset import DatasetGenerator, GenerationConfig
from raven3d.factory import create_default_registry
from raven3d.rules.groups import list_all_rules

RECORDS_PATH = Path("exam_records.csv")
RESULTS_DIR = Path("results")
TOTAL_QUESTIONS = 20
PLY_HEIGHT = 320
POINTS_PER_CLOUD = 16384


def sort_rule_ids(rule_ids: List[str]) -> List[str]:
    def key(rule_id: str) -> tuple:
        if rule_id.startswith("R") and "-" in rule_id:
            group_part, _, idx_part = rule_id.partition("-")
            group_num = group_part[1:]
            if group_num.isdigit() and idx_part.isdigit():
                return (int(group_num), int(idx_part), rule_id)
        return (99, 999, rule_id)

    return sorted(rule_ids, key=key)


RULE_IDS = sort_rule_ids(list_all_rules())
RECORD_COLUMNS = ["username", "mode", "score", "total", "accuracy", "reason", "result_path"]


def pl_component(ply_content_str: str, height: int = PLY_HEIGHT, reset_nonce: int = 0) -> None:
    import uuid

    container_id = f"pc_{reset_nonce}_{uuid.uuid4().hex}"
    overlay_id = f"{container_id}_overlay"
    ply_json = json.dumps(ply_content_str)
    html = f"""
    <div style="width:100%; height:{height}px; position:relative; background:#fff;">
      <div id="{container_id}" style="width:100%; height:100%;"></div>
      <div id="{overlay_id}" style="
        position:absolute; left:8px; top:8px; font-size:12px; color:#333;
        background:rgba(255,255,255,0.85); border:1px solid rgba(0,0,0,0.08);
        padding:6px 8px; border-radius:8px; pointer-events:none;">
        加载中…
      </div>
    </div>
    <!-- reset:{reset_nonce} -->
    <script type="importmap">
      {{
        "imports": {{
          "three": "https://unpkg.com/three@0.161.0/build/three.module.js"
        }}
      }}
    </script>
    <script type="module">
      import * as THREE from "three";
      import {{ OrbitControls }} from "https://unpkg.com/three@0.161.0/examples/jsm/controls/OrbitControls.js";
      import {{ PLYLoader }} from "https://unpkg.com/three@0.161.0/examples/jsm/loaders/PLYLoader.js";

      const container = document.getElementById("{container_id}");
      const overlay = document.getElementById("{overlay_id}");
      function setOverlay(text) {{
        if (overlay) overlay.textContent = text;
      }}
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0xffffff);

      const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1e9);
      camera.position.set(0, 0, 5);

      const renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: "high-performance" }});
      renderer.setPixelRatio(1);
      renderer.setClearColor(0xffffff, 1);
      container.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.screenSpacePanning = true;

      scene.add(new THREE.AmbientLight(0xffffff, 1.0));

      const loader = new PLYLoader();
      const plyText = {ply_json};
      const blob = new Blob([plyText], {{ type: "text/plain" }});
      const url = URL.createObjectURL(blob);

      function fitCamera(geometry, material) {{
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        if (!box) return;
        const size = new THREE.Vector3();
        const center = new THREE.Vector3();
        box.getSize(size);
        box.getCenter(center);

        const radius = size.length() * 0.5;
        controls.target.copy(center);

        const fov = THREE.MathUtils.degToRad(camera.fov);
        let dist = radius / Math.tan(fov / 2);
        dist *= 1.15;

        const dir = new THREE.Vector3().subVectors(camera.position, controls.target).normalize();
        if (dir.lengthSq() < 1e-12) dir.set(0, 0, 1);
        camera.position.copy(center).addScaledVector(dir, dist);
        camera.near = Math.max(dist / 10000, 0.0001);
        camera.far = Math.max(dist * 1000, camera.near + 1);
        camera.updateProjectionMatrix();

        controls.minDistance = dist * 0.85;
        controls.maxDistance = dist * 1.35;

        const sizeVal = Math.max(radius * 0.002, 0.001) * 3.0;
        material.size = sizeVal;
      }}

      loader.load(url, (geometry) => {{
        URL.revokeObjectURL(url);
        if (geometry.computeVertexNormals) {{
          geometry.computeVertexNormals();
        }}
        const material = new THREE.PointsMaterial({{
          size: 1.0,
          vertexColors: true,
          sizeAttenuation: true,
          transparent: true,
          opacity: 1.0
        }});
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        fitCamera(geometry, material);
        setOverlay("加载完成");
      }}, undefined, (err) => {{
        URL.revokeObjectURL(url);
        console.error(err);
        setOverlay("加载失败，请检查控制台错误");
      }});

      function resize() {{
        const width = container.clientWidth || 300;
        const height = container.clientHeight || {height};
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height, false);
      }}

      const observer = new ResizeObserver(resize);
      observer.observe(container);
      resize();

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();
    </script>
    """
    components.html(html, height=height)


def stable_seed(username: str, mode: str) -> int:
    key = f"{username}|{mode}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def init_state() -> None:
    defaults = {
        "logged_in": False,
        "is_admin": False,
        "username": "",
        "mode": RULE_IDS[0] if RULE_IDS else "",
        "question_index": 0,
        "answers": {},
        "exam_generated": False,
        "temp_dir_obj": None,
        "exam_dir": "",
        "exam_meta": [],
        "result_ready": False,
        "result_json": "",
        "result_saved": False,
        "score": 0,
        "seed": None,
        "viewer_reset_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if RULE_IDS and st.session_state.get("mode") not in RULE_IDS:
        st.session_state["mode"] = RULE_IDS[0]


def reset_exam_state() -> None:
    temp_dir_obj = st.session_state.get("temp_dir_obj")
    if temp_dir_obj is not None:
        try:
            temp_dir_obj.cleanup()
        except Exception:
            pass
    st.session_state.temp_dir_obj = None
    st.session_state.exam_dir = ""
    st.session_state.exam_meta = []
    st.session_state.answers = {}
    st.session_state.exam_generated = False
    st.session_state.question_index = 0
    st.session_state.result_ready = False
    st.session_state.result_json = ""
    st.session_state.result_saved = False
    st.session_state.score = 0
    st.session_state.seed = None
    st.session_state.viewer_reset_nonce = 0
    for idx in range(TOTAL_QUESTIONS):
        st.session_state.pop(f"answer_{idx}", None)


def generate_exam(username: str, mode: str) -> None:
    reset_exam_state()
    temp_dir_obj = tempfile.TemporaryDirectory()
    exam_dir = Path(temp_dir_obj.name)
    st.session_state.temp_dir_obj = temp_dir_obj
    st.session_state.exam_dir = str(exam_dir)
    seed = stable_seed(username, mode)
    st.session_state.seed = seed
    registry = create_default_registry()
    config = GenerationConfig(n_points=POINTS_PER_CLOUD, rule_filter={mode})
    generator = DatasetGenerator(registry, config=config, seed=seed)
    generator.generate_dataset(exam_dir, TOTAL_QUESTIONS)
    meta_path = exam_dir / "meta.json"
    st.session_state.exam_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st.session_state.exam_generated = True


def load_ply_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build_result(
    username: str, mode: str, meta: List[Dict], answers: Dict[int, str]
) -> Dict:
    details = []
    correct_count = 0
    wrong_reasons: List[str] = []
    for idx, entry in enumerate(meta):
        user_option = answers.get(idx)
        gt_option = entry.get("gt_option")
        cand_reasons = {
            "A": entry.get("cand1_reason", ""),
            "B": entry.get("cand2_reason", ""),
            "C": entry.get("cand3_reason", ""),
            "D": entry.get("cand4_reason", ""),
        }
        is_correct = user_option == gt_option
        if is_correct:
            correct_count += 1
        if user_option is None:
            wrong_reason = "未作答"
        else:
            wrong_reason = cand_reasons.get(user_option, "")
        if not is_correct:
            wrong_reasons.append(wrong_reason or "未知原因")
        details.append(
            {
                "id": entry.get("id", f"q{idx + 1:02d}"),
                "rule_id": entry.get("rule_id", mode),
                "gt_option": gt_option,
                "user_option": user_option,
                "is_correct": is_correct,
            }
        )
    total = len(meta)
    accuracy = correct_count / total if total else 0.0
    reason_ratio = {}
    if wrong_reasons:
        counts = Counter(wrong_reasons)
        total_wrong = sum(counts.values())
        reason_ratio = {k: round(v / total_wrong, 4) for k, v in counts.items()}
    return {
        "username": username,
        "mode": mode,
        "total": total,
        "score": correct_count,
        "accuracy": round(accuracy, 4),
        "error_reason_ratio": reason_ratio,
        "questions": details,
    }


def append_record(record: Dict) -> None:
    file_exists = RECORDS_PATH.exists()
    with RECORDS_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def safe_slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    return cleaned.strip("_") or "user"


def load_records() -> pd.DataFrame:
    if not RECORDS_PATH.exists():
        return pd.DataFrame(columns=RECORD_COLUMNS)
    try:
        df = pd.read_csv(RECORDS_PATH)
    except pd.errors.ParserError:
        df = pd.read_csv(RECORDS_PATH, engine="python", on_bad_lines="skip")
    for col in RECORD_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[RECORD_COLUMNS]


def render_admin() -> None:
    st.title("Raven3D 管理后台")
    if st.button("退出登录"):
        reset_exam_state()
        st.session_state.logged_in = False
        st.session_state.is_admin = False
        st.session_state.username = ""
        st.rerun()

    df = load_records()
    if df.empty:
        st.info("暂无考试记录。")
        return

    st.subheader("考试记录")
    st.dataframe(df, use_container_width=True)

    st.subheader("删除考试记录")
    df = df.reset_index(drop=True)
    label_to_index = {}
    option_labels = []
    for idx, row in df.iterrows():
        label = (
            f"{idx + 1}: {row.get('username', '')} | {row.get('mode', '')} | "
            f"{row.get('score', '')}/{row.get('total', '')} | {row.get('accuracy', '')}"
        )
        option_labels.append(label)
        label_to_index[label] = idx
    selected = st.multiselect("选择要删除的记录", option_labels)
    if st.button("删除选中记录"):
        if not selected:
            st.warning("请先选择要删除的记录。")
        else:
            drop_indices = [label_to_index[label] for label in selected]
            new_df = df.drop(index=drop_indices).reset_index(drop=True)
            new_df.to_csv(RECORDS_PATH, index=False)
            st.success(f"已删除 {len(drop_indices)} 条记录。")
            st.rerun()

    st.subheader("各用户正确率")
    user_acc = df.groupby("username")["accuracy"].mean().sort_values(ascending=False)
    st.bar_chart(user_acc)

    st.subheader("下载答题结果")
    df = df.reset_index(drop=True)
    downloadable = df[df["result_path"].notna() & df["result_path"].astype(str).str.len() > 0]
    if downloadable.empty:
        st.info("暂无可下载的 result.json。")
    else:
        label_to_path = {}
        options = []
        for idx, row in downloadable.iterrows():
            label = (
                f"{idx + 1}: {row.get('username', '')} | {row.get('mode', '')} | "
                f"{row.get('score', '')}/{row.get('total', '')}"
            )
            options.append(label)
            label_to_path[label] = Path(str(row.get("result_path", "")))
        selected_label = st.selectbox("选择记录", options)
        selected_path = label_to_path.get(selected_label)
        if selected_path and selected_path.exists():
            st.download_button(
                "下载 result.json",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
                mime="application/json",
            )
        else:
            st.warning("该记录的 result.json 文件不存在。")

    st.subheader("规则平均正确率")
    mode_acc = df.groupby("mode")["accuracy"].mean()
    line_df = pd.DataFrame(
        {"accuracy": [mode_acc.get(rule_id, 0.0) for rule_id in RULE_IDS]},
        index=RULE_IDS,
    )
    st.line_chart(line_df)

    st.download_button(
        "下载 CSV",
        data=RECORDS_PATH.read_bytes(),
        file_name="exam_records.csv",
        mime="text/csv",
    )


def render_exam() -> None:
    st.title("Raven3D 3D 逻辑推理考试")

    st.sidebar.header("考试设置")
    st.sidebar.write(f"用户：{st.session_state.username}")

    selected_mode = st.sidebar.selectbox(
        "Mode (规则 ID)", RULE_IDS, index=RULE_IDS.index(st.session_state.mode)
    )
    if selected_mode != st.session_state.mode:
        st.session_state.mode = selected_mode
        reset_exam_state()
        st.sidebar.info("已切换模式，请重新生成试卷。")

    button_label = "生成试卷" if not st.session_state.exam_generated else "重新生成试卷"
    if st.sidebar.button(button_label):
        with st.spinner("生成题目中..."):
            generate_exam(st.session_state.username, st.session_state.mode)
        st.sidebar.success("试卷已生成。")

    if st.sidebar.button("退出登录"):
        reset_exam_state()
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.is_admin = False
        st.rerun()

    if not st.session_state.exam_generated:
        st.info("请先在侧边栏生成试卷。")
        return

    answered = len(st.session_state.answers)
    st.sidebar.metric("已作答", f"{answered}/{TOTAL_QUESTIONS}")
    current_q = st.sidebar.number_input(
        "当前题号",
        min_value=1,
        max_value=TOTAL_QUESTIONS,
        value=st.session_state.question_index + 1,
        step=1,
    )
    st.session_state.question_index = int(current_q) - 1

    idx = st.session_state.question_index
    entry = st.session_state.exam_meta[idx]
    exam_root = Path(st.session_state.exam_dir)

    st.subheader(f"题目 {idx + 1}/{TOTAL_QUESTIONS}")
    if st.button("重置当前题目视角"):
        st.session_state.viewer_reset_nonce += 1
        st.rerun()

    ref_cols = st.columns(2)
    reset_nonce = st.session_state.viewer_reset_nonce
    with ref_cols[0]:
        st.caption("Ref1")
        pl_component(
            load_ply_text(exam_root / entry["ref1_path"]),
            reset_nonce=reset_nonce,
        )
    with ref_cols[1]:
        st.caption("Ref2")
        pl_component(
            load_ply_text(exam_root / entry["ref2_path"]),
            reset_nonce=reset_nonce,
        )

    cand_cols = st.columns(4)
    cand_paths = [
        ("A", entry["cand1_path"]),
        ("B", entry["cand2_path"]),
        ("C", entry["cand3_path"]),
        ("D", entry["cand4_path"]),
    ]
    for col, (label, rel_path) in zip(cand_cols, cand_paths):
        with col:
            st.caption(f"Option {label}")
            pl_component(
                load_ply_text(exam_root / rel_path),
                reset_nonce=reset_nonce,
            )

    options = ["未作答", "A", "B", "C", "D"]
    current_answer = st.session_state.answers.get(idx, "未作答")
    answer_key = f"answer_{idx}"
    if answer_key not in st.session_state:
        st.session_state[answer_key] = current_answer
    choice = st.radio(
        "选择答案",
        options,
        index=options.index(st.session_state[answer_key]),
        key=answer_key,
        horizontal=True,
    )
    if choice == "未作答":
        st.session_state.answers.pop(idx, None)
    else:
        st.session_state.answers[idx] = choice

    nav_cols = st.columns(3)
    with nav_cols[0]:
        if st.button("上一题"):
            st.session_state.question_index = max(0, idx - 1)
            st.rerun()
    with nav_cols[1]:
        if st.button("下一题"):
            st.session_state.question_index = min(TOTAL_QUESTIONS - 1, idx + 1)
            st.rerun()
    with nav_cols[2]:
        finish = st.button("结束考试")

    if finish and not st.session_state.result_ready:
        result = build_result(
            st.session_state.username,
            st.session_state.mode,
            st.session_state.exam_meta,
            st.session_state.answers,
        )
        st.session_state.result_json = json.dumps(result, ensure_ascii=False, indent=2)
        st.session_state.result_ready = True
        st.session_state.score = result["score"]
        result_path = Path(st.session_state.exam_dir) / "result.json"
        result_path.write_text(st.session_state.result_json, encoding="utf-8")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_slug(st.session_state.username)}_{st.session_state.mode}_{timestamp}.json"
        persistent_path = RESULTS_DIR / filename
        persistent_path.write_text(st.session_state.result_json, encoding="utf-8")
        if not st.session_state.result_saved:
            record = {
                "username": st.session_state.username,
                "mode": st.session_state.mode,
                "score": result["score"],
                "total": result["total"],
                "accuracy": result["accuracy"],
                "reason": json.dumps(result["error_reason_ratio"], ensure_ascii=False),
                "result_path": str(persistent_path),
            }
            append_record(record)
            st.session_state.result_saved = True

    if st.session_state.result_ready:
        st.success(
            f"考试结束，得分 {st.session_state.score}/{TOTAL_QUESTIONS}，"
            f"正确率 {st.session_state.score / TOTAL_QUESTIONS:.2%}"
        )
        st.download_button(
            "下载 result.json",
            data=st.session_state.result_json,
            file_name="result.json",
            mime="application/json",
        )


def render_login() -> None:
    st.title("Raven3D 登录")
    with st.form("login_form"):
        username = st.text_input("姓名/ID")
        password = st.text_input("管理员密码（仅 admin）", type="password")
        submitted = st.form_submit_button("登录")
    if not submitted:
        return
    if not username.strip():
        st.error("请输入姓名/ID。")
        return
    if username.strip().lower() == "admin":
        if password != "123456":
            st.error("管理员密码错误。")
            return
        st.session_state.logged_in = True
        st.session_state.is_admin = True
        st.session_state.username = "admin"
    else:
        st.session_state.logged_in = True
        st.session_state.is_admin = False
        st.session_state.username = username.strip()
    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Raven3D Exam", layout="wide")
    init_state()
    if not st.session_state.logged_in:
        render_login()
        return
    if st.session_state.is_admin:
        render_admin()
        return
    render_exam()


if __name__ == "__main__":
    main()
