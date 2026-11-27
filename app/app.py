"""
app/app.py
Bảng điều khiển Bento 2 Tab phân tích phản hồi sinh viên UIT-VSFC (Song ngữ 100% Tiếng Việt / Tiếng Anh).
Đối sánh trực tiếp PhoBERT Đa nhiệm và Mô hình Cơ sở (TF-IDF + Hồi quy Logistic).
"""

import os
import sys
sys.dont_write_bytecode = True
import html
import json
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
import torch

APP_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from app.ui_charts import (
        compute_prediction_features,
        extract_token_feature_importance,
        format_class_label,
        inject_vibrant_theme,
        make_confidence_gauge,
        make_confusion_matrix_heatmap,
        make_headline_comparison_chart,
        make_label_distribution_chart,
        make_multimetric_radar_chart,
        make_per_class_f1_chart,
        make_probability_bar_chart,
        make_sentence_multitask_radar_chart,
        make_split_donut_chart,
        make_token_importance_chart,
        render_kpi_grid,
        render_section_header,
        render_segmented_tokens,
        render_top_title_bar,
    )
except (ImportError, ModuleNotFoundError):
    from ui_charts import (
        compute_prediction_features,
        extract_token_feature_importance,
        format_class_label,
        inject_vibrant_theme,
        make_confidence_gauge,
        make_confusion_matrix_heatmap,
        make_headline_comparison_chart,
        make_label_distribution_chart,
        make_multimetric_radar_chart,
        make_per_class_f1_chart,
        make_probability_bar_chart,
        make_sentence_multitask_radar_chart,
        make_split_donut_chart,
        make_token_importance_chart,
        render_kpi_grid,
        render_section_header,
        render_segmented_tokens,
        render_top_title_bar,
    )

from data.dataset import SENTIMENT_LABELS, TOPIC_LABELS, segment_text
from models.baseline_model import VietnameseFeedbackBaseline
from models.phobert_model import PhoBERTMultiTask

st.set_page_config(
    page_title="UIT-VSFC AI Feedback Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASELINE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
PHOBERT_DIR = os.path.join(PROJECT_ROOT, "saved_models", "phobert")
METRICS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics.json")

SCENARIOS: List[Dict[str, str]] = [
    {
        "id": "01",
        "title_vi": "01 · Giảng viên (Tích cực) — Giảng dạy nhiệt tình",
        "title_en": "01 · Lecturer (Positive) — Enthusiastic teaching",
        "text": "Thầy cô giảng dạy rất nhiệt tình, giải đáp thắc mắc chi tiết và hỗ trợ sinh viên hết mình.",
    },
    {
        "id": "02",
        "title_vi": "02 · Giảng viên (Tiêu cực) — Vào trễ, dạy nhanh",
        "title_en": "02 · Lecturer (Negative) — Late & rushed lecture",
        "text": "Giảng viên vào lớp trễ, giảng bài quá nhanh và không hướng dẫn sinh viên làm bài tập.",
    },
    {
        "id": "03",
        "title_vi": "03 · Cơ sở vật chất (Tích cực) — Phòng tốt, máy lạnh êm",
        "title_en": "03 · Facility (Positive) — Modern room & cool AC",
        "text": "Phòng học khang trang, máy lạnh chạy êm và máy chiếu phục vụ thuyết trình rất sắc nét.",
    },
    {
        "id": "04",
        "title_vi": "04 · Cơ sở vật chất (Tiêu cực) — Phòng nóng, rớt mạng",
        "title_en": "04 · Facility (Negative) — Hot room & broken AC",
        "text": "Phòng học quá nóng bức, máy lạnh bị hỏng và kết nối wifi liên tục bị rớt mạng.",
    },
    {
        "id": "05",
        "title_vi": "05 · Chương trình ĐT (Tích cực) — Thực tế, hữu ích",
        "title_en": "05 · Curriculum (Positive) — Practical career skills",
        "text": "Nội dung môn học rất thực tế, bài tập lớn giúp nâng cao kỹ năng nghề nghiệp rõ rệt.",
    },
    {
        "id": "06",
        "title_vi": "06 · Chương trình ĐT (Tiêu cực) — Thi dày, nặng lý thuyết",
        "title_en": "06 · Curriculum (Negative) — Dense exams & heavy theory",
        "text": "Lịch thi bố trí quá dày đặc và chương trình học quá nặng về lý thuyết hàn lâm.",
    },
    {
        "id": "07",
        "title_vi": "07 · Chủ đề khác (Trung tính) — Thông báo học phí",
        "title_en": "07 · Others (Neutral) — Tuition & registration notice",
        "text": "Nhà trường đã thông báo thời hạn đóng học phí và mở cổng đăng ký môn học học kỳ mới.",
    },
    {
        "id": "08",
        "title_vi": "08 · Mỉa mai / Ẩn ý — Phòng học xông hơi giữa trưa",
        "title_en": "08 · Sarcasm / Irony — Midday sauna classroom",
        "text": "Cảm ơn nhà trường đã tạo cơ hội cho sinh viên trải nghiệm phòng học xông hơi giữa trưa nắng.",
    },
    {
        "id": "09",
        "title_vi": "09 · Tự nhập câu phản hồi mới",
        "title_en": "09 · Custom feedback input",
        "text": "",
    },
]

DEFAULT_CM_SENT_PHOBERT = [[1361, 21, 27], [27, 92, 48], [50, 18, 1522]]
DEFAULT_CM_SENT_BASELINE = [[1302, 63, 44], [58, 66, 43], [112, 66, 1412]]
DEFAULT_CM_TOPIC_PHOBERT = [
    [2126, 123, 4, 37],
    [87, 458, 4, 23],
    [2, 4, 138, 1],
    [37, 32, 3, 87],
]
DEFAULT_CM_TOPIC_BASELINE = [
    [1943, 238, 5, 104],
    [71, 454, 5, 42],
    [2, 5, 137, 1],
    [33, 41, 2, 83],
]


def inject_realtime_typing_hook() -> None:
    """Kích hoạt tự động gửi nội dung ô nhập phản hồi theo thời gian thực ngay khi gõ (debounce 320ms, bảo toàn gõ dấu tiếng Việt IME)."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function bindRealtimeTextarea() {
                const areas = doc.querySelectorAll('textarea');
                areas.forEach((ta) => {
                    if (ta.dataset.rtBound === '1') return;
                    ta.dataset.rtBound = '1';
                    ta.dataset.lastCommitted = ta.value;
                    let timer = null;
                    let composing = false;

                    function commitNow() {
                        if (composing) return;
                        if (ta.value === ta.dataset.lastCommitted) return;
                        ta.dataset.lastCommitted = ta.value;
                        const ev = new KeyboardEvent('keydown', {
                            key: 'Enter',
                            code: 'Enter',
                            keyCode: 13,
                            which: 13,
                            ctrlKey: true,
                            metaKey: true,
                            bubbles: true,
                            cancelable: true
                        });
                        ta.dispatchEvent(ev);
                    }

                    ta.addEventListener('compositionstart', () => {
                        composing = true;
                        if (timer) clearTimeout(timer);
                    });
                    ta.addEventListener('compositionend', () => {
                        composing = false;
                        if (timer) clearTimeout(timer);
                        timer = setTimeout(commitNow, 320);
                    });
                    ta.addEventListener('input', () => {
                        if (composing) return;
                        if (timer) clearTimeout(timer);
                        timer = setTimeout(commitNow, 320);
                    });
                });
            }
            bindRealtimeTextarea();
            if (!window.parent.__rtObserverBound) {
                window.parent.__rtObserverBound = true;
                const obs = new MutationObserver(bindRealtimeTextarea);
                obs.observe(doc.body, { childList: true, subtree: true });
            }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


@st.fragment
def render_live_demo_tab(lang: str, baseline_model: Any, phobert_model: Any) -> None:
    """Hiển thị Tab 1 với cơ chế @st.fragment giúp cập nhật kết quả và biểu đồ tức thời theo thời gian thực khi gõ."""
    left_col, right_col = st.columns([0.92, 1.68], gap="large")

    with left_col:
        scenario_idx = st.selectbox(
            label="🎯 Kịch bản mẫu:" if lang == "vi" else "🎯 Preset Scenario:",
            options=list(range(len(SCENARIOS))),
            format_func=lambda i: SCENARIOS[i]["title_vi" if lang == "vi" else "title_en"],
            index=0,
            key=f"scenario_selector_{lang}",
        )
        active_scenario = SCENARIOS[scenario_idx]

        mode_options = (
            ["So sánh 2 mô hình", "PhoBERT", "Mô hình Cơ sở"]
            if lang == "vi"
            else ["Compare Both", "PhoBERT", "Baseline"]
        )
        mode_idx = st.radio(
            label="⚙️ Mô hình:" if lang == "vi" else "⚙️ Model:",
            options=[0, 1, 2],
            format_func=lambda i: mode_options[i],
            index=0,
            horizontal=True,
            key=f"inference_mode_radio_{lang}",
        )

        text_input = st.text_area(
            label="✍️ Câu phản hồi (phân tích tức thời khi gõ):" if lang == "vi" else "✍️ Feedback Sentence (real-time as you type):",
            value=active_scenario["text"],
            height=84,
            key=f"feedback_input_{lang}_{scenario_idx}",
            placeholder=(
                "Gõ câu phản hồi sinh viên tại đây..."
                if lang == "vi"
                else "Type Vietnamese student feedback here..."
            ),
        )
        inject_realtime_typing_hook()

        cleaned = text_input.strip()
        segmented = segment_text(cleaned) if cleaned else ""
        if segmented:
            render_segmented_tokens(segmented, lang=lang)

    with right_col:
        if not cleaned:
            st.info("Gõ câu phản hồi hoặc chọn kịch bản ở cột trái để xem phân tích thời gian thực." if lang == "vi" else "Type feedback or select a scenario on the left for real-time analysis.")
        else:
            res_p: Optional[Dict[str, Any]] = None
            res_b: Optional[Dict[str, Any]] = None
            dt_p = 0.0
            dt_b = 0.0

            if mode_idx in (0, 1) and phobert_model is not None:
                t0 = time.perf_counter()
                res_p = phobert_model.predict(cleaned, word_segment=True)
                dt_p = time.perf_counter() - t0
            if mode_idx in (0, 2) and baseline_model is not None:
                t1 = time.perf_counter()
                res_b = baseline_model.predict(cleaned, word_segment=True)
                dt_b = time.perf_counter() - t1

            p_title = "🚀 PhoBERT Đa nhiệm" if lang == "vi" else "🚀 PhoBERT Multi-Task"
            b_title = "📊 Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "📊 Classical Baseline"

            if mode_idx == 0 and res_p is not None and res_b is not None:
                fc1, fc2 = st.columns(2)
                with fc1:
                    render_feature_summary_card(p_title, res_p, dt_p, "#8B5CF6", lang)
                with fc2:
                    render_feature_summary_card(b_title, res_b, dt_b, "#06B6D4", lang)
            elif res_p is not None:
                render_feature_summary_card(p_title, res_p, dt_p, "#8B5CF6", lang)
            elif res_b is not None:
                render_feature_summary_card(b_title, res_b, dt_b, "#06B6D4", lang)

            primary_res = res_p if res_p is not None else res_b
            compare_res = res_b if (mode_idx == 0 and res_p is not None and res_b is not None) else None
            primary_name = (
                ("PhoBERT Đa nhiệm" if lang == "vi" else "PhoBERT Multi-Task")
                if res_p is not None
                else ("Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "Classical Baseline")
            )
            compare_name = "Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "Classical Baseline"

            if primary_res is not None:
                pb1, pb2 = st.columns(2)
                with pb1:
                    st.plotly_chart(
                        make_probability_bar_chart(
                            primary_res["sentiment"]["distribution"],
                            "Xác suất Cảm xúc (3 Lớp)" if lang == "vi" else "Sentiment Probability (3 Classes)",
                            accent_color="#8B5CF6",
                            lang=lang,
                            compare_distribution=(
                                compare_res["sentiment"]["distribution"] if compare_res is not None else None
                            ),
                            primary_name=primary_name,
                            compare_name=compare_name,
                        ),
                        use_container_width=True,
                        key="prob_bar_sent",
                    )
                with pb2:
                    st.plotly_chart(
                        make_probability_bar_chart(
                            primary_res["topic"]["distribution"],
                            "Xác suất Chủ đề (4 Lớp)" if lang == "vi" else "Topic Probability (4 Classes)",
                            accent_color="#06B6D4",
                            lang=lang,
                            compare_distribution=(
                                compare_res["topic"]["distribution"] if compare_res is not None else None
                            ),
                            primary_name=primary_name,
                            compare_name=compare_name,
                        ),
                        use_container_width=True,
                        key="prob_bar_topic",
                    )

                sent_lbl = str(primary_res["sentiment"]["label"])
                topic_lbl = str(primary_res["topic"]["label"])
                sent_id = SENTIMENT_LABELS.index(sent_lbl) if sent_lbl in SENTIMENT_LABELS else 0
                topic_id = TOPIC_LABELS.index(topic_lbl) if topic_lbl in TOPIC_LABELS else 0
                token_weights = extract_token_feature_importance(
                    baseline_model,
                    segmented,
                    sent_label_id=sent_id,
                    topic_label_id=topic_id,
                    top_k=8,
                )

                rd1, rd2 = st.columns(2)
                with rd1:
                    st.plotly_chart(
                        make_sentence_multitask_radar_chart(
                            primary_res,
                            compare_res=compare_res,
                            lang=lang,
                            primary_name=primary_name,
                            compare_name=compare_name,
                        ),
                        use_container_width=True,
                        key="sentence_radar_7d",
                    )
                with rd2:
                    st.plotly_chart(
                        make_token_importance_chart(token_weights, lang=lang),
                        use_container_width=True,
                        key="sentence_token_importance",
                    )


@st.cache_resource(show_spinner=False)
def load_models():
    """Nạp và lưu đệm mô hình Cơ sở (TF-IDF) và PhoBERT Đa nhiệm."""
    b_model = VietnameseFeedbackBaseline.load(BASELINE_PATH) if os.path.isfile(BASELINE_PATH) else None
    p_model = None
    has_weights = os.path.isfile(os.path.join(PHOBERT_DIR, "model.pt")) or os.path.isfile(
        os.path.join(PHOBERT_DIR, "pytorch_model.bin")
    )
    has_meta = os.path.isfile(os.path.join(PHOBERT_DIR, "model_meta.json"))
    if os.path.isdir(PHOBERT_DIR) and (has_weights or has_meta):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            p_model = PhoBERTMultiTask.from_pretrained(PHOBERT_DIR, device=device)
        except Exception:
            p_model = None
    return b_model, p_model


def load_metrics_data() -> Dict[str, Any]:
    """Đọc dữ liệu thực nghiệm từ results/metrics.json kèm cấu trúc dự phòng an toàn."""
    data: Dict[str, Any] = {}
    if os.path.isfile(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data.setdefault("num_test_samples", 3166)
    data.setdefault(
        "baseline",
        {
            "sent_acc": 0.8781,
            "sent_wf1": 0.8806,
            "sent_mf1": 0.7276,
            "topic_acc": 0.8266,
            "topic_wf1": 0.8371,
            "topic_mf1": 0.7369,
            "mean_mf1": 0.7322,
            "latency_sec": 0.1325,
            "speed": 23887.95,
        },
    )
    data.setdefault(
        "phobert",
        {
            "sent_acc": 0.9397,
            "sent_wf1": 0.9377,
            "sent_mf1": 0.8429,
            "topic_acc": 0.8872,
            "topic_wf1": 0.8878,
            "topic_mf1": 0.8030,
            "mean_mf1": 0.8230,
            "latency_sec": 187.96,
            "speed": 16.84,
        },
    )
    data.setdefault(
        "dataset_stats",
        {
            "split_sizes": {"train": 11426, "validation": 1583, "test": 3166, "total": 16175},
            "sentiment_distribution": {"negative": 7439, "neutral": 698, "positive": 8038},
            "topic_distribution": {
                "lecturer": 11607,
                "training_program": 3040,
                "facility": 712,
                "others": 816,
            },
        },
    )
    return data


def _badge_color_for_sentiment(label: str) -> str:
    lbl = label.lower()
    if lbl == "positive":
        return "#10B981"
    if lbl == "negative":
        return "#F43F5E"
    return "#F59E0B"


def _badge_color_for_topic(label: str) -> str:
    lbl = label.lower()
    if lbl == "lecturer":
        return "#8B5CF6"
    if lbl == "training_program":
        return "#06B6D4"
    if lbl == "facility":
        return "#F59E0B"
    return "#94A3B8"


def _format_pair_label(raw_pair: str, lang: str) -> str:
    parts = [p.strip() for p in str(raw_pair).split("|") if p.strip()]
    if not parts:
        return ""
    return " · ".join(format_class_label(p, lang) for p in parts)


def render_feature_summary_card(
    title: str,
    result: Dict[str, Any],
    latency_sec: float,
    accent_color: str,
    lang: str,
) -> None:
    """Hiển thị thẻ kết quả dự đoán 2 tầng trực quan có thanh tiến trình Neon phát sáng và chỉ báo độ chắc chắn."""
    feats = compute_prediction_features(result)
    sent_raw = str(result["sentiment"]["label"]).lower()
    topic_raw = str(result["topic"]["label"]).lower()
    sent_name = format_class_label(sent_raw, lang)
    topic_name = format_class_label(topic_raw, lang)
    sent_color = _badge_color_for_sentiment(sent_raw)
    topic_color = _badge_color_for_topic(topic_raw)
    latency_ms = latency_sec * 1000.0

    sent_conf = min(max(float(feats["sent_conf"]), 0.0), 100.0)
    topic_conf = min(max(float(feats["topic_conf"]), 0.0), 100.0)
    joint_conf = min(max(float(feats["joint_conf"]), 0.0), 100.0)
    avg_margin = (float(feats["sent_margin"]) + float(feats["topic_margin"])) / 2.0

    sent_icon = "😊" if sent_raw == "positive" else ("😡" if sent_raw == "negative" else "😐")
    topic_icons = {"lecturer": "👨‍🏫", "training_program": "📚", "facility": "🏢", "others": "📌"}
    topic_icon = topic_icons.get(topic_raw, "📌")

    if joint_conf >= 82.0 and avg_margin >= 50.0:
        status_color = "#10B981"
        status_text = "Độ chắc chắn cao" if lang == "vi" else "High Certainty"
    elif joint_conf >= 62.0:
        status_color = "#F59E0B"
        status_text = "Độ chắc chắn vừa" if lang == "vi" else "Moderate Certainty"
    else:
        status_color = "#F43F5E"
        status_text = "Ngữ cảnh phân vân" if lang == "vi" else "Ambiguous Signal"

    lbl_sent = "CẢM XÚC" if lang == "vi" else "SENTIMENT"
    lbl_topic = "CHỦ ĐỀ" if lang == "vi" else "TOPIC"
    lbl_joint = "Đồng thuận kép" if lang == "vi" else "Joint Score"
    lbl_gap = "Cách biệt Top-2" if lang == "vi" else "Top-2 Gap"

    st.markdown(
        f'<div class="bento-card" style="border-top:3px solid {accent_color}; padding:0.85rem 1rem; margin-bottom:0.65rem;">'
        # Header: Model name + latency pill
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.65rem;">'
        f'<div style="display:flex; align-items:center; gap:0.45rem;">'
        f'<span style="width:9px; height:9px; border-radius:50%; background:{accent_color}; box-shadow:0 0 10px {accent_color}; display:inline-block;"></span>'
        f'<span style="font-size:0.98rem; font-weight:800; color:#F8FAFC;">{html.escape(title)}</span>'
        f'</div>'
        f'<span style="background:#161F30; border:1px solid {accent_color}55; color:{accent_color}; '
        f'padding:0.16rem 0.58rem; border-radius:6px; font-size:0.73rem; font-weight:600; font-family:\'Geist Mono\',monospace;">'
        f'⚡ {latency_ms:.1f} ms</span>'
        f'</div>'
        # Tier 1: 2 Large Visual Task Cards with Progress Bars
        f'<div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:0.6rem; margin-bottom:0.6rem;">'
        # Sentiment Block
        f'<div style="background:#161F30; border:1px solid #26334D; border-radius:10px; padding:0.6rem 0.75rem;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.28rem;">'
        f'<span style="font-size:0.68rem; font-weight:600; color:#94A3B8; letter-spacing:0.04em;">{sent_icon} {lbl_sent}</span>'
        f'<span style="font-size:1.02rem; font-weight:700; color:#F8FAFC; font-family:\'Geist Mono\',monospace;">{sent_conf:.1f}%</span>'
        f'</div>'
        f'<div style="font-size:0.92rem; font-weight:700; color:{sent_color}; margin-bottom:0.42rem;">{html.escape(sent_name)}</div>'
        f'<div style="width:100%; height:6px; background:#1E293B; border-radius:999px; overflow:hidden;">'
        f'<div style="width:{sent_conf:.1f}%; height:100%; background:{sent_color}; '
        f'border-radius:999px;"></div>'
        f'</div>'
        f'</div>'
        # Topic Block
        f'<div style="background:#161F30; border:1px solid #26334D; border-radius:10px; padding:0.6rem 0.75rem;">'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.28rem;">'
        f'<span style="font-size:0.68rem; font-weight:600; color:#94A3B8; letter-spacing:0.04em;">{topic_icon} {lbl_topic}</span>'
        f'<span style="font-size:1.02rem; font-weight:700; color:#F8FAFC; font-family:\'Geist Mono\',monospace;">{topic_conf:.1f}%</span>'
        f'</div>'
        f'<div style="font-size:0.92rem; font-weight:700; color:{topic_color}; margin-bottom:0.42rem;">{html.escape(topic_name)}</div>'
        f'<div style="width:100%; height:6px; background:#1E293B; border-radius:999px; overflow:hidden;">'
        f'<div style="width:{topic_conf:.1f}%; height:100%; background:{topic_color}; '
        f'border-radius:999px;"></div>'
        f'</div>'
        f'</div>'
        f'</div>'
        # Tier 2: Smart Status Strip
        f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; '
        f'background:#161F30; border:1px solid #26334D; border-radius:8px; padding:0.36rem 0.65rem; font-size:0.74rem;">'
        f'<span style="display:inline-flex; align-items:center; gap:0.35rem; color:{status_color}; font-weight:600;">'
        f'<span style="width:7px; height:7px; border-radius:50%; background:{status_color};"></span>'
        f'{status_text}</span>'
        f'<span style="color:#CBD5E1;">{lbl_joint}: <b style="color:#38BDF8; font-family:\'Geist Mono\',monospace;">{joint_conf:.1f}%</b></span>'
        f'<span style="color:#CBD5E1;">{lbl_gap}: <b style="color:#34D399; font-family:\'Geist Mono\',monospace;">+{avg_margin:.0f}%</b></span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_architecture_comparison_box(lang: str) -> None:
    """Hiển thị bảng Bento đối sánh cấu hình huấn luyện và kiến trúc giữa PhoBERT và Mô hình Cơ sở."""
    if lang == "vi":
        p_title = "🚀 Kiến trúc PhoBERT Đa nhiệm (Mô hình Đề xuất)"
        b_title = "📊 Mô hình Cơ sở (TF-IDF + Hồi quy Logistic Kép)"
        p_rows = [
            ("Mạng nền tảng", "vinai/phobert-base-v2 (135 triệu tham số, gộp từ vị chuẩn BPE)"),
            ("Cơ chế Gộp đặc trưng", "Gộp trung bình có mặt nạ chú ý (Masked Mean Pooling)"),
            ("Nhánh Phân loại Kép", "2 nhánh mạng nơ-ron đa tầng độc lập (Hàm kích hoạt GELU, Dropout 0.2)"),
            ("Chiến lược Tinh chỉnh", "Mở khóa 6 tầng biến đổi trên cùng · Tối ưu hóa AdamW (3e-5 / 4.5e-4)"),
            ("Dung lượng Lưu trữ", "516,56 MB (Tự động tải trọng số từ kho phát hành đám mây)"),
        ]
        b_rows = [
            ("Tiền xử lý Ngôn ngữ", "Chuẩn hóa Unicode NFC & tách từ ghép tiếng Việt bằng thư viện PyVi"),
            ("Trích xuất Đặc trưng", "TF-IDF 25.000 cụm từ (1-2 âm tiết, thang đo tần suất logarit)"),
            ("Bộ Phân loại Tuyến tính", "2 bộ Hồi quy Logistic độc lập cho Cảm xúc (3 lớp) & Chủ đề (4 lớp)"),
            ("Xử lý Mất cân bằng", "Tự động cân bằng trọng số lớp học nghịch đảo tần suất (C = 3,0)"),
            ("Dung lượng & Tốc độ", "1,23 MB · Tốc độ suy luận siêu nhanh (> 23.000 câu/giây trên CPU)"),
        ]
        team_note = (
            "👥 <b>Nhóm Thực hiện 25</b> · Lê Văn Hoài Rin (<code>29211332113</code>) &amp; "
            "Nguyễn Trung Hiếu (<code>29211362521</code>) · Tập dữ liệu chuẩn <b>UIT-VSFC (16.175 câu)</b>"
        )
    else:
        p_title = "🚀 PhoBERT Multi-Task Architecture (Proposed Model)"
        b_title = "📊 Classical Baseline (TF-IDF + Dual Logistic Regression)"
        p_rows = [
            ("Backbone Encoder", "vinai/phobert-base-v2 (135M parameters, Vietnamese BPE subwords)"),
            ("Sequence Pooling", "Masked Mean Pooling across valid attention mask tokens"),
            ("Dual Task Heads", "2 independent 2-layer MLP heads (GELU activation, Dropout 0.2)"),
            ("Fine-Tuning Regimen", "Top 6 Transformer layers unfrozen · AdamW (lr = 3e-5 / 4.5e-4)"),
            ("Checkpoint Size", "516.56 MB (Automated weight provisioning via GitHub Releases)"),
        ]
        b_rows = [
            ("Text Preprocessing", "Unicode NFC normalization & PyVi Vietnamese compound word segmentation"),
            ("Feature Extraction", "TF-IDF Vectorizer with 25,000 n-grams (range 1–2, sublinear TF)"),
            ("Linear Classifiers", "Dual independent Logistic Regression heads (3 Sentiments & 4 Topics)"),
            ("Imbalance Handling", "Inverse class-frequency balanced weighting (Regularization C = 3.0)"),
            ("Footprint & Speed", "1.23 MB · Ultra-fast CPU inference (> 23,000 sentences/sec)"),
        ]
        team_note = (
            "👥 <b>Project Team 25</b> · Le Van Hoai Rin (<code>29211332113</code>) &amp; "
            "Nguyen Trung Hieu (<code>29211362521</code>) · Benchmark Corpus: <b>UIT-VSFC (16,175 sentences)</b>"
        )

    c1, c2 = st.columns(2)
    for col, title, rows, color in [(c1, p_title, p_rows, "#8B5CF6"), (c2, b_title, b_rows, "#06B6D4")]:
        rows_html = "".join(
            f'<div style="display:flex; justify-content:space-between; gap:0.75rem; padding:0.38rem 0; '
            f'border-bottom:1px solid rgba(148,163,184,0.14); font-size:0.81rem;">'
            f'<span style="color:#94A3B8; font-weight:600; min-width:34%;">{html.escape(k)}</span>'
            f'<span style="color:#F8FAFC; text-align:right;">{html.escape(v)}</span></div>'
            for k, v in rows
        )
        with col:
            st.markdown(
                f'<div class="bento-card" style="border-top:3px solid {color};">'
                f'<div style="font-size:1.02rem; font-weight:800; color:#F8FAFC; margin-bottom:0.55rem;">{html.escape(title)}</div>'
                f'{rows_html}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="background:rgba(13,21,40,0.75); border:1px solid rgba(139,92,246,0.28); '
        f'border-radius:12px; padding:0.6rem 1rem; margin-bottom:0.6rem; font-size:0.82rem; color:#CBD5E1; text-align:center;">'
        f'{team_note}</div>',
        unsafe_allow_html=True,
    )


def render_error_analysis_grid(samples: List[Dict[str, Any]], lang: str = "vi") -> None:
    """Hiển thị lưới Bento 2 cột cho các ca kiểm thử điển hình trên tập kiểm thử (100% theo ngôn ngữ đã chọn)."""
    if not samples:
        return
    hdr_text = (
        "🔍 Các Ca Kiểm thử Thực tế Điển hình (PhoBERT Vượt trội & Ngôn ngữ Ẩn ý / Đa nghĩa)"
        if lang == "vi"
        else "🔍 Representative Empirical Test Cases (PhoBERT Advantage & Ambiguous Context)"
    )
    st.markdown(
        f"<div style='margin: 0.75rem 0 0.5rem 0; font-size: 1.02rem; font-weight: 700; color: #E2E8F0;'>{hdr_text}</div>",
        unsafe_allow_html=True,
    )
    lbl_case = "CA KIỂM THỬ" if lang == "vi" else "TEST CASE"
    lbl_gt = "Nhãn Thực tế" if lang == "vi" else "Ground Truth"
    lbl_base = "Mô hình Cơ sở (TF-IDF)" if lang == "vi" else "Classical Baseline"
    lbl_pho = "PhoBERT Đa nhiệm" if lang == "vi" else "PhoBERT Multi-Task"

    cols = st.columns(2)
    for idx, item in enumerate(samples):
        text_quote = html.escape(str(item.get("text", "")))
        gt = html.escape(_format_pair_label(str(item.get("ground_truth", "")), lang))
        b_pred = html.escape(_format_pair_label(str(item.get("baseline_pred", "")), lang))
        p_pred = html.escape(_format_pair_label(str(item.get("phobert_pred", "")), lang))
        raw_type = str(item.get("type", ""))
        is_phobert_win = "PhoBERT" in raw_type
        if is_phobert_win:
            sample_type = "PhoBERT Vượt trội Mô hình Cơ sở" if lang == "vi" else "PhoBERT Outperforms Baseline"
            badge_color = "#10B981"
            border_color = "rgba(16, 185, 129, 0.38)"
        else:
            sample_type = "Thử thách Ngữ nghĩa & Đa nghĩa" if lang == "vi" else "Ambiguous & Implicit Context"
            badge_color = "#F59E0B"
            border_color = "rgba(245, 158, 11, 0.35)"

        with cols[idx % 2]:
            st.markdown(
                f"""
                <div class="error-card bento-card" style="border:1px solid {border_color}; margin-bottom:0.85rem; padding:1rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.55rem;">
                        <span style="font-size:0.75rem; color:#94A3B8; font-weight:600;">{lbl_case} #{idx + 1}</span>
                        <span style="background:{badge_color}22; border:1px solid {badge_color}; color:{badge_color};
                                     padding:0.18rem 0.6rem; border-radius:999px; font-size:0.73rem; font-weight:700;">
                            {html.escape(sample_type)}
                        </span>
                    </div>
                    <div style="font-size:0.95rem; color:#F8FAFC; font-style:italic; margin-bottom:0.75rem;
                                padding:0.55rem 0.75rem; background:rgba(7,9,19,0.65); border-left:3px solid {badge_color}; border-radius:6px;">
                        “{text_quote}”
                    </div>
                    <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:0.45rem; font-size:0.78rem;">
                        <div style="background:rgba(15,23,42,0.75); padding:0.4rem 0.55rem; border-radius:6px; border:1px solid rgba(148,163,184,0.2);">
                            <div style="color:#94A3B8; font-size:0.7rem;">{lbl_gt}</div>
                            <div style="color:#E2E8F0; font-weight:700;">{gt}</div>
                        </div>
                        <div style="background:rgba(15,23,42,0.75); padding:0.4rem 0.55rem; border-radius:6px; border:1px solid rgba(6,182,212,0.25);">
                            <div style="color:#22D3EE; font-size:0.7rem;">{lbl_base}</div>
                            <div style="color:#E2E8F0; font-weight:600;">{b_pred}</div>
                        </div>
                        <div style="background:rgba(15,23,42,0.75); padding:0.4rem 0.55rem; border-radius:6px; border:1px solid rgba(139,92,246,0.35);">
                            <div style="color:#A855F7; font-size:0.7rem;">{lbl_pho}</div>
                            <div style="color:#F8FAFC; font-weight:700;">{p_pred}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def main() -> None:
    inject_vibrant_theme()
    st.session_state.setdefault("lang", "vi")

    # ── THANH TIÊU ĐỀ GỌN 1 DÒNG & 1 NÚT ICON CHUYỂN NGÔN NGỮ ───────────────
    hdr_col, lang_col = st.columns([13.2, 0.8], vertical_alignment="center")
    with lang_col:
        tooltip_lang = (
            "Chuyển sang Tiếng Anh (Switch to English)"
            if st.session_state["lang"] == "vi"
            else "Switch to Vietnamese (Chuyển sang Tiếng Việt)"
        )
        if st.button("🌐", key="btn_lang_toggle", help=tooltip_lang, use_container_width=True):
            st.session_state["lang"] = "en" if st.session_state["lang"] == "vi" else "vi"
            st.rerun()

    lang: str = st.session_state["lang"]
    with hdr_col:
        render_top_title_bar(lang=lang)

    baseline_model, phobert_model = load_models()
    metrics_data = load_metrics_data()

    tab1_label = "⚡ Demo Phân tích Phản hồi" if lang == "vi" else "⚡ Live Feedback Analyzer"
    tab2_label = "📊 Thông số Huấn luyện & Đánh giá Mô hình" if lang == "vi" else "📊 Model & Training Metrics"
    tab_demo, tab_metrics = st.tabs([tab1_label, tab2_label])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: DEMO PHÂN TÍCH PHẢN HỒI TRỰC TIẾP (2 CỘT TRÁI - PHẢI, REAL-TIME)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_demo:
        render_live_demo_tab(lang, baseline_model, phobert_model)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: THÔNG SỐ HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP UIT-VSFC
    # ══════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        render_kpi_grid(metrics_data, lang=lang)

        render_section_header(
            "KIẾN TRÚC & CẤU HÌNH HUẤN LUYỆN" if lang == "vi" else "ARCHITECTURE & TRAINING CONFIGURATION",
            (
                "🧠 Đối sánh Kiến trúc Mô hình & Siêu tham số Huấn luyện"
                if lang == "vi"
                else "🧠 Model Architecture & Hyperparameter Comparison"
            ),
            (
                "So sánh cấu trúc mạng PhoBERT Đa nhiệm và Mô hình Cơ sở trên tập dữ liệu UIT-VSFC"
                if lang == "vi"
                else "Detailed structural comparison between PhoBERT Multi-Task and Classical Baseline on UIT-VSFC"
            ),
        )
        render_architecture_comparison_box(lang=lang)

        render_section_header(
            "ĐÁNH GIÁ THỰC NGHIỆM" if lang == "vi" else "EMPIRICAL BENCHMARK",
            (
                "📊 Đối sánh Hiệu năng Thực nghiệm (3.166 Mẫu Kiểm thử)"
                if lang == "vi"
                else "📊 Empirical Performance Benchmark (Test Set N = 3,166)"
            ),
            (
                "Đánh giá độc lập trên 3.166 câu kiểm thử UIT-VSFC theo Độ chính xác và Điểm F1 Vĩ mô"
                if lang == "vi"
                else "Independent evaluation on 3,166 held-out UIT-VSFC test sentences across Accuracy and Macro-F1"
            ),
        )

        row2a_c1, row2a_c2 = st.columns(2)
        with row2a_c1:
            st.plotly_chart(
                make_headline_comparison_chart(metrics_data, lang=lang),
                use_container_width=True,
                key="chart_headline",
            )
        with row2a_c2:
            st.plotly_chart(
                make_multimetric_radar_chart(metrics_data, lang=lang),
                use_container_width=True,
                key="chart_radar",
            )

        row2b_c1, row2b_c2 = st.columns(2)
        with row2b_c1:
            st.plotly_chart(
                make_per_class_f1_chart(metrics_data, "sentiment", lang=lang),
                use_container_width=True,
                key="chart_f1_sent",
            )
        with row2b_c2:
            st.plotly_chart(
                make_per_class_f1_chart(metrics_data, "topic", lang=lang),
                use_container_width=True,
                key="chart_f1_topic",
            )

        render_section_header(
            "CHẨN ĐOÁN & PHÂN TÍCH LỖI" if lang == "vi" else "DIAGNOSTICS & ERROR ANALYSIS",
            (
                "🎯 Ma trận Nhầm lẫn & Phân tích Ca Kiểm thử Thực tế"
                if lang == "vi"
                else "🎯 Confusion Matrices & Empirical Error Analysis"
            ),
            (
                "Trực quan hóa Ma trận Nhầm lẫn 3×3 (Cảm xúc) & 4×4 (Chủ đề) cùng các câu phản hồi ẩn ý"
                if lang == "vi"
                else "Visualizing 3×3 Sentiment and 4×4 Topic Confusion Matrices alongside implicit/sarcastic test cases"
            ),
        )

        cm_root = metrics_data.get("confusion_matrix", {})
        cm_p_sent = cm_root.get("phobert", {}).get("sentiment", DEFAULT_CM_SENT_PHOBERT)
        cm_b_sent = cm_root.get("baseline", {}).get("sentiment", DEFAULT_CM_SENT_BASELINE)
        cm_p_topic = cm_root.get("phobert", {}).get("topic", DEFAULT_CM_TOPIC_PHOBERT)
        cm_b_topic = cm_root.get("baseline", {}).get("topic", DEFAULT_CM_TOPIC_BASELINE)

        cm_top_c1, cm_top_c2 = st.columns(2)
        with cm_top_c1:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_p_sent,
                    SENTIMENT_LABELS,
                    (
                        "PhoBERT Đa nhiệm — Ma trận Nhầm lẫn Cảm xúc (3×3)"
                        if lang == "vi"
                        else "PhoBERT Multi-Task — Sentiment Confusion Matrix (3×3)"
                    ),
                    colorscale="Purples",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_phobert_sent",
            )
        with cm_top_c2:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_b_sent,
                    SENTIMENT_LABELS,
                    (
                        "Mô hình Cơ sở — Ma trận Nhầm lẫn Cảm xúc (3×3)"
                        if lang == "vi"
                        else "Classical Baseline — Sentiment Confusion Matrix (3×3)"
                    ),
                    colorscale="Blues",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_baseline_sent",
            )

        cm_bot_c1, cm_bot_c2 = st.columns(2)
        with cm_bot_c1:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_p_topic,
                    TOPIC_LABELS,
                    (
                        "PhoBERT Đa nhiệm — Ma trận Nhầm lẫn Chủ đề (4×4)"
                        if lang == "vi"
                        else "PhoBERT Multi-Task — Topic Confusion Matrix (4×4)"
                    ),
                    colorscale="Purples",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_phobert_topic",
            )
        with cm_bot_c2:
            st.plotly_chart(
                make_confusion_matrix_heatmap(
                    cm_b_topic,
                    TOPIC_LABELS,
                    (
                        "Mô hình Cơ sở — Ma trận Nhầm lẫn Chủ đề (4×4)"
                        if lang == "vi"
                        else "Classical Baseline — Topic Confusion Matrix (4×4)"
                    ),
                    colorscale="Blues",
                    lang=lang,
                ),
                use_container_width=True,
                key="cm_baseline_topic",
            )

        render_error_analysis_grid(metrics_data.get("error_analysis_samples", []), lang=lang)

        render_section_header(
            "THỐNG KÊ TẬP DỮ LIỆU" if lang == "vi" else "DATASET EXPLORER",
            (
                "🗂️ Toàn cảnh Tập dữ liệu Học thuật UIT-VSFC (16.175 Câu)"
                if lang == "vi"
                else "🗂️ UIT-VSFC Academic Corpus Overview (16,175 Sentences)"
            ),
            (
                "Phân bố 16.175 câu phản hồi sinh viên qua 3 tập dữ liệu và tỷ lệ phân bố nhãn"
                if lang == "vi"
                else "Distribution across Train / Validation / Test splits and class imbalance ratios"
            ),
        )

        dataset_stats = metrics_data.get("dataset_stats", {})
        ds_c1, ds_c2, ds_c3 = st.columns(3)
        with ds_c1:
            st.plotly_chart(
                make_split_donut_chart(dataset_stats, lang=lang),
                use_container_width=True,
                key="chart_split_donut",
            )
        with ds_c2:
            st.plotly_chart(
                make_label_distribution_chart(dataset_stats, "sentiment", lang=lang),
                use_container_width=True,
                key="chart_dist_sent",
            )
        with ds_c3:
            st.plotly_chart(
                make_label_distribution_chart(dataset_stats, "topic", lang=lang),
                use_container_width=True,
                key="chart_dist_topic",
            )


if __name__ == "__main__":
    main()
