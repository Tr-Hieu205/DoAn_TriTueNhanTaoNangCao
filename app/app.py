"""
app/app.py
Interactive Feedback Analyzer Demo Dashboard for Vietnamese Student Feedback.
Built with Streamlit comparing Classical Baseline (TF-IDF + LR) vs PhoBERT Multi-Task (v2).
Conforms strictly to PROJECT.md § Interface Contracts & Milestone 5 specifications.
"""

import html
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Windows console encoding safeguard
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
def get_project_root() -> str:
    """Finds repository root containing PROJECT.md for robust execution."""
    curr = os.path.dirname(os.path.abspath(__file__))
    while curr and curr != os.path.dirname(curr):
        if os.path.isfile(os.path.join(curr, "PROJECT.md")):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROJECT_ROOT = get_project_root()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BASELINE_PATH = os.path.join(PROJECT_ROOT, "saved_models", "baseline.joblib")
DEFAULT_PHOBERT_DIR = os.path.join(PROJECT_ROOT, "saved_models", "phobert_multitask")

import streamlit as st
import torch
from transformers import AutoTokenizer

from models.baseline_model import (
    SENTIMENT_LABELS,
    SENTIMENT_MAP,
    TOPIC_LABELS,
    TOPIC_MAP,
    VietnameseFeedbackBaseline,
)
from models.phobert_model import PhoBERTMultiTask

# ---------------------------------------------------------------------------
# Visual Design System Constants (Themes & Bilingual I18N)
# ---------------------------------------------------------------------------
MODEL_MODE_BASELINE = "Trung bình"
MODEL_MODE_PHOBERT = "Cao"
MODEL_MODE_BOTH = "Cả hai"

MODEL_CARD_TITLES: Dict[str, str] = {
    "baseline": "Trung bình",
    "phobert": "Cao",
}

MODEL_RING_COLORS: Dict[str, str] = {
    "baseline": "#10B981",
    "phobert": "#4F46E5",
    "both": "#6366F1",
}

THEME_PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        "app_bg": "#FAFBFD",
        "header_bg": "rgba(250, 251, 253, 0.88)",
        "sidebar_bg": "#F4F6FB",
        "sidebar_border": "#EAEEF6",
        "sidebar_hover": "#ECEFF8",
        "sidebar_active_bg": "#E7EAFC",
        "sidebar_active_text": "#312E81",
        "card_bg": "#FFFFFF",
        "subcard_bg": "#FFFFFF",
        "card_border": "#EAEEF6",
        "subcard_border": "#EEF2F7",
        "card_shadow": "0 8px 28px rgba(15, 23, 42, 0.04)",
        "subcard_shadow": "0 2px 6px rgba(15, 23, 42, 0.025)",
        "input_bg": "#FAFBFD",
        "input_border": "#E2E8F0",
        "text_primary": "#0F172A",
        "text_body": "#1E293B",
        "text_muted": "#64748B",
        "text_subtle": "#475569",
        "divider": "#E6EAF3",
        "row_divider": "#F1F5F9",
        "pill_bg": "#F4F6FB",
        "pill_border": "#E2E8F0",
        "track_bg": "#EEF2F7",
        "secondary_btn_bg": "#FFFFFF",
        "secondary_btn_border": "#E2E8F0",
        "secondary_btn_text": "#334155",
        "secondary_btn_hover": "#F8FAFC",
        "brand_text": "#312E81",
        "primary": "#4F46E5",
        "primary_hover": "#4338CA",
        "winner_pct": "#312E81",
        "alert_bg": "#FFFFFF",
        "seg_track": "#E9EDF7",
        "seg_active_bg": "#FFFFFF",
        "seg_active_text": "#312E81",
        "seg_active_shadow": "0 2px 6px rgba(15, 23, 42, 0.12), 0 1px 2px rgba(15, 23, 42, 0.06)",
        "theme_icon_active": "#D97706",
    },
    "dark": {
        "app_bg": "#0B1120",
        "header_bg": "rgba(11, 17, 32, 0.88)",
        "sidebar_bg": "#111827",
        "sidebar_border": "#1E293B",
        "sidebar_hover": "#1E293B",
        "sidebar_active_bg": "rgba(99, 102, 241, 0.18)",
        "sidebar_active_text": "#E0E7FF",
        "card_bg": "#151E31",
        "subcard_bg": "#1B263B",
        "card_border": "#24324B",
        "subcard_border": "#2B3B58",
        "card_shadow": "0 10px 30px rgba(0, 0, 0, 0.35)",
        "subcard_shadow": "0 2px 8px rgba(0, 0, 0, 0.22)",
        "input_bg": "#0F172A",
        "input_border": "#334155",
        "text_primary": "#F8FAFC",
        "text_body": "#E2E8F0",
        "text_muted": "#94A3B8",
        "text_subtle": "#CBD5E1",
        "divider": "#24324B",
        "row_divider": "#24324B",
        "pill_bg": "#1E293B",
        "pill_border": "#334155",
        "track_bg": "#26344D",
        "secondary_btn_bg": "#1E293B",
        "secondary_btn_border": "#334155",
        "secondary_btn_text": "#E2E8F0",
        "secondary_btn_hover": "#283548",
        "brand_text": "#E0E7FF",
        "primary": "#6366F1",
        "primary_hover": "#4F46E5",
        "winner_pct": "#C7D2FE",
        "alert_bg": "#151E31",
        "seg_track": "#1A2438",
        "seg_active_bg": "#6366F1",
        "seg_active_text": "#FFFFFF",
        "seg_active_shadow": "0 2px 10px rgba(99, 102, 241, 0.42), 0 1px 3px rgba(0, 0, 0, 0.3)",
        "theme_icon_active": "#FFFFFF",
    },
}

I18N: Dict[str, Dict[str, str]] = {
    "vi": {
        "app_title": "Phân loại Phản hồi Sinh viên",
        "brand_sub": "Phân tích phản hồi",
        "sec_appearance": "Giao diện & Ngôn ngữ",
        "theme_label": "Chế độ màu",
        "theme_help": "Chuyển chế độ giao diện Sáng / Tối",
        "theme_light": "Sáng",
        "theme_dark": "Tối",
        "lang_label": "Ngôn ngữ",
        "lang_help": "Chuyển ngôn ngữ Tiếng Việt / English",
        "sec_controls": "Điều khiển & Tùy chọn",
        "mode_radio_label": "Chọn chế độ dự đoán (Model Mode):",
        "mode_radio_help": "Chọn chế độ đơn lẻ (Trung bình / Cao) hoặc so sánh cả hai.",
        "mode_baseline": MODEL_MODE_BASELINE,
        "mode_phobert": MODEL_MODE_PHOBERT,
        "mode_both": MODEL_MODE_BOTH,
        "no_ckpt_err": "❌ Không tìm thấy checkpoint nào trong thư mục `saved_models/`.",
        "sec_presets": "Thư viện mẫu nhận xét",
        "preset_select_label": "Chọn câu nhận xét mẫu sinh viên:",
        "load_preset_btn": "Nạp câu mẫu",
        "status_title": "Trạng thái hệ thống",
        "status_ready": "✅ Sẵn sàng",
        "status_missing": "❌ Thiếu weights",
        "status_device": "Thiết bị",
        "header_sub_1": "Bộ dữ liệu",
        "header_sub_2": "Cảm xúc",
        "header_sub_2_val": "3 lớp",
        "header_sub_3": "Chủ đề",
        "header_sub_3_val": "4 lớp",
        "expander_title": "Thông tin bài toán & cấu trúc hai chế độ dự đoán",
        "expander_task1_title": "Tác vụ 1: Phân loại Cảm xúc (Sentiment)",
        "expander_task1_body": (
            "- **Negative (0)**: Nhận xét tiêu cực, phàn nàn, bất mãn.\n"
            "- **Neutral (1)**: Nhận xét khách quan, trung tính (5.27% dữ liệu).\n"
            "- **Positive (2)**: Nhận xét tích cực, khen ngợi, hài lòng."
        ),
        "expander_task2_title": "Tác vụ 2: Phân loại Chủ đề (Topic)",
        "expander_task2_body": (
            "- **Lecturer (0)**: Giảng viên, phương pháp dạy, tác phong (72.33%).\n"
            "- **Training Program (1)**: Khung chương trình đào tạo, môn học (18.07%).\n"
            "- **Facility (2)**: Cơ sở vật chất, phòng máy, điều hòa (4.58%).\n"
            "- **Others (3)**: Khác, thủ tục hành chính, học phí, hoạt động (5.02%)."
        ),
        "expander_models_body": (
            "**So sánh mô hình**:\n"
            "- **Trung bình (Classical Baseline)**: TF-IDF n-gram (1,2) + 2 bộ phân loại Logistic Regression cân bằng. Nhẹ (~0.95 MB), độ trễ cực thấp (~1-2 ms).\n"
            "- **Cao (PhoBERT Multi-Task v2)**: Kiến trúc Transformer RoBERTa huấn luyện trước trên 20GB tiếng Việt, chia sẻ biểu diễn `[CLS]` 768 chiều với 2 đầu tuyến tính. Hiểu ngữ cảnh sâu và câu phức."
        ),
        "phobert_missing_warn": (
            "⚠️ **Chưa tìm thấy checkpoint mô hình Cao (PhoBERT)** tại `saved_models/phobert_multitask`.\n\n"
            "Hệ thống tự động sử dụng chế độ **Trung bình**. "
            "Để kích hoạt chế độ **Cao**, vui lòng huấn luyện mô hình bằng lệnh:\n\n"
            "```bash\npython training/train_phobert.py\n```"
        ),
        "composer_title": "Nhập nhận xét của sinh viên",
        "composer_lang_tag": "Tiếng Việt",
        "textarea_label": "Nội dung phản hồi (tiếng Việt):",
        "textarea_placeholder": "Nhập nhận xét sinh viên tại đây (ví dụ: 'Thầy dạy rất tận tâm và dễ hiểu...')...",
        "textarea_help": "Có thể chỉnh sửa câu mẫu hoặc gõ nhận xét tùy ý.",
        "counter_fmt": "Độ dài: **{words}** từ · **{chars}** ký tự",
        "clear_btn": "Xóa trắng",
        "analyze_btn": "Phân tích nhận xét",
        "empty_warn": "⚠️ Vui lòng nhập nội dung nhận xét trước khi nhấn 'Phân tích'.",
        "fallback_both_warn": "⚠️ Một trong hai mô hình chưa có checkpoint. Tự động chuyển về mô hình khả dụng.",
        "err_baseline_missing": "Checkpoint mô hình Trung bình không tìm thấy tại `saved_models/baseline.joblib`.",
        "err_phobert_missing": "Checkpoint mô hình Cao không tìm thấy tại `saved_models/phobert_multitask`.",
        "spin_baseline": "Đang phân tích ở chế độ Trung bình...",
        "spin_phobert": "Đang phân tích ở chế độ Cao...",
        "spin_both": "Đang phân tích đồng thời ở cả hai chế độ...",
        "oom_warn": "⚠️ CUDA Out of Memory! Tự động chuyển suy luận mô hình Cao sang CPU.",
        "task_sentiment": "Cảm xúc (Sentiment)",
        "task_topic": "Chủ đề (Topic)",
        "dist_prefix": "Phân bố xác suất",
        "top_badge": "Chính",
        "consensus_heading": "Phân tích so sánh & Đồng thuận",
        "agree_sent": "**Đồng thuận Cảm xúc**: Cả hai đều dự đoán **{label}** (`{raw}`)",
        "diff_sent": "**Bất đồng Cảm xúc**: Trung bình = **{b}** vs Cao = **{p}**",
        "agree_topic": "**Đồng thuận Chủ đề**: Cả hai đều dự đoán **{label}** (`{raw}`)",
        "diff_topic": "**Bất đồng Chủ đề**: Trung bình = **{b}** vs Cao = **{p}**",
        "speed_title": "Chênh lệch tốc độ",
        "speed_b_faster": "**Chênh lệch tốc độ**: Trung bình nhanh hơn **{x:.1f}x** so với Cao.",
        "speed_p_faster": "**Chênh lệch tốc độ**: Cao nhanh hơn **{x:.1f}x** so với Trung bình.",
        "donut_section": "Sơ đồ tròn phân bố xác suất",
        "donut_clear": "Rõ nét",
        "donut_moderate": "Khá rõ",
        "donut_balanced": "Phân tán",
        "en_gloss_prefix": "Bản dịch tiếng Anh mẫu",
    },
    "en": {
        "app_title": "Text Classification of Vietnamese Student Feedback",
        "brand_sub": "Feedback Analyzer",
        "sec_appearance": "Appearance & Language",
        "theme_label": "Theme",
        "theme_help": "Switch between Light and Dark theme",
        "theme_light": "Light",
        "theme_dark": "Dark",
        "lang_label": "Language",
        "lang_help": "Switch language: Tiếng Việt / English",
        "sec_controls": "Controls & Options",
        "mode_radio_label": "Select prediction mode (Model Mode):",
        "mode_radio_help": "Choose a single tier (Medium / High) or compare both side by side.",
        "mode_baseline": "Medium",
        "mode_phobert": "High",
        "mode_both": "Both",
        "no_ckpt_err": "❌ No model checkpoints found in `saved_models/`.",
        "sec_presets": "Feedback Presets",
        "preset_select_label": "Choose a sample student feedback:",
        "load_preset_btn": "Load Preset",
        "status_title": "System Status",
        "status_ready": "✅ Ready",
        "status_missing": "❌ Missing weights",
        "status_device": "Device",
        "header_sub_1": "Dataset",
        "header_sub_2": "Sentiment",
        "header_sub_2_val": "3 classes",
        "header_sub_3": "Topic",
        "header_sub_3_val": "4 classes",
        "expander_title": "Task overview & model tier architecture",
        "expander_task1_title": "Task 1: Sentiment Classification",
        "expander_task1_body": (
            "- **Negative (0)**: Dissatisfaction, complaints, or critical remarks.\n"
            "- **Neutral (1)**: Objective, balanced, or factual remarks (5.27% of corpus).\n"
            "- **Positive (2)**: Praise, satisfaction, and positive comments."
        ),
        "expander_task2_title": "Task 2: Topic Classification",
        "expander_task2_body": (
            "- **Lecturer (0)**: Teaching style, clarity, dedication, punctuality (72.33%).\n"
            "- **Training Program (1)**: Curriculum, syllabus, exams, workload (18.07%).\n"
            "- **Facility (2)**: Classrooms, labs, projectors, AC, Wi-Fi (4.58%).\n"
            "- **Others (3)**: Tuition, administration, student services (5.02%)."
        ),
        "expander_models_body": (
            "**Model Tiers**:\n"
            "- **Medium (Classical Baseline)**: TF-IDF n-gram (1,2) + dual balanced Logistic Regression heads (~0.95 MB, ~1-2 ms latency).\n"
            "- **High (PhoBERT Multi-Task v2)**: Pre-trained Vietnamese RoBERTa encoder sharing a 768-d `[CLS]` vector across dual linear heads for deeper contextual understanding."
        ),
        "phobert_missing_warn": (
            "⚠️ **High-tier model checkpoint (PhoBERT) not found** at `saved_models/phobert_multitask`.\n\n"
            "Automatically running in **Medium** mode. "
            "To enable **High** mode, please fine-tune PhoBERT using:\n\n"
            "```bash\npython training/train_phobert.py\n```"
        ),
        "composer_title": "Student Feedback Input",
        "composer_lang_tag": "Vietnamese text",
        "textarea_label": "Feedback text (Vietnamese):",
        "textarea_placeholder": "Enter Vietnamese student feedback here (e.g. 'Thầy dạy rất tận tâm và dễ hiểu...')...",
        "textarea_help": "Edit a preset or type custom Vietnamese feedback.",
        "counter_fmt": "Length: **{words}** words · **{chars}** characters",
        "clear_btn": "Clear",
        "analyze_btn": "Analyze Feedback",
        "empty_warn": "⚠️ Please enter feedback text before clicking 'Analyze Feedback'.",
        "fallback_both_warn": "⚠️ One model checkpoint is missing. Switching to the available mode.",
        "err_baseline_missing": "Medium model checkpoint not found at `saved_models/baseline.joblib`.",
        "err_phobert_missing": "High model checkpoint not found at `saved_models/phobert_multitask`.",
        "spin_baseline": "Analyzing with Medium mode...",
        "spin_phobert": "Analyzing with High mode...",
        "spin_both": "Analyzing with both modes simultaneously...",
        "oom_warn": "⚠️ CUDA Out of Memory! Automatically failing over High model inference to CPU.",
        "task_sentiment": "Sentiment",
        "task_topic": "Topic",
        "dist_prefix": "Probability Distribution:",
        "top_badge": "Top",
        "consensus_heading": "Comparison & Consensus Analysis",
        "agree_sent": "**Sentiment Agreement**: Both predicted **{label}** (`{raw}`)",
        "diff_sent": "**Sentiment Divergence**: Medium = **{b}** vs High = **{p}**",
        "agree_topic": "**Topic Agreement**: Both predicted **{label}** (`{raw}`)",
        "diff_topic": "**Topic Divergence**: Medium = **{b}** vs High = **{p}**",
        "speed_title": "Latency Difference",
        "speed_b_faster": "**Latency Difference**: Medium is **{x:.1f}x** faster than High.",
        "speed_p_faster": "**Latency Difference**: High is **{x:.1f}x** faster than Medium.",
        "donut_section": "Probability Breakdown (Donut)",
        "donut_clear": "Clear lead",
        "donut_moderate": "Moderate",
        "donut_balanced": "Balanced",
        "en_gloss_prefix": "English meaning of selected Vietnamese sample",
    },
}

SENTIMENT_UI_META: Dict[str, Dict[str, str]] = {
    "positive": {
        "vn_name": "Tích cực (Positive)",
        "en_name": "Positive",
        "emoji": "😊",
        "color": "#10B981",
        "bg_color": "#D1FAE5",
        "border_color": "#34D399",
        "dark_color": "#34D399",
        "dark_bg_color": "rgba(16, 185, 129, 0.16)",
        "dark_border_color": "rgba(52, 211, 153, 0.35)",
    },
    "neutral": {
        "vn_name": "Trung tính (Neutral)",
        "en_name": "Neutral",
        "emoji": "😐",
        "color": "#F59E0B",
        "bg_color": "#FEF3C7",
        "border_color": "#FBBF24",
        "dark_color": "#FBBF24",
        "dark_bg_color": "rgba(245, 158, 11, 0.16)",
        "dark_border_color": "rgba(251, 191, 36, 0.35)",
    },
    "negative": {
        "vn_name": "Tiêu cực (Negative)",
        "en_name": "Negative",
        "emoji": "😞",
        "color": "#EF4444",
        "bg_color": "#FEE2E2",
        "border_color": "#F87171",
        "dark_color": "#F87171",
        "dark_bg_color": "rgba(239, 68, 68, 0.16)",
        "dark_border_color": "rgba(248, 113, 113, 0.35)",
    },
}

TOPIC_UI_META: Dict[str, Dict[str, str]] = {
    "lecturer": {
        "vn_name": "Giảng viên (Lecturer)",
        "en_name": "Lecturer",
        "icon": "👨‍🏫",
        "color": "#3B82F6",
        "bg_color": "#DBEAFE",
        "border_color": "#60A5FA",
        "dark_color": "#60A5FA",
        "dark_bg_color": "rgba(59, 130, 246, 0.16)",
        "dark_border_color": "rgba(96, 165, 250, 0.35)",
    },
    "training_program": {
        "vn_name": "Chương trình đào tạo (Training Program)",
        "en_name": "Training Program",
        "icon": "📚",
        "color": "#8B5CF6",
        "bg_color": "#EDE9FE",
        "border_color": "#A78BFA",
        "dark_color": "#A78BFA",
        "dark_bg_color": "rgba(139, 92, 246, 0.18)",
        "dark_border_color": "rgba(167, 139, 250, 0.35)",
    },
    "facility": {
        "vn_name": "Cơ sở vật chất (Facility)",
        "en_name": "Facility",
        "icon": "🏢",
        "color": "#EC4899",
        "bg_color": "#FCE7F3",
        "border_color": "#F472B6",
        "dark_color": "#F472B6",
        "dark_bg_color": "rgba(236, 72, 153, 0.16)",
        "dark_border_color": "rgba(244, 114, 182, 0.35)",
    },
    "others": {
        "vn_name": "Khác (Others)",
        "en_name": "Others",
        "icon": "📌",
        "color": "#6B7280",
        "bg_color": "#F3F4F6",
        "border_color": "#9CA3AF",
        "dark_color": "#94A3B8",
        "dark_bg_color": "rgba(148, 163, 184, 0.16)",
        "dark_border_color": "rgba(148, 163, 184, 0.35)",
    },
}

# ---------------------------------------------------------------------------
# Preset Examples Library (10 Diverse & Authentic Vietnamese Presets)
# ---------------------------------------------------------------------------
def get_presets() -> List[Dict[str, str]]:
    """
    Returns curated authentic Vietnamese student feedback presets
    covering all combinations of 3 sentiments and 4 topics, plus linguistic nuance.
    """
    return [
        {
            "id": "pos_lec",
            "title": "1. [Tích cực - Giảng viên] Thầy dạy nhiệt tình, tận tâm",
            "title_en": "1. [Positive - Lecturer] Enthusiastic and clear teaching",
            "text": "Thầy dạy rất nhiệt tình, giảng giải cặn kẽ và luôn tạo không khí học tập thoải mái cho sinh viên.",
            "text_en": "The lecturer teaches enthusiastically, explains concepts thoroughly, and keeps a comfortable classroom atmosphere.",
            "expected_sentiment": "positive",
            "expected_topic": "lecturer",
            "note": "Từ khóa rõ ràng: nhiệt tình, giảng giải, sinh viên",
        },
        {
            "id": "neg_fac",
            "title": "2. [Tiêu cực - Cơ sở vật chất] Phòng máy hư, mạng wifi yếu",
            "title_en": "2. [Negative - Facility] Broken lab PCs and weak Wi-Fi",
            "text": "Phòng máy tính hư hỏng nhiều, máy lạnh bị rò rỉ nước và mạng wifi trường quá chập chờn.",
            "text_en": "Many lab computers are broken, the air conditioner leaks water, and campus Wi-Fi is very unstable.",
            "expected_sentiment": "negative",
            "expected_topic": "facility",
            "note": "Phản ánh cơ sở vật chất phần cứng: phòng máy, máy lạnh, wifi",
        },
        {
            "id": "neu_prog",
            "title": "3. [Trung tính - Chương trình đào tạo] Môn học vừa phải",
            "title_en": "3. [Neutral - Training Program] Moderate course workload",
            "text": "Chương trình đào tạo nhìn chung tạm ổn, nội dung các môn cơ sở ngành ở mức độ vừa phải.",
            "text_en": "The curriculum is generally fine, with core major subjects kept at a moderate difficulty.",
            "expected_sentiment": "neutral",
            "expected_topic": "training_program",
            "note": "Nhận xét khách quan cân bằng: tạm ổn, vừa phải",
        },
        {
            "id": "neg_lec",
            "title": "4. [Tiêu cực - Giảng viên] Giảng bài khô khan, nói nhanh",
            "title_en": "4. [Negative - Lecturer] Dry lectures, talks too fast",
            "text": "Giáo viên giảng bài khô khan, nói quá nhanh và không hướng dẫn sinh viên làm bài tập thực hành.",
            "text_en": "The instructor delivers dry lectures, speaks too fast, and gives little guidance on practical exercises.",
            "expected_sentiment": "negative",
            "expected_topic": "lecturer",
            "note": "Phản ánh phương pháp sư phạm của giảng viên",
        },
        {
            "id": "pos_fac",
            "title": "5. [Tích cực - Cơ sở vật chất] Phòng học máy lạnh mát mẻ",
            "title_en": "5. [Positive - Facility] Cool classroom and great projector",
            "text": "Phòng học thoáng mát, máy lạnh chạy êm và trang thiết bị máy chiếu phục vụ thuyết trình rất tốt.",
            "text_en": "The classroom is airy and cool, the AC runs quietly, and the projector works great for presentations.",
            "expected_sentiment": "positive",
            "expected_topic": "facility",
            "note": "Khen ngợi điều kiện tiện nghi giảng đường",
        },
        {
            "id": "neg_prog",
            "title": "6. [Tiêu cực - Chương trình đào tạo] Lịch học và thi dày đặc",
            "title_en": "6. [Negative - Training Program] Packed class & exam schedule",
            "text": "Lịch học và lịch thi bố trí quá dày đặc, sinh viên không kịp tiếp thu kiến thức và làm đồ án.",
            "text_en": "Classes and exams are scheduled too tightly, leaving students insufficient time to absorb lessons and finish projects.",
            "expected_sentiment": "negative",
            "expected_topic": "training_program",
            "note": "Phàn nàn về cấu trúc chương trình và tiến độ thi",
        },
        {
            "id": "pos_prog",
            "title": "7. [Tích cực - Chương trình đào tạo] Bài tập thực hành thực tế",
            "title_en": "7. [Positive - Training Program] Practical assignments",
            "text": "Môn học có tính ứng dụng thực tế rất cao, hệ thống bài tập lớn giúp sinh viên nâng cao kỹ năng nghề nghiệp.",
            "text_en": "The course is highly practical, and the major assignments significantly strengthen students' career skills.",
            "expected_sentiment": "positive",
            "expected_topic": "training_program",
            "note": "Đánh giá cao nội dung chuyên môn và kỹ năng thực hành",
        },
        {
            "id": "neg_oth",
            "title": "8. [Tiêu cực - Khác] Học phí tăng, thủ tục hành chính chậm",
            "title_en": "8. [Negative - Others] Rising tuition & slow admin paperwork",
            "text": "Học phí tăng hàng năm nhưng thủ tục hành chính tại phòng công tác sinh viên giải quyết rất chậm.",
            "text_en": "Tuition rises annually, yet administrative paperwork at the student affairs office is processed very slowly.",
            "expected_sentiment": "negative",
            "expected_topic": "others",
            "note": "Chủ đề hành chính và tài chính ngoài học thuật",
        },
        {
            "id": "neu_lec",
            "title": "9. [Trung tính - Giảng viên] Dạy đúng slide và giờ giấc",
            "title_en": "9. [Neutral - Lecturer] Follows slides and stays on schedule",
            "text": "Thầy dạy theo đúng giáo trình trong slide, thời gian lên lớp đúng giờ quy định.",
            "text_en": "The instructor follows the syllabus slides faithfully and always starts class on time.",
            "expected_sentiment": "neutral",
            "expected_topic": "lecturer",
            "note": "Miêu tả khách quan tác phong giảng viên, không khen chê mạnh",
        },
        {
            "id": "sarcasm",
            "title": "10. [Ngữ cảnh đặc biệt - Mỉa mai] Phòng học xông hơi",
            "title_en": "10. [Special - Sarcasm] Sauna classroom irony",
            "text": "Cảm ơn nhà trường đã tạo cơ hội cho chúng em trải nghiệm phòng học xông hơi giữa trưa nắng.",
            "text_en": "Thanks to the university for letting us experience a sauna classroom in the midday heat.",
            "expected_sentiment": "negative",
            "expected_topic": "facility",
            "note": "Mỉa mai: dùng từ cảm ơn bề mặt nhưng hàm ý chê cơ sở vật chất nóng",
        },
    ]

# ---------------------------------------------------------------------------
# Model Caching Helpers (@st.cache_resource)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_baseline_model(checkpoint_path: Optional[str] = None) -> Optional[VietnameseFeedbackBaseline]:
    """Loads and caches the Classical Baseline model."""
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_BASELINE_PATH
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(PROJECT_ROOT, checkpoint_path)

    if not os.path.isfile(checkpoint_path):
        return None
    try:
        return VietnameseFeedbackBaseline.load(checkpoint_path)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_phobert_pipeline(
    checkpoint_dir: Optional[str] = None
) -> Tuple[Optional[PhoBERTMultiTask], Any, str]:
    """
    Loads and caches PhoBERT Multi-Task model and offline AutoTokenizer.
    Returns (model, tokenizer, device_str).
    """
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_PHOBERT_DIR
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.join(PROJECT_ROOT, checkpoint_dir)

    if not os.path.isdir(checkpoint_dir):
        return None, None, "cpu"

    weights_exist = any(
        os.path.isfile(os.path.join(checkpoint_dir, f))
        for f in ("model.pt", "pytorch_model.bin")
    )
    if not weights_exist:
        return None, None, "cpu"

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=False)

        try:
            model = PhoBERTMultiTask.from_pretrained(checkpoint_dir, device=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            if device_str == "cuda":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                device_str = "cpu"
                device = torch.device(device_str)
                model = PhoBERTMultiTask.from_pretrained(checkpoint_dir, device=device)
            else:
                return None, None, "cpu"

        if hasattr(model, "eval"):
            model.eval()
        return model, tokenizer, device_str
    except Exception:
        return None, None, device_str

# ---------------------------------------------------------------------------
# UI Rendering Helpers
# ---------------------------------------------------------------------------
def render_html(html: str) -> None:
    """Renders compact single-line HTML so Streamlit Markdown never parses indented <pre><code> blocks."""
    compact = " ".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(compact, unsafe_allow_html=True)


def make_card_container(key: str):
    """Creates a bordered Streamlit container with a stable CSS key when supported."""
    try:
        return st.container(border=True, key=key)
    except TypeError:
        return st.container(border=True)


def _get_lang() -> str:
    """Returns active UI language code ('vi' or 'en')."""
    lang = st.session_state.get("ui_lang", "vi")
    return lang if lang in I18N else "vi"


def _get_theme() -> str:
    """Returns active UI theme mode ('light' or 'dark')."""
    theme = st.session_state.get("theme_mode", "light")
    return theme if theme in THEME_PALETTES else "light"


def _resolve_meta_colors(meta: Dict[str, str], theme_mode: str) -> Tuple[str, str, str]:
    """Selects light or dark semantic accent, chip background, and border colors."""
    if theme_mode == "dark":
        return (
            meta.get("dark_color", meta.get("color", "#818CF8")),
            meta.get("dark_bg_color", "rgba(99, 102, 241, 0.16)"),
            meta.get("dark_border_color", "rgba(129, 140, 248, 0.35)"),
        )
    return (
        meta.get("color", "#4F46E5"),
        meta.get("bg_color", "#EEF2FF"),
        meta.get("border_color", "#C7D2FE"),
    )


def _get_label_display_name(meta: Dict[str, str], fallback: str, lang: str) -> str:
    """Returns localized class label name."""
    if lang == "en":
        return meta.get("en_name", fallback)
    return meta.get("vn_name", fallback).split("(")[0].strip()


def inject_custom_css(theme_mode: str = "light") -> None:
    """Injects light or dark SaaS board styling with icon toggles and smooth mode transitions."""
    p = THEME_PALETTES.get(theme_mode, THEME_PALETTES["light"])
    vn_flag_uri = (
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
        "%3Ccircle cx='12' cy='12' r='11.5' fill='%23DA251D'/%3E"
        "%3Cpath fill='%23FFCD00' d='M12 5.8L13.39 10.08L17.9 10.08L14.25 12.73L15.64 17.02L12 14.37L8.36 17.02L9.75 12.73L6.1 10.08L10.61 10.08Z'/%3E"
        "%3C/svg%3E"
    )
    en_flag_uri = (
        "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
        "%3Cdefs%3E%3CclipPath id='ukc'%3E%3Ccircle cx='12' cy='12' r='11.5'/%3E%3C/clipPath%3E%3C/defs%3E"
        "%3Cg clip-path='url(%23ukc)'%3E"
        "%3Crect width='24' height='24' fill='%23012169'/%3E"
        "%3Cpath d='M0 0L24 24M24 0L0 24' stroke='%23FFFFFF' stroke-width='4.6'/%3E"
        "%3Cpath d='M0 0L24 24M24 0L0 24' stroke='%23C8102E' stroke-width='2.2'/%3E"
        "%3Cpath d='M12 0V24M0 12H24' stroke='%23FFFFFF' stroke-width='6.8'/%3E"
        "%3Cpath d='M12 0V24M0 12H24' stroke='%23C8102E' stroke-width='3.8'/%3E"
        "%3C/g%3E%3C/svg%3E"
    )
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    @keyframes boardSlideIn {{
        from {{ opacity: 0; transform: translateY(10px) scale(0.992); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}

    @keyframes iconSelectPop {{
        0% {{ transform: scale(0.82); }}
        55% {{ transform: scale(1.12); }}
        100% {{ transform: scale(1); }}
    }}

    @keyframes barGrow {{
        from {{ transform: scaleX(0); }}
        to {{ transform: scaleX(1); }}
    }}

    @keyframes modeBadgeSlide {{
        from {{ opacity: 0; transform: translateY(-4px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes donutReveal {{
        from {{ opacity: 0; transform: scale(0.86) rotate(-14deg); }}
        to {{ opacity: 1; transform: scale(1) rotate(0deg); }}
    }}

    .stApp {{
        background-color: {p['app_bg']};
        color: {p['text_body']};
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        transition: background-color 0.36s cubic-bezier(0.22, 1, 0.36, 1), color 0.26s ease;
    }}

    .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp textarea, .stApp input, .stApp button, [data-testid="stMarkdownContainer"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: {p['text_body']};
        transition: color 0.24s ease;
    }}

    [data-testid="stDecoration"] {{
        display: none;
    }}

    header[data-testid="stHeader"] {{
        background: {p['header_bg']};
        backdrop-filter: blur(8px);
        transition: background-color 0.36s cubic-bezier(0.22, 1, 0.36, 1);
    }}

    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] svg {{
        color: {p['text_muted']} !important;
        fill: currentColor;
    }}

    .main .block-container {{
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }}

    .stApp code {{
        background-color: {p['pill_bg']} !important;
        color: {p['text_primary']} !important;
        border: 1px solid {p['pill_border']};
        border-radius: 5px;
        padding: 1px 6px;
        transition: background-color 0.28s ease, border-color 0.28s ease, color 0.24s ease;
    }}

    [data-testid="stCaptionContainer"] p {{
        color: {p['text_muted']} !important;
    }}

    /* Sidebar styled like a clean SaaS navigation rail */
    section[data-testid="stSidebar"] {{
        background-color: {p['sidebar_bg']};
        border-right: 1px solid {p['sidebar_border']};
        transition: background-color 0.36s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.3s ease;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.25rem;
    }}

    section[data-testid="stSidebar"] [data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] button svg {{
        color: {p['text_muted']} !important;
    }}

    /* Compact side-by-side icon capsules (Light/Dark & VI/EN) with sliding pill thumb */
    section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.st-key-seg_theme_mode) {{
        margin-bottom: 0.25rem;
    }}

    .st-key-seg_theme_mode [data-testid="stButtonGroup"] > div[role="radiogroup"],
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] > div[role="radiogroup"] {{
        position: relative !important;
        display: flex !important;
        flex-wrap: nowrap !important;
        width: 100% !important;
        background-color: {p['seg_track']} !important;
        border-radius: 11px !important;
        padding: 3px !important;
        gap: 4px !important;
        border: 1px solid {p['sidebar_border']} !important;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.06) !important;
        box-sizing: border-box !important;
        transition: background-color 0.32s ease, border-color 0.32s ease !important;
    }}

    .st-key-seg_theme_mode [data-testid="stButtonGroup"] > div[role="radiogroup"]::before,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] > div[role="radiogroup"]::before {{
        content: "" !important;
        position: absolute !important;
        top: 3px !important;
        bottom: 3px !important;
        left: 3px !important;
        width: calc(50% - 5px) !important;
        border-radius: 8px !important;
        background-color: {p['seg_active_bg']} !important;
        box-shadow: {p['seg_active_shadow']} !important;
        transition: transform 0.34s cubic-bezier(0.34, 1.35, 0.64, 1),
                    background-color 0.28s ease,
                    box-shadow 0.28s ease !important;
        pointer-events: none !important;
        z-index: 0 !important;
    }}

    .st-key-seg_theme_mode [data-testid="stButtonGroup"] > div[role="radiogroup"]:has(> button:nth-of-type(2)[aria-checked="true"])::before,
    .st-key-seg_theme_mode [data-testid="stButtonGroup"] > div[role="radiogroup"]:has(> button:nth-of-type(2)[data-selected])::before,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] > div[role="radiogroup"]:has(> button:nth-of-type(2)[aria-checked="true"])::before,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] > div[role="radiogroup"]:has(> button:nth-of-type(2)[data-selected])::before {{
        transform: translateX(calc(100% + 4px)) !important;
    }}

    .st-key-seg_theme_mode [data-testid="stButtonGroup"] button[role="radio"],
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"] {{
        position: relative !important;
        z-index: 1 !important;
        flex: 1 1 0% !important;
        width: auto !important;
        min-width: 0 !important;
        border: none !important;
        border-radius: 8px !important;
        background-color: transparent !important;
        box-shadow: none !important;
        padding: 0 6px !important;
        min-height: 32px !important;
        height: 32px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 5px !important;
        color: {p['text_subtle']} !important;
        transition: color 0.24s ease, transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }}

    .st-key-seg_theme_mode [data-testid="stButtonGroup"] button[role="radio"]:hover,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"]:hover {{
        background-color: transparent !important;
        color: {p['text_primary']} !important;
    }}

    .st-key-seg_theme_mode [data-testid="stButtonGroup"] button[role="radio"]:active,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"]:active {{
        transform: scale(0.95) !important;
    }}

    section[data-testid="stSidebar"] .st-key-seg_theme_mode [data-testid="stIconMaterial"] {{
        color: {p['text_subtle']} !important;
        font-size: 1.08rem !important;
        transition: transform 0.28s cubic-bezier(0.34, 1.56, 0.64, 1), color 0.24s ease !important;
    }}

    section[data-testid="stSidebar"] .st-key-seg_theme_mode button[role="radio"]:nth-of-type(1):hover [data-testid="stIconMaterial"] {{
        transform: rotate(28deg) scale(1.12);
        color: {p['theme_icon_active']} !important;
    }}

    section[data-testid="stSidebar"] .st-key-seg_theme_mode button[role="radio"]:nth-of-type(2):hover [data-testid="stIconMaterial"] {{
        transform: rotate(-16deg) scale(1.12);
        color: {p['text_primary']} !important;
    }}

    section[data-testid="stSidebar"] .st-key-seg_theme_mode button[role="radio"][aria-checked="true"] [data-testid="stIconMaterial"],
    section[data-testid="stSidebar"] .st-key-seg_theme_mode button[role="radio"][data-selected] [data-testid="stIconMaterial"] {{
        color: {p['theme_icon_active']} !important;
        animation: iconSelectPop 0.34s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}

    /* Language SVG Circular Flags + Micro Badge */
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"]::before {{
        content: "";
        display: inline-block;
        width: 15px;
        height: 15px;
        flex-shrink: 0;
        border-radius: 999px;
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.14);
        transition: transform 0.24s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"]:nth-of-type(1)::before {{
        background-image: url("{vn_flag_uri}");
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"]:nth-of-type(2)::before {{
        background-image: url("{en_flag_uri}");
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"]:hover::before {{
        transform: scale(1.12);
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"]::before,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"][data-selected]::before {{
        animation: iconSelectPop 0.34s cubic-bezier(0.34, 1.56, 0.64, 1);
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"] [data-testid="stMarkdownContainer"] {{
        width: auto !important;
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"] p,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"] span {{
        color: {p['text_subtle']} !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        line-height: 1 !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }}

    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] p,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] span,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"][data-selected] p,
    .st-key-seg_ui_lang [data-testid="stButtonGroup"] button[role="radio"][data-selected] span {{
        color: {p['seg_active_text']} !important;
    }}

    /* Model mode sidebar radio navigation with smooth slide-hover & active accent */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {{
        gap: 0.32rem;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {{
        background: transparent;
        border-radius: 9px;
        padding: 0.54rem 0.75rem;
        margin: 0;
        border-left: 3px solid transparent;
        transition: background-color 0.24s ease,
                    border-color 0.24s ease,
                    box-shadow 0.24s ease,
                    transform 0.22s cubic-bezier(0.22, 1, 0.36, 1) !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
        background-color: {p['sidebar_hover']};
        transform: translateX(3px);
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) > div:first-of-type,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) > div:first-of-type > div,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) > div:first-of-type > div > div {{
        background-color: transparent !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:not(:has(input:checked)) > div:first-of-type > div {{
        border-color: {p['text_muted']} !important;
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: {p['sidebar_active_bg']};
        border-left: 3px solid {p['primary']};
        box-shadow: {p['subcard_shadow']};
        transform: translateX(2px);
    }}

    section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {{
        color: {p['sidebar_active_text']} !important;
        font-weight: 600;
    }}

    /* Selectbox & Virtual Dropdown */
    div[data-testid="stSelectbox"] [data-baseweb="select"],
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] .react-aria-ComboBox > div,
    div[data-testid="stSelectbox"] div[role="group"],
    div[data-testid="stSelectbox"] div:has(> input[role="combobox"]) {{
        background-color: {p['input_bg']} !important;
        color: {p['text_primary']} !important;
        transition: background-color 0.3s ease, border-color 0.24s ease, color 0.24s ease !important;
    }}

    div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[role="group"] {{
        border: 1px solid {p['input_border']} !important;
        border-radius: 9px !important;
    }}

    div[data-testid="stSelectbox"] input[role="combobox"],
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] svg {{
        color: {p['text_primary']} !important;
        -webkit-text-fill-color: {p['text_primary']} !important;
        fill: currentColor !important;
    }}

    div[data-testid="stSelectbox"] label[data-testid="stWidgetLabel"] p {{
        color: {p['text_body']} !important;
        -webkit-text-fill-color: {p['text_body']} !important;
    }}

    div[data-testid="stSelectboxVirtualDropdown"],
    div[data-testid="stSelectboxVirtualDropdown"] > div,
    div[data-testid="stSelectboxVirtualDropdown"] [role="listbox"],
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] [role="listbox"] {{
        background-color: {p['card_bg']} !important;
        color: {p['text_body']} !important;
    }}

    div[data-testid="stSelectboxVirtualDropdown"],
    div[data-baseweb="popover"] > div {{
        border: 1px solid {p['card_border']} !important;
        border-radius: 10px !important;
        box-shadow: {p['card_shadow']} !important;
        overflow: hidden !important;
    }}

    div[data-testid="stSelectboxVirtualDropdown"] [role="option"],
    div[data-testid="stSelectboxVirtualDropdown"] li,
    div[data-baseweb="popover"] [role="option"],
    div[data-baseweb="popover"] li {{
        background-color: transparent !important;
        color: {p['text_body']} !important;
        transition: background-color 0.16s ease, color 0.16s ease !important;
    }}

    div[data-testid="stSelectboxVirtualDropdown"] [role="option"] *,
    div[data-testid="stSelectboxVirtualDropdown"] li *,
    div[data-baseweb="popover"] [role="option"] *,
    div[data-baseweb="popover"] li * {{
        color: {p['text_body']} !important;
    }}

    div[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover,
    div[data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"],
    div[data-testid="stSelectboxVirtualDropdown"] li:hover,
    div[data-baseweb="popover"] [role="option"]:hover,
    div[data-baseweb="popover"] [aria-selected="true"],
    div[data-baseweb="popover"] li:hover {{
        background-color: {p['sidebar_hover']} !important;
        color: {p['text_primary']} !important;
    }}

    /* Elevated board cards + mode transition glide */
    .st-key-composer_card,
    .st-key-board_card_baseline,
    .st-key-board_card_phobert,
    .st-key-board_card_single,
    .st-key-board_card_single_baseline,
    .st-key-board_card_single_phobert {{
        background-color: {p['card_bg']};
        border: 1px solid {p['card_border']} !important;
        border-radius: 16px !important;
        box-shadow: {p['card_shadow']};
        padding: 1.25rem 1.35rem !important;
        transition: background-color 0.35s cubic-bezier(0.22, 1, 0.36, 1),
                    border-color 0.28s ease,
                    box-shadow 0.28s ease,
                    transform 0.25s cubic-bezier(0.22, 1, 0.36, 1);
    }}

    .st-key-board_card_baseline,
    .st-key-board_card_single,
    .st-key-board_card_single_baseline,
    .st-key-board_card_single_phobert {{
        animation: boardSlideIn 0.34s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}

    .st-key-board_card_phobert {{
        animation: boardSlideIn 0.36s cubic-bezier(0.22, 1, 0.36, 1) 0.065s both;
    }}

    .st-key-board_card_baseline:hover,
    .st-key-board_card_phobert:hover,
    .st-key-board_card_single:hover,
    .st-key-board_card_single_baseline:hover,
    .st-key-board_card_single_phobert:hover {{
        transform: translateY(-2px);
    }}

    .vsfc-mode-pill {{
        animation: modeBadgeSlide 0.28s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: background-color 0.3s ease, border-color 0.3s ease, color 0.24s ease;
    }}

    .vsfc-conf-fill {{
        transform-origin: left center;
        animation: barGrow 0.46s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}

    .vsfc-donut-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(215px, 1fr));
        gap: 10px;
        margin-bottom: 12px;
    }}

    .vsfc-donut-ring {{
        animation: donutReveal 0.42s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}

    /* Text area */
    .stTextArea textarea {{
        background-color: {p['input_bg']} !important;
        border: 1px solid {p['input_border']} !important;
        border-radius: 10px !important;
        color: {p['text_primary']} !important;
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
        padding: 0.85rem 1rem !important;
        box-shadow: none !important;
        transition: background-color 0.3s ease, border-color 0.25s ease, color 0.24s ease, box-shadow 0.25s ease !important;
    }}

    .stTextArea textarea::placeholder {{
        color: {p['text_muted']} !important;
    }}

    .stTextArea textarea:focus {{
        border-color: {p['primary']} !important;
        background-color: {p['card_bg']} !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.18) !important;
    }}

    /* Buttons */
    .stButton button[kind="primary"] {{
        background-color: {p['primary']} !important;
        border: 1px solid {p['primary']} !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.1rem !important;
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.25) !important;
        transition: background-color 0.24s ease, border-color 0.24s ease, transform 0.18s ease, box-shadow 0.24s ease !important;
    }}

    .stButton button[kind="primary"] p {{
        color: #FFFFFF !important;
    }}

    .stButton button[kind="primary"]:hover {{
        background-color: {p['primary_hover']} !important;
        border-color: {p['primary_hover']} !important;
        transform: translateY(-1px);
    }}

    .stButton button[kind="secondary"] {{
        background-color: {p['secondary_btn_bg']} !important;
        border: 1px solid {p['secondary_btn_border']} !important;
        color: {p['secondary_btn_text']} !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: background-color 0.24s ease, border-color 0.24s ease, color 0.24s ease, transform 0.18s ease !important;
    }}

    .stButton button[kind="secondary"] p {{
        color: {p['secondary_btn_text']} !important;
    }}

    .stButton button[kind="secondary"]:hover {{
        background-color: {p['secondary_btn_hover']} !important;
        border-color: {p['primary']} !important;
        transform: translateY(-1px);
    }}

    /* Sleek thin progress bars */
    div[data-testid="stProgress"] {{
        margin-top: -0.25rem;
        margin-bottom: 0.55rem;
    }}

    div[data-testid="stProgress"] > div > div {{
        background-color: {p['track_bg']} !important;
        height: 5px !important;
        border-radius: 999px !important;
        transition: background-color 0.3s ease !important;
    }}

    div[data-testid="stProgress"] > div > div > div > div {{
        background-color: {p['primary']} !important;
        border-radius: 999px !important;
        transform-origin: left center;
        animation: barGrow 0.48s cubic-bezier(0.22, 1, 0.36, 1) both;
        transition: background-color 0.3s ease !important;
    }}

    /* Refined alerts */
    div[data-testid="stAlert"],
    div[data-testid="stAlert"] > div,
    div[data-testid="stAlertContainer"],
    div[data-testid="stAlert"] [role="alert"] {{
        background-color: {p['alert_bg']} !important;
        color: {p['text_body']} !important;
        transition: background-color 0.34s ease, color 0.24s ease !important;
    }}

    div[data-testid="stAlert"] > div,
    div[data-testid="stAlertContainer"] {{
        border: none !important;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 12px !important;
        border: 1px solid {p['card_border']} !important;
        box-shadow: {p['subcard_shadow']};
        animation: boardSlideIn 0.34s cubic-bezier(0.22, 1, 0.36, 1) 0.1s both;
    }}

    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] strong {{
        color: {p['text_body']} !important;
    }}

    /* Clean expander */
    div[data-testid="stExpander"] details {{
        background-color: {p['card_bg']} !important;
        border: 1px solid {p['card_border']} !important;
        border-radius: 12px !important;
        overflow: hidden;
        transition: background-color 0.34s ease, border-color 0.3s ease !important;
    }}

    div[data-testid="stExpander"] details summary,
    div[data-testid="stExpander"] details[open] > summary,
    div[data-testid="stExpander"] details summary:focus {{
        background-color: {p['card_bg']} !important;
        color: {p['text_primary']} !important;
        transition: background-color 0.25s ease, color 0.24s ease !important;
    }}

    div[data-testid="stExpander"] details summary:hover {{
        background-color: {p['sidebar_hover']} !important;
        color: {p['text_primary']} !important;
    }}

    div[data-testid="stExpanderDetails"] {{
        background-color: {p['card_bg']} !important;
        transition: background-color 0.34s ease !important;
    }}

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary svg {{
        color: {p['text_primary']} !important;
        fill: currentColor;
    }}

    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    </style>
    """
    render_html(css)


def render_header(lang: str = "vi", theme_mode: str = "light", mode_key: Optional[str] = "both"):
    """Renders a crisp left-aligned SaaS workspace header with active mode indicator pill."""
    p = THEME_PALETTES[theme_mode]
    t = I18N[lang]
    mode_badge_html = ""
    if mode_key in MODEL_RING_COLORS:
        mode_icons = {"baseline": "⚡", "phobert": "🧠", "both": "⚖️"}
        mode_names = {
            "baseline": t["mode_baseline"],
            "phobert": t["mode_phobert"],
            "both": t["mode_both"],
        }
        ring_color = MODEL_RING_COLORS[mode_key]
        theme_icon = "🌙" if theme_mode == "dark" else "☀️"
        lang_code = "EN" if lang == "en" else "VI"
        mode_badge_html = (
            f"<div class='vsfc-mode-pill' style='display:inline-flex; align-items:center; gap:8px; "
            f"background:{p['card_bg']}; border:1px solid {p['card_border']}; border-radius:999px; "
            f"padding:5px 12px; font-size:0.78rem; font-weight:600; color:{p['text_subtle']}; box-shadow:{p['subcard_shadow']};'>"
            f"<span style='width:8px; height:8px; border-radius:999px; background:{ring_color}; display:inline-block;'></span>"
            f"<span style='color:{p['text_primary']};'>{mode_icons.get(mode_key, '')} {mode_names.get(mode_key, '')}</span>"
            f"<span style='color:{p['divider']};'>|</span>"
            f"<span>{theme_icon}</span>"
            f"<span style='color:{p['text_muted']};'>{lang_code}</span>"
            f"</div>"
        )

    render_html(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:12px; margin-bottom:1.1rem;">
            <div>
                <h1 style="margin:0; font-size:1.72rem; font-weight:750; color:{p['text_primary']}; letter-spacing:-0.025em; line-height:1.25;">
                    {t['app_title']}
                </h1>
                <div style="display:flex; align-items:center; flex-wrap:wrap; gap:10px; margin-top:6px; font-size:0.88rem; color:{p['text_muted']};">
                    <span>{t['header_sub_1']} <b style="color:{p['text_subtle']};">UIT-VSFC</b></span>
                    <span style="color:{p['divider']};">•</span>
                    <span>{t['header_sub_2']} <b style="color:{p['text_subtle']};">{t['header_sub_2_val']}</b></span>
                    <span style="color:{p['divider']};">•</span>
                    <span>{t['header_sub_3']} <b style="color:{p['text_subtle']};">{t['header_sub_3_val']}</b></span>
                </div>
            </div>
            {mode_badge_html}
        </div>
        """
    )

    with st.expander(t["expander_title"], expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"#### {t['expander_task1_title']}\n{t['expander_task1_body']}")
        with col_b:
            st.markdown(f"#### {t['expander_task2_title']}\n{t['expander_task2_body']}")
        st.markdown(f"---\n{t['expander_models_body']}")


def render_distribution_bars(
    task_title: str,
    dist_dict: Dict[str, float],
    ui_meta: Dict[str, Any],
    predicted_label: str,
    lang: str = "vi",
    theme_mode: str = "light",
):
    """Renders interactive probability distribution bars summing to 100%."""
    p = THEME_PALETTES[theme_mode]
    t = I18N[lang]
    heading_text = (
        f"{t['dist_prefix']} {task_title}"
        if lang == "vi"
        else f"{task_title} {t['dist_prefix']}"
    )
    render_html(
        f"""
        <div style="font-size:0.8rem; font-weight:700; color:{p['text_subtle']}; text-transform:uppercase; letter-spacing:0.04em; margin:0.65rem 0 0.45rem 0;">
            {heading_text}
        </div>
        """
    )
    for key, prob in dist_dict.items():
        meta = ui_meta.get(key, {})
        display_name = _get_label_display_name(meta, key, lang)
        accent, chip_bg, _ = _resolve_meta_colors(meta, theme_mode)
        pct = prob * 100.0
        is_winner = (key == predicted_label)

        row_color = p["text_primary"] if is_winner else p["text_muted"]
        row_weight = "700" if is_winner else "500"
        dot_color = accent if is_winner else p["divider"]
        pct_color = p["winner_pct"] if is_winner else p["text_subtle"]
        badge_html = (
            f"<span style='background:{chip_bg}; color:{accent}; "
            f"font-size:0.7rem; font-weight:700; padding:1px 6px; border-radius:4px; margin-left:6px;'>{t['top_badge']}</span>"
            if is_winner
            else ""
        )

        render_html(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.84rem; margin-bottom:2px; color:{row_color}; font-weight:{row_weight};">
                <span style="display:inline-flex; align-items:center; gap:6px;">
                    <span style="width:7px; height:7px; border-radius:999px; background:{dot_color}; display:inline-block;"></span>
                    <span>{display_name}</span>{badge_html}
                </span>
                <span style="font-variant-numeric:tabular-nums; color:{pct_color};">{pct:5.2f}%</span>
            </div>
            """
        )
        st.progress(min(max(float(prob), 0.0), 1.0))


def _render_task_subcard_html(
    task_label: str,
    pred_name: str,
    confidence: float,
    meta: Dict[str, str],
    theme_mode: str = "light",
) -> str:
    """Builds compact HTML for a single clean task sub-card modeled after board cards."""
    p = THEME_PALETTES[theme_mode]
    accent, chip_bg, chip_border = _resolve_meta_colors(meta, theme_mode)
    conf_pct = max(0.0, min(100.0, confidence * 100.0))

    return (
        f"<div style='background:{p['subcard_bg']}; border:1px solid {p['subcard_border']}; border-radius:10px; "
        f"padding:12px 14px; margin-bottom:10px; box-shadow:{p['subcard_shadow']};'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
        f"<span style='font-size:0.74rem; font-weight:600; color:{p['text_muted']}; text-transform:uppercase; letter-spacing:0.04em;'>{task_label}</span>"
        f"<span style='background:{chip_bg}; color:{accent}; border:1px solid {chip_border}; "
        f"font-size:0.74rem; font-weight:700; padding:2px 8px; border-radius:6px; font-variant-numeric:tabular-nums;'>{conf_pct:.1f}%</span>"
        f"</div>"
        f"<div style='font-size:1.05rem; font-weight:700; color:{p['text_primary']}; margin-bottom:10px;'>{pred_name}</div>"
        f"<div style='width:100%; height:4px; background:{p['track_bg']}; border-radius:999px; overflow:hidden;'>"
        f"<div class='vsfc-conf-fill' style='width:{conf_pct:.1f}%; height:100%; background:{accent}; border-radius:999px;'></div>"
        f"</div>"
        f"</div>"
    )


def _build_conic_donut_html(
    task_title: str,
    dist_dict: Dict[str, float],
    ui_meta: Dict[str, Any],
    predicted_label: str,
    lang: str = "vi",
    theme_mode: str = "light",
) -> str:
    """Builds a compact CSS conic-gradient donut chart card with legend and clarity tag."""
    p = THEME_PALETTES[theme_mode]
    t = I18N[lang]
    items = [(k, max(0.0, float(v))) for k, v in dist_dict.items()]
    total = sum(v for _, v in items)

    nonzero: List[Tuple[str, float, str]] = []
    for key, val in items:
        deg = (val / total) * 360.0 if total > 0 else 0.0
        if deg >= 0.4:
            accent, _, _ = _resolve_meta_colors(ui_meta.get(key, {}), theme_mode)
            nonzero.append((key, deg, accent))

    if total <= 0 or not nonzero:
        gradient = p["track_bg"]
    elif len(nonzero) == 1:
        gradient = nonzero[0][2]
    else:
        stops: List[str] = []
        cur_deg = 0.0
        for idx, (_, span, color) in enumerate(nonzero):
            end_deg = 360.0 if idx == len(nonzero) - 1 else cur_deg + span
            if end_deg - cur_deg >= 5.0:
                stops.append(f"{color} {cur_deg:.2f}deg {end_deg - 1.5:.2f}deg")
                stops.append(f"{p['subcard_bg']} {end_deg - 1.5:.2f}deg {end_deg:.2f}deg")
            else:
                stops.append(f"{color} {cur_deg:.2f}deg {end_deg:.2f}deg")
            cur_deg = end_deg
        gradient = f"conic-gradient({', '.join(stops)})"

    top_prob = max([v / total for _, v in items], default=0.0) if total > 0 else 0.0
    if top_prob >= 0.70:
        clarity_text, clarity_color = t["donut_clear"], "#10B981"
    elif top_prob >= 0.50:
        clarity_text, clarity_color = t["donut_moderate"], "#F59E0B"
    else:
        clarity_text, clarity_color = t["donut_balanced"], "#6366F1"

    legend_rows: List[str] = []
    for key, val in items:
        meta = ui_meta.get(key, {})
        accent, _, _ = _resolve_meta_colors(meta, theme_mode)
        name = _get_label_display_name(meta, key, lang)
        pct = (val / total) * 100.0 if total > 0 else 0.0
        is_winner = (key == predicted_label)
        txt_col = p["text_primary"] if is_winner else p["text_muted"]
        wt = "700" if is_winner else "500"
        legend_rows.append(
            f"<div style='display:flex; justify-content:space-between; align-items:center; gap:6px; font-size:0.72rem; line-height:1.3; color:{txt_col}; font-weight:{wt};'>"
            f"<span style='display:inline-flex; align-items:center; gap:5px; min-width:0;'>"
            f"<span style='width:7px; height:7px; border-radius:999px; background:{accent}; flex-shrink:0; display:inline-block;'></span>"
            f"<span style='overflow:hidden; text-overflow:ellipsis; white-space:nowrap;' title='{html.escape(name)}'>{html.escape(name)}</span>"
            f"</span>"
            f"<span style='font-variant-numeric:tabular-nums; flex-shrink:0;'>{pct:.1f}%</span>"
            f"</div>"
        )

    return (
        f"<div style='background:{p['subcard_bg']}; border:1px solid {p['subcard_border']}; border-radius:10px; "
        f"padding:10px 12px; box-shadow:{p['subcard_shadow']};'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center; gap:6px; margin-bottom:9px;'>"
        f"<span style='font-size:0.72rem; font-weight:700; color:{p['text_muted']}; text-transform:uppercase; letter-spacing:0.04em;'>{html.escape(task_title)}</span>"
        f"<span style='font-size:0.68rem; font-weight:700; color:{clarity_color}; background:{p['pill_bg']}; border:1px solid {p['pill_border']}; padding:1px 7px; border-radius:999px; white-space:nowrap;'>{html.escape(clarity_text)}</span>"
        f"</div>"
        f"<div style='display:flex; align-items:center; gap:11px;'>"
        f"<div class='vsfc-donut-ring' style='width:74px; height:74px; flex-shrink:0; border-radius:999px; background:{gradient}; display:flex; align-items:center; justify-content:center; box-shadow:inset 0 0 0 1px rgba(15,23,42,0.04);'>"
        f"<div style='width:48px; height:48px; border-radius:999px; background:{p['subcard_bg']}; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 1px 3px rgba(15,23,42,0.08);'>"
        f"<span style='font-size:0.8rem; font-weight:800; color:{p['text_primary']}; line-height:1.05; font-variant-numeric:tabular-nums;'>{top_prob * 100.0:.1f}%</span>"
        f"<span style='font-size:0.56rem; font-weight:700; color:{p['text_muted']}; text-transform:uppercase; letter-spacing:0.03em;'>{html.escape(t['top_badge'])}</span>"
        f"</div>"
        f"</div>"
        f"<div style='flex:1; min-width:0; display:flex; flex-direction:column; gap:3px;'>{''.join(legend_rows)}</div>"
        f"</div>"
        f"</div>"
    )


def render_prediction_card(
    model_title: str,
    result: Dict[str, Any],
    latency_ms: float,
    device_info: str = "CPU",
    model_size_mb: float = 0.0,
    container_key: str = "board_card_single",
    model_key: str = "baseline",
    lang: str = "vi",
    theme_mode: str = "light",
):
    """Renders a unified board column containing header, stacked task sub-cards, donut charts, and probability bars."""
    p = THEME_PALETTES[theme_mode]
    t = I18N[lang]
    sent_res = result["sentiment"]
    topic_res = result["topic"]

    s_meta = SENTIMENT_UI_META.get(sent_res["label"], {})
    t_meta = TOPIC_UI_META.get(topic_res["label"], {})

    s_name = _get_label_display_name(s_meta, sent_res["label"], lang)
    t_name = _get_label_display_name(t_meta, topic_res["label"], lang)

    ring_color = MODEL_RING_COLORS.get(model_key, MODEL_RING_COLORS["baseline"])
    size_text = f" · {model_size_mb:.1f} MB" if model_size_mb > 0 else ""

    with make_card_container(container_key):
        header_html = (
            f"<div style='display:flex; justify-content:space-between; align-items:center; "
            f"padding-bottom:0.85rem; margin-bottom:0.9rem; border-bottom:1px solid {p['row_divider']};'>"
            f"<div style='display:flex; align-items:center; gap:9px;'>"
            f"<span style='width:11px; height:11px; border-radius:999px; border:2.5px solid {ring_color}; display:inline-block;'></span>"
            f"<span style='font-size:1.05rem; font-weight:700; color:{p['text_primary']};'>{model_title}</span>"
            f"</div>"
            f"<span style='background:{p['pill_bg']}; color:{p['text_subtle']}; border:1px solid {p['pill_border']}; font-size:0.75rem; font-weight:600; "
            f"padding:3px 9px; border-radius:6px; font-variant-numeric:tabular-nums;'>"
            f"{latency_ms:.2f} ms · {device_info}{size_text}"
            f"</span>"
            f"</div>"
        )
        donuts_html = (
            f"<div style='font-size:0.78rem; font-weight:700; color:{p['text_subtle']}; text-transform:uppercase; letter-spacing:0.04em; margin:0.5rem 0 0.45rem 0;'>{html.escape(t['donut_section'])}</div>"
            f"<div class='vsfc-donut-grid'>"
            + _build_conic_donut_html(t["task_sentiment"], sent_res["distribution"], SENTIMENT_UI_META, sent_res["label"], lang, theme_mode)
            + _build_conic_donut_html(t["task_topic"], topic_res["distribution"], TOPIC_UI_META, topic_res["label"], lang, theme_mode)
            + f"</div>"
        )
        render_html(
            header_html
            + _render_task_subcard_html(t["task_sentiment"], s_name, float(sent_res["confidence"]), s_meta, theme_mode)
            + _render_task_subcard_html(t["task_topic"], t_name, float(topic_res["confidence"]), t_meta, theme_mode)
            + donuts_html
        )

        render_distribution_bars(
            t["task_sentiment"], sent_res["distribution"], SENTIMENT_UI_META, sent_res["label"], lang, theme_mode
        )
        render_html("<div style='height:6px;'></div>")
        render_distribution_bars(
            t["task_topic"], topic_res["distribution"], TOPIC_UI_META, topic_res["label"], lang, theme_mode
        )


def render_side_by_side(
    baseline_res: Dict[str, Any],
    baseline_lat: float,
    phobert_res: Dict[str, Any],
    phobert_lat: float,
    phobert_device: str,
    lang: str = "vi",
    theme_mode: str = "light",
):
    """Renders dual-column board comparison with consensus insights."""
    p = THEME_PALETTES[theme_mode]
    t = I18N[lang]
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        render_prediction_card(
            model_title=t["mode_baseline"],
            result=baseline_res,
            latency_ms=baseline_lat,
            device_info="CPU",
            model_size_mb=0.95,
            container_key="board_card_baseline",
            model_key="baseline",
            lang=lang,
            theme_mode=theme_mode,
        )

    with col2:
        render_prediction_card(
            model_title=t["mode_phobert"],
            result=phobert_res,
            latency_ms=phobert_lat,
            device_info=phobert_device.upper(),
            model_size_mb=515.06,
            container_key="board_card_phobert",
            model_key="phobert",
            lang=lang,
            theme_mode=theme_mode,
        )

    render_html(
        f"<div style='margin-top:1.1rem; margin-bottom:0.65rem; font-size:1.02rem; font-weight:700; color:{p['text_primary']};'>"
        f"{t['consensus_heading']}"
        f"</div>"
    )

    same_sentiment = (baseline_res["sentiment"]["label"] == phobert_res["sentiment"]["label"])
    same_topic = (baseline_res["topic"]["label"] == phobert_res["topic"]["label"])

    b_sent_label = _get_label_display_name(
        SENTIMENT_UI_META.get(baseline_res["sentiment"]["label"], {}), baseline_res["sentiment"]["label"], lang
    )
    p_sent_label = _get_label_display_name(
        SENTIMENT_UI_META.get(phobert_res["sentiment"]["label"], {}), phobert_res["sentiment"]["label"], lang
    )
    b_top_label = _get_label_display_name(
        TOPIC_UI_META.get(baseline_res["topic"]["label"], {}), baseline_res["topic"]["label"], lang
    )
    p_top_label = _get_label_display_name(
        TOPIC_UI_META.get(phobert_res["topic"]["label"], {}), phobert_res["topic"]["label"], lang
    )

    c_box1, c_box2, c_box3 = st.columns(3, gap="medium")
    with c_box1:
        if same_sentiment:
            st.success(t["agree_sent"].format(label=b_sent_label, raw=baseline_res["sentiment"]["label"]))
        else:
            st.warning(t["diff_sent"].format(b=b_sent_label, p=p_sent_label))

    with c_box2:
        if same_topic:
            st.success(t["agree_topic"].format(label=b_top_label, raw=baseline_res["topic"]["label"]))
        else:
            st.warning(t["diff_topic"].format(b=b_top_label, p=p_top_label))

    with c_box3:
        if baseline_lat <= phobert_lat:
            speedup = (phobert_lat / baseline_lat) if baseline_lat > 0 else 1.0
            st.info(t["speed_b_faster"].format(x=speedup))
        else:
            inv_speedup = (baseline_lat / phobert_lat) if phobert_lat > 0 else 1.0
            st.info(t["speed_p_faster"].format(x=inv_speedup))


def _predict_phobert_safe(
    phobert_model: PhoBERTMultiTask,
    clean_text: str,
    phobert_tok: Any,
    phobert_device: str,
    oom_msg: str,
) -> Tuple[Dict[str, Any], str]:
    """Runs PhoBERT inference with automatic CUDA OOM cache clearing and CPU failover."""
    try:
        res = phobert_model.predict(clean_text, tokenizer=phobert_tok, device=phobert_device)
        return res, phobert_device
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if phobert_device == "cuda":
            if hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
            if hasattr(phobert_model, "to"):
                phobert_model.to("cpu")
            st.warning(oom_msg)
            res = phobert_model.predict(clean_text, tokenizer=phobert_tok, device="cpu")
            return res, "cpu"
        raise


def _render_sidebar_segmented(
    label: str,
    options: List[str],
    labels_map: Dict[str, str],
    state_key: str,
    widget_key: str,
    help_text: Optional[str] = None,
) -> str:
    """Renders a compact single-select segmented icon control keeping selection non-empty."""
    current = st.session_state.get(state_key, options[0])
    if current not in options:
        current = options[0]

    if hasattr(st, "segmented_control"):
        selected = st.segmented_control(
            label,
            options=options,
            default=current,
            required=True,
            format_func=lambda opt: labels_map.get(opt, opt),
            key=widget_key,
            help=help_text,
            label_visibility="collapsed",
            width="stretch",
            wrap=False,
        )
        resolved = selected if selected in options else current
    else:
        idx = options.index(current)
        resolved = st.selectbox(
            label,
            options=options,
            index=idx,
            format_func=lambda opt: labels_map.get(opt, opt),
            key=widget_key,
            help=help_text,
            label_visibility="collapsed",
        )

    st.session_state[state_key] = resolved
    return resolved


# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------
def main():
    _init_lang = (
        st.session_state["seg_ui_lang"]
        if st.session_state.get("seg_ui_lang") in I18N
        else st.session_state.get("ui_lang", "vi")
    )
    st.set_page_config(
        page_title=I18N.get(_init_lang, I18N["vi"])["app_title"],
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if st.session_state.get("seg_theme_mode") in THEME_PALETTES:
        st.session_state["theme_mode"] = st.session_state["seg_theme_mode"]
    elif "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "light"

    if st.session_state.get("seg_ui_lang") in I18N:
        st.session_state["ui_lang"] = st.session_state["seg_ui_lang"]
    elif "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "vi"

    presets = get_presets()
    if "user_text" not in st.session_state:
        st.session_state["user_text"] = presets[0]["text"]

    # 1. Load models
    baseline_model = load_baseline_model()
    phobert_model, phobert_tok, phobert_device = load_phobert_pipeline()

    if "selected_mode_key" not in st.session_state:
        if baseline_model is not None and phobert_model is not None:
            st.session_state["selected_mode_key"] = "both"
        elif baseline_model is not None:
            st.session_state["selected_mode_key"] = "baseline"
        else:
            st.session_state["selected_mode_key"] = "phobert"

    # 2. Sidebar Setup (appearance & language first via segmented_control so radio[0]/selectbox[0]/button[0] stay invariant)
    with st.sidebar:
        cur_lang = _get_lang()
        cur_theme = _get_theme()
        t_cur = I18N[cur_lang]
        p_cur = THEME_PALETTES[cur_theme]

        render_html(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:0.85rem; padding-bottom:0.75rem; border-bottom:1px solid {p_cur['divider']};">
                <div style="width:28px; height:28px; border-radius:8px; background:{p_cur['primary']}; color:#FFFFFF; display:flex; align-items:center; justify-content:center; font-weight:800; font-size:0.78rem; letter-spacing:-0.02em;">VS</div>
                <div>
                    <div style="font-weight:800; font-size:1.05rem; color:{p_cur['brand_text']}; line-height:1.15; letter-spacing:-0.02em;">VSFC.Board</div>
                    <div style="font-size:0.74rem; color:{p_cur['text_muted']};">{t_cur['brand_sub']}</div>
                </div>
            </div>
            """
        )

        col_theme, col_lang = st.columns([0.92, 1.08], gap="small")
        with col_theme:
            theme_mode = _render_sidebar_segmented(
                label=t_cur["theme_label"],
                options=["light", "dark"],
                labels_map={"light": ":material/light_mode:", "dark": ":material/dark_mode:"},
                state_key="theme_mode",
                widget_key="seg_theme_mode",
                help_text=t_cur["theme_help"],
            )
        with col_lang:
            lang = _render_sidebar_segmented(
                label=t_cur["lang_label"],
                options=["vi", "en"],
                labels_map={"vi": "VI", "en": "EN"},
                state_key="ui_lang",
                widget_key="seg_ui_lang",
                help_text=t_cur["lang_help"],
            )

        t = I18N[lang]
        p = THEME_PALETTES[theme_mode]

        render_html(f"<div style='height:1px; background:{p['divider']}; margin:0.85rem 0;'></div>")
        render_html(
            f"<div style='font-size:0.72rem; font-weight:700; color:{p['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.35rem;'>"
            f"{t['sec_controls']}"
            f"</div>"
        )

        mode_pairs: List[Tuple[str, str]] = []
        if baseline_model is not None:
            mode_pairs.append(("baseline", t["mode_baseline"]))
        if phobert_model is not None:
            mode_pairs.append(("phobert", t["mode_phobert"]))
        if baseline_model is not None and phobert_model is not None:
            mode_pairs.append(("both", t["mode_both"]))

        if not mode_pairs:
            st.error(t["no_ckpt_err"])
            selected_mode_key = None
        else:
            keys_list = [k for k, _ in mode_pairs]
            labels_list = [lbl for _, lbl in mode_pairs]
            label_to_key = {lbl: k for k, lbl in mode_pairs}
            stored_key = st.session_state.get("selected_mode_key", keys_list[-1])
            default_idx = keys_list.index(stored_key) if stored_key in keys_list else len(labels_list) - 1

            chosen_label = st.radio(
                t["mode_radio_label"],
                options=labels_list,
                index=default_idx,
                key=f"model_mode_radio_{lang}",
                help=t["mode_radio_help"],
            )
            selected_mode_key = label_to_key[chosen_label]
            st.session_state["selected_mode_key"] = selected_mode_key

        render_html(f"<div style='height:1px; background:{p['divider']}; margin:0.9rem 0;'></div>")
        render_html(
            f"<div style='font-size:0.72rem; font-weight:700; color:{p['text_muted']}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>{t['sec_presets']}</div>"
        )

        preset_title_key = "title_en" if lang == "en" else "title"
        preset_dict = {item.get(preset_title_key, item["title"]): item["text"] for item in presets}
        preset_options = list(preset_dict.keys())
        cur_preset_idx = st.session_state.get("selected_preset_idx", 0)
        if not isinstance(cur_preset_idx, int) or not (0 <= cur_preset_idx < len(preset_options)):
            cur_preset_idx = 0

        selected_preset_title = st.selectbox(
            t["preset_select_label"],
            options=preset_options,
            index=cur_preset_idx,
            key=f"preset_select_{lang}",
        )
        if selected_preset_title in preset_options:
            st.session_state["selected_preset_idx"] = preset_options.index(selected_preset_title)

        if st.button(t["load_preset_btn"], use_container_width=True):
            st.session_state["user_text"] = preset_dict[selected_preset_title]
            st.session_state.pop("last_analyzed_text", None)

        render_html(f"<div style='height:1px; background:{p['divider']}; margin:0.9rem 0;'></div>")
        st.markdown(
            """
            **{}**:
            - {}: `{}`
            - {}: `{}` ({}: `{}`)
            """.format(
                t["status_title"],
                t["mode_baseline"],
                t["status_ready"] if baseline_model else t["status_missing"],
                t["mode_phobert"],
                t["status_ready"] if phobert_model else t["status_missing"],
                t["status_device"],
                phobert_device.upper(),
            )
        )

    # Apply theme CSS after theme selection so toggling updates immediately in the same rerun
    inject_custom_css(theme_mode)

    # 3. Main Header
    render_header(lang=lang, theme_mode=theme_mode, mode_key=selected_mode_key)

    # Fallback guidance when PhoBERT weights are missing
    if baseline_model is not None and phobert_model is None:
        st.warning(t["phobert_missing_warn"])

    # 4. Composer Card
    with make_card_container("composer_card"):
        render_html(
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>"
            f"<span style='font-size:0.95rem; font-weight:700; color:{p['text_primary']};'>{t['composer_title']}</span>"
            f"<span style='font-size:0.76rem; color:{p['text_muted']}; background:{p['pill_bg']}; border:1px solid {p['pill_border']}; padding:2px 8px; border-radius:6px;'>{t['composer_lang_tag']}</span>"
            f"</div>"
        )
        input_text = st.text_area(
            label=t["textarea_label"],
            value=st.session_state["user_text"],
            height=112,
            placeholder=t["textarea_placeholder"],
            help=t["textarea_help"],
            label_visibility="collapsed",
        )

        if lang == "en":
            matched_preset = next(
                (item for item in presets if item["text"].strip() == input_text.strip()),
                None,
            )
            if matched_preset and matched_preset.get("text_en"):
                render_html(
                    f"<div style='background:{p['pill_bg']}; border:1px solid {p['pill_border']}; border-radius:8px; "
                    f"padding:6px 10px; margin:2px 0 6px 0; font-size:0.8rem; color:{p['text_subtle']};'>"
                    f"<b style='color:{p['text_primary']};'>{html.escape(t['en_gloss_prefix'])}:</b> {html.escape(matched_preset['text_en'])}"
                    f"</div>"
                )

        words = len(input_text.strip().split()) if input_text.strip() else 0
        chars = len(input_text.strip())

        col_meta, col_clear, col_submit = st.columns([4.2, 1.15, 1.65], vertical_alignment="center")
        with col_meta:
            st.caption(t["counter_fmt"].format(words=words, chars=chars))
        with col_clear:
            if st.button(t["clear_btn"], use_container_width=True):
                st.session_state["user_text"] = ""
                st.session_state.pop("last_analyzed_text", None)
                st.rerun()
        with col_submit:
            analyze_clicked = st.button(t["analyze_btn"], type="primary", use_container_width=True)

    # 5. Analysis Execution Flow
    clean_text = input_text.strip()
    if analyze_clicked:
        if not clean_text:
            st.session_state.pop("last_analyzed_text", None)
            st.warning(t["empty_warn"])
            return
        st.session_state["last_analyzed_text"] = clean_text

    should_render_results = analyze_clicked or (
        bool(clean_text) and st.session_state.get("last_analyzed_text") == clean_text
    )

    if should_render_results:
        render_html("<div style='height:0.6rem;'></div>")

        if selected_mode_key == "both" and (baseline_model is None or phobert_model is None):
            st.warning(t["fallback_both_warn"])
            selected_mode_key = "baseline" if baseline_model else "phobert"

        if selected_mode_key == "baseline":
            if baseline_model is None:
                st.error(t["err_baseline_missing"])
                return
            with st.spinner(t["spin_baseline"]):
                t0 = time.perf_counter()
                res = baseline_model.predict(clean_text)
                dt_ms = (time.perf_counter() - t0) * 1000.0

            render_prediction_card(
                model_title=t["mode_baseline"],
                result=res,
                latency_ms=dt_ms,
                device_info="CPU",
                model_size_mb=0.95,
                container_key="board_card_single_baseline",
                model_key="baseline",
                lang=lang,
                theme_mode=theme_mode,
            )

        elif selected_mode_key == "phobert":
            if phobert_model is None:
                st.error(t["err_phobert_missing"])
                return
            with st.spinner(t["spin_phobert"]):
                t0 = time.perf_counter()
                res, phobert_device = _predict_phobert_safe(
                    phobert_model, clean_text, phobert_tok, phobert_device, t["oom_warn"]
                )
                dt_ms = (time.perf_counter() - t0) * 1000.0

            render_prediction_card(
                model_title=t["mode_phobert"],
                result=res,
                latency_ms=dt_ms,
                device_info=phobert_device.upper(),
                model_size_mb=515.06,
                container_key="board_card_single_phobert",
                model_key="phobert",
                lang=lang,
                theme_mode=theme_mode,
            )

        elif selected_mode_key == "both":
            with st.spinner(t["spin_both"]):
                t0 = time.perf_counter()
                b_res = baseline_model.predict(clean_text)
                b_lat = (time.perf_counter() - t0) * 1000.0

                t1 = time.perf_counter()
                p_res, phobert_device = _predict_phobert_safe(
                    phobert_model, clean_text, phobert_tok, phobert_device, t["oom_warn"]
                )
                p_lat = (time.perf_counter() - t1) * 1000.0

            render_side_by_side(
                baseline_res=b_res,
                baseline_lat=b_lat,
                phobert_res=p_res,
                phobert_lat=p_lat,
                phobert_device=phobert_device,
                lang=lang,
                theme_mode=theme_mode,
            )

if __name__ == "__main__":
    main()
