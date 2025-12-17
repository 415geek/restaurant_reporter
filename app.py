import os
import json
import time
import textwrap
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# =========================
# App Config
# =========================
APP_TITLE = "AuraInsight 报告生成器（Trade Area & Growth Diagnostic）"

OUTPUT_DIR = "output"
ASSETS_DIR = "assets"
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

BG_COVER_DEFAULT = os.path.join(ASSETS_DIR, "bg_cover.png")
BG_CONTENT_DEFAULT = os.path.join(ASSETS_DIR, "bg_content.png")

# Static fonts (your GitHub uploaded files)
FONT_NOTO_REG = os.path.join(FONTS_DIR, "NotoSansSC-Regular.ttf")
FONT_NOTO_BOLD = os.path.join(FONTS_DIR, "NotoSansSC-Bold.ttf")

FONT_ROBOTO_REG = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
FONT_ROBOTO_BOLD = os.path.join(FONTS_DIR, "Roboto-Bold.ttf")
FONT_ROBOTO_ITALIC = os.path.join(FONTS_DIR, "Roboto-Italic.ttf")

PAGE_W, PAGE_H = letter  # 612x792 points


# =========================
# Data Models
# =========================
@dataclass
class ReportInputs:
    report_date: str
    restaurant_cn: str
    restaurant_en: str
    address: str
    radius_miles: float
    platform_links: Dict[str, str]
    competitor_names: List[str]


# =========================
# Fonts / Typography
# =========================
def register_aurainsight_fonts():
    """
    Register production-safe static fonts.
    Must be called once before any setFont usage with these names.
    """
    # Register only if file exists; otherwise fall back to Helvetica
    if os.path.exists(FONT_NOTO_REG):
        pdfmetrics.registerFont(TTFont("Noto", FONT_NOTO_REG))
    if os.path.exists(FONT_NOTO_BOLD):
        pdfmetrics.registerFont(TTFont("Noto-Bold", FONT_NOTO_BOLD))

    if os.path.exists(FONT_ROBOTO_REG):
        pdfmetrics.registerFont(TTFont("Roboto", FONT_ROBOTO_REG))
    if os.path.exists(FONT_ROBOTO_BOLD):
        pdfmetrics.registerFont(TTFont("Roboto-Bold", FONT_ROBOTO_BOLD))
    if os.path.exists(FONT_ROBOTO_ITALIC):
        pdfmetrics.registerFont(TTFont("Roboto-Italic", FONT_ROBOTO_ITALIC))


def f_cn(bold: bool = False) -> str:
    if bold and "Noto-Bold" in pdfmetrics.getRegisteredFontNames():
        return "Noto-Bold"
    if "Noto" in pdfmetrics.getRegisteredFontNames():
        return "Noto"
    return "Helvetica-Bold" if bold else "Helvetica"


def f_en(bold: bool = False, italic: bool = False) -> str:
    if italic and "Roboto-Italic" in pdfmetrics.getRegisteredFontNames():
        return "Roboto-Italic"
    if bold and "Roboto-Bold" in pdfmetrics.getRegisteredFontNames():
        return "Roboto-Bold"
    if "Roboto" in pdfmetrics.getRegisteredFontNames():
        return "Roboto"
    return "Helvetica-Bold" if bold else "Helvetica"


# =========================
# Google Places Helpers
# =========================
def google_geocode(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": address, "key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])


def google_nearby_restaurants(lat: float, lng: float, api_key: str, radius_m: int = 1200) -> List[Dict[str, Any]]:
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    results: List[Dict[str, Any]] = []

    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": "restaurant",
        "key": api_key
    }

    for _ in range(3):
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            break

        results.extend(data.get("results", []))

        token = data.get("next_page_token")
        if not token:
            break
        time.sleep(2)  # required by Google
        params = {"pagetoken": token, "key": api_key}

    return results


def google_place_details(place_id: str, api_key: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = ",".join([
        "name", "formatted_address", "rating", "user_ratings_total",
        "types", "url", "website", "formatted_phone_number",
        "opening_hours", "reviews", "geometry"
    ])
    r = requests.get(url, params={"place_id": place_id, "fields": fields, "key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return {}
    return data.get("result", {})


def google_textsearch_place_id(query: str, api_key: str) -> Optional[str]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    r = requests.get(url, params={"query": query, "key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        return None
    return data["results"][0].get("place_id")


# =========================
# OpenAI Helper (Responses API)
# =========================
def openai_generate_report(prompt: str, api_key: str, model: str) -> str:
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": prompt,
        "temperature": 0.35,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    data = r.json()

    out = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))

    return "\n".join(out).strip()


# =========================
# Prompt Builder (McKinsey style)
# =========================
def build_prompt(
    place: Dict[str, Any],
    inputs: ReportInputs,
    competitor_places: List[Dict[str, Any]],
) -> str:
    def safe(d, k, default=None):
        if not isinstance(d, dict):
            return default
        return d.get(k, default)

    # compress reviews sample
    reviews = safe(place, "reviews", []) or []
    reviews_sample = []
    for rv in reviews[:10]:
        reviews_sample.append({
            "rating": rv.get("rating"),
            "relative_time_description": rv.get("relative_time_description"),
            "text": (rv.get("text") or "")[:280]
        })

    comp_brief = []
    for cp in competitor_places[:8]:
        comp_brief.append({
            "name": safe(cp, "name", ""),
            "address": safe(cp, "formatted_address", ""),
            "rating": safe(cp, "rating", ""),
            "user_ratings_total": safe(cp, "user_ratings_total", ""),
            "types": safe(cp, "types", []),
            "opening_hours": safe(cp, "opening_hours", {}),
        })

    data_blob = {
        "restaurant": {
            "name": safe(place, "name", ""),
            "address": safe(place, "formatted_address", inputs.address),
            "rating": safe(place, "rating", ""),
            "user_ratings_total": safe(place, "user_ratings_total", ""),
            "phone": safe(place, "formatted_phone_number", ""),
            "website": safe(place, "website", ""),
            "google_url": safe(place, "url", ""),
            "opening_hours": safe(place, "opening_hours", {}),
            "types": safe(place, "types", []),
            "reviews_sample": reviews_sample,
        },
        "trade_area": {
            "radius_miles": inputs.radius_miles,
            "city": "San Francisco",
            "zip_hint": "94116",
        },
        "platform_links": inputs.platform_links,
        "competitors": comp_brief,
        "competitor_names_input": inputs.competitor_names,
    }

    return f"""
你是 AuraInsight 的咨询顾问。请基于给定数据，输出一份“麦肯锡风格”的《门店商圈与增长诊断报告》（中文为主，保留英文主标题），要求：

写作规则：
- 结论先行：每个章节开头给 2–4 条 Key Takeaways（要短、要硬、要可验证）。
- 数据说话：必须引用输入 JSON 中已有数据（评分 rating、评价数 user_ratings_total、营业时间、竞对对比、评论关键词等）。对无法直接量化的内容，用“可验证指标 + 采数方式”表达，避免空泛。
- 输出结构必须包含并按顺序排列：
  Executive Summary
  1. Trade Area Definition & Demand Fundamentals
  2. Platform Ecosystem Analysis（Direct/UberEats/Fantuan/HungryPanda 的角色分工 + KPI）
  3. Competitive Landscape（竞对至少3家：线上效率/信任资产/平台覆盖/风险点）
  4. Pricing & Order Economics（价格带、锚点、动态调价机制：含每月价格基准化流程）
  5. Time-based Demand Structure（午餐/晚餐/下午茶/夜宵：策略与验证指标）
  6. Strategic Implications（数据型结论）
  7. Strategic Initiatives & Execution Roadmap（重点写深：各平台菜单结构调整、竞对价格对齐、运营期动态调价、虚拟店/虚拟品牌如“华记冰室”的上线策略、KPI与里程碑）
  Data Gaps（数据缺口清单：为了下次报告更精准，需要补充哪些字段）

风格：
- 专业、克制、像咨询交付物。
- 不要夸张营销语。
- 可以使用小表格（文本形式即可）。

报告信息：
- 报告日期：{inputs.report_date}
- 商家中文名：{inputs.restaurant_cn}
- 商家英文名：{inputs.restaurant_en}
- 地址：{inputs.address}
- 配送半径：{inputs.radius_miles} miles

数据输入(JSON)：
{json.dumps(data_blob, ensure_ascii=False, indent=2)}
""".strip()


# =========================
# PDF Rendering
# =========================
def draw_bg(c: canvas.Canvas, bg_path: str):
    if bg_path and os.path.exists(bg_path):
        c.drawImage(bg_path, 0, 0, width=PAGE_W, height=PAGE_H, mask="auto")


def wrap_lines(text: str, max_chars: int) -> List[str]:
    lines: List[str] = []
    for para in text.splitlines():
        para = para.rstrip()
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=max_chars, break_long_words=False, replace_whitespace=False))
    return lines


def draw_footer(c: canvas.Canvas, report_date: str, page_num: int):
    # Footer style: subtle, consulting-like
    c.setFillColor(colors.HexColor("#7A7A7A"))
    c.setFont(f_en(False), 8)
    c.drawString(0.75 * inch, 0.55 * inch, f"Confidential | Generated by AuraInsight | {report_date}")
    c.drawRightString(PAGE_W - 0.75 * inch, 0.55 * inch, f"Page {page_num}")
    c.setFillColor(colors.black)


def render_pdf(report_text: str, inputs: ReportInputs, bg_cover: str, bg_content: str) -> str:
    register_aurainsight_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_name = "".join([ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in inputs.restaurant_en]).strip()
    safe_name = safe_name.replace(" ", "_") or "Restaurant"

    filename = f"AuraInsight_{safe_name}_{inputs.report_date.replace('/','-')}.pdf"
    out_path = os.path.join(OUTPUT_DIR, filename)

    c = canvas.Canvas(out_path, pagesize=letter)

    # ---- Cover Page ----
    draw_bg(c, bg_cover)

    # Cover typography coordinates tuned for your style
    c.setFillColor(colors.HexColor("#1F2A33"))
    c.setFont(f_en(True), 26)
    c.drawCentredString(PAGE_W / 2, 315, "AuraInsight")

    c.setFont(f_cn(True), 18)
    c.setFillColor(colors.black)
    c.drawCentredString(PAGE_W / 2, 285, "【门店分析报告】")

    c.setFont(f_en(False), 11)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(PAGE_W / 2, 260, inputs.report_date)

    c.setFont(f_cn(False), 15)
    c.setFillColor(colors.black)
    c.drawCentredString(PAGE_W / 2, 165, inputs.restaurant_cn or inputs.restaurant_en)

    c.setFont(f_en(False), 12)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(PAGE_W / 2, 144, inputs.restaurant_en)

    c.setFont(f_en(False), 10)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(PAGE_W / 2, 124, inputs.address)

    c.showPage()

    # ---- Content Pages ----
    draw_bg(c, bg_content)
    page_num = 1

    # Header (first content page)
    left = 0.85 * inch
    right = PAGE_W - 0.85 * inch
    top = PAGE_H - 1.05 * inch

    c.setFillColor(colors.black)
    c.setFont(f_en(True), 16)
    c.drawString(left, top, "Trade Area & Growth Diagnostic Report")
    c.setFont(f_en(False), 10)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawString(left, top - 16, "Data-driven | Community-based | Legacy Brand")

    y = top - 40

    # Rendering logic: chunk by paragraphs, apply heading styles when detected
    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]

    def new_content_page():
        nonlocal y, page_num
        draw_footer(c, inputs.report_date, page_num)
        c.showPage()
        page_num += 1
        draw_bg(c, bg_content)
        y = PAGE_H - 1.05 * inch

    def draw_heading(text: str):
        nonlocal y
        if y < 1.4 * inch:
            new_content_page()
        c.setFillColor(colors.black)
        c.setFont(f_en(True), 13)  # heading in English bold by default
        # If it looks Chinese-heavy, use CN bold
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            c.setFont(f_cn(True), 13)
        c.drawString(left, y, text[:120])
        y -= 18

    def draw_body(text: str):
        nonlocal y
        if y < 1.4 * inch:
            new_content_page()
        c.setFillColor(colors.black)

        # Choose font based on Chinese presence
        has_cn = any("\u4e00" <= ch <= "\u9fff" for ch in text)
        body_font = f_cn(False) if has_cn else f_en(False)

        c.setFont(body_font, 10)
        max_chars = 110  # conservative; depends on font
        lines = wrap_lines(text, max_chars)

        for line in lines:
            if y < 1.2 * inch:
                new_content_page()
            c.drawString(left, y, line)
            y -= 14
        y -= 6

    # Simple heading detection
    heading_starts = (
        "Executive Summary",
        "1.", "2.", "3.", "4.", "5.", "6.", "7.",
        "Data Gaps", "数据缺口", "附录", "Appendix"
    )

    for p in paragraphs:
        first_line = p.splitlines()[0].strip()

        if first_line.startswith(heading_starts):
            draw_heading(first_line)
            rest = "\n".join(p.splitlines()[1:]).strip()
            if rest:
                draw_body(rest)
        else:
            draw_body(p)

    # Footer last page
    draw_footer(c, inputs.report_date, page_num)
    c.save()
    return out_path


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

google_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
openai_key = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("配置")
    model = st.selectbox("OpenAI 模型", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    radius_miles = st.slider("商圈半径（miles）", 1.0, 6.0, 4.0, 0.5)
    nearby_radius_m = st.slider("Google Nearby 搜索半径（米）", 300, 3000, 1200, 100)

    bg_cover = st.text_input("封面背景图路径", BG_COVER_DEFAULT)
    bg_content = st.text_input("内容页背景图路径", BG_CONTENT_DEFAULT)

    st.divider()
    st.caption("字体文件（需已上传到 assets/fonts/）")
    st.code(
        "NotoSansSC-Regular.ttf\n"
        "NotoSansSC-Bold.ttf\n"
        "Roboto-Regular.ttf\n"
        "Roboto-Bold.ttf\n"
        "Roboto-Italic.ttf"
    )

if not google_key:
    st.warning("未检测到 GOOGLE_MAPS_API_KEY，请在 .streamlit/secrets.toml 配置。")
if not openai_key:
    st.warning("未检测到 OPENAI_API_KEY，请在 .streamlit/secrets.toml 配置。")

st.subheader("Step 1｜输入地址 → 搜索附近餐厅")
address_input = st.text_input("输入地址（用于定位与搜索附近餐厅）", value="2406 19th Ave, San Francisco, CA 94116")

colA, colB = st.columns([1, 1])

with colA:
    if st.button("搜索附近餐厅", type="primary", disabled=not google_key):
        geo = google_geocode(address_input, google_key)
        if not geo:
            st.error("无法解析地址，请输入更完整地址（含城市/州）。")
        else:
            lat, lng = geo
            places = google_nearby_restaurants(lat, lng, google_key, radius_m=nearby_radius_m)
            st.session_state["places"] = places
            st.session_state["geo"] = (lat, lng)
            st.success(f"已找到 {len(places)} 家附近餐厅。")

places = st.session_state.get("places", [])

selected_place_id = None
place_details = st.session_state.get("place_details", {})

if places:
    options = []
    id_map = {}
    for p in places:
        name = p.get("name", "")
        addr = p.get("vicinity", "")
        rating = p.get("rating", "NA")
        total = p.get("user_ratings_total", "NA")
        pid = p.get("place_id", "")
        label = f"{name} | {addr} | ⭐{rating} ({total})"
        options.append(label)
        id_map[label] = pid

    selected_label = st.selectbox("选择目标餐厅（Google Nearby）", options)
    selected_place_id = id_map.get(selected_label)

    if st.button("拉取餐厅详情（Google Place Details）", disabled=not google_key):
        if not selected_place_id:
            st.error("请先选择一个餐厅。")
        else:
            details = google_place_details(selected_place_id, google_key)
            if not details:
                st.error("拉取详情失败。")
            else:
                st.session_state["place_details"] = details
                place_details = details
                st.success("已拉取餐厅详情。")

if place_details:
    st.subheader("Step 2｜确认信息 + 填平台链接 + 配竞对")
    col1, col2 = st.columns(2)

    with col1:
        restaurant_en = st.text_input("餐厅英文名", value=place_details.get("name", ""))
        restaurant_cn = st.text_input("餐厅中文名（可选）", value="")
        formatted_address = st.text_input("餐厅地址", value=place_details.get("formatted_address", address_input))

        rating = place_details.get("rating", "")
        total = place_details.get("user_ratings_total", "")
        st.caption(f"Google 数据：⭐{rating}（{total} reviews）")

    with col2:
        st.markdown("### 平台链接（建议填全）")
        direct_url = st.text_input("Direct / order.online", value="")
        uber_url = st.text_input("Uber Eats", value="")
        fantuan_url = st.text_input("饭团 Fantuan", value="")
        panda_url = st.text_input("HungryPanda 熊猫", value="")

        st.markdown("### 竞对（每行一个）")
        competitor_names = st.text_area(
            "竞对名单",
            value="Smile House Cafe\n凤凰聚会\n大家乐",
            height=110
        )

    # Competitor pull
    competitors = [x.strip() for x in competitor_names.splitlines() if x.strip()]
    competitor_places = st.session_state.get("competitor_places", [])

    colX, colY = st.columns([1, 1])

    with colX:
        if st.button("拉取竞对 Google 数据（可选）", disabled=not google_key):
            pulled = []
            for name in competitors[:8]:
                pid = google_textsearch_place_id(f"{name} San Francisco", google_key)
                if pid:
                    pulled.append(google_place_details(pid, google_key))
            st.session_state["competitor_places"] = pulled
            competitor_places = pulled
            st.success(f"已拉取 {len(pulled)} 家竞对详情。")

    with colY:
        st.caption("提示：竞对数据越完整，报告越“数据化”。")

    st.subheader("Step 3｜生成深度分析（ChatGPT）")
    report_date = dt.datetime.now().strftime("%m/%d/%Y")

    platform_links = {
        "direct": direct_url.strip(),
        "uber_eats": uber_url.strip(),
        "fantuan": fantuan_url.strip(),
        "hungrypanda": panda_url.strip(),
    }

    if st.button("生成咨询级报告内容", type="primary", disabled=not openai_key):
        inputs = ReportInputs(
            report_date=report_date,
            restaurant_cn=restaurant_cn.strip() or restaurant_en.strip(),
            restaurant_en=restaurant_en.strip(),
            address=formatted_address.strip(),
            radius_miles=radius_miles,
            platform_links=platform_links,
            competitor_names=competitors,
        )
        prompt = build_prompt(place_details, inputs, competitor_places)

        with st.spinner("正在生成报告（咨询级结构 + 数据引用）..."):
            report_text = openai_generate_report(prompt, openai_key, model=model)

        st.session_state["report_text"] = report_text
        st.session_state["report_inputs"] = inputs
        st.success("报告内容已生成。")

# Preview + PDF
report_text = st.session_state.get("report_text", "")
report_inputs: Optional[ReportInputs] = st.session_state.get("report_inputs", None)

if report_text and report_inputs:
    st.subheader("预览（可编辑）")
    edited = st.text_area("报告正文（可直接改）", value=report_text, height=460)
    st.session_state["report_text"] = edited

    st.subheader("Step 4｜生成 PDF（套用封面/内容页背景图）")

    colP, colQ = st.columns([1, 1])
    with colP:
        if not os.path.exists(bg_cover):
            st.warning(f"封面背景图不存在：{bg_cover}")
        if not os.path.exists(bg_content):
            st.warning(f"内容页背景图不存在：{bg_content}")

    with colQ:
        st.caption("建议：背景图尽量用 1275x1650 或更高分辨率，避免打印模糊。")

    if st.button("生成 PDF", type="primary"):
        with st.spinner("正在生成 PDF..."):
            pdf_path = render_pdf(
                report_text=st.session_state["report_text"],
                inputs=report_inputs,
                bg_cover=bg_cover,
                bg_content=bg_content,
            )
        st.success("PDF 生成完成。")
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="下载 PDF",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
        st.caption(f"输出路径：{pdf_path}")
else:
    st.info("完成餐厅选择并生成报告后，这里会显示预览与 PDF 下载。")
