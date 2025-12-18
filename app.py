import os
import re
import json
import time
import textwrap
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import hmac


# ======================================
# AUTH GATE (MUST RUN BEFORE UI)
# ======================================
def _get_allowed_passwords() -> List[str]:
    pw_list = st.secrets.get("ADMIN_PASSWORDS", None)
    if pw_list and isinstance(pw_list, (list, tuple)) and len(pw_list) > 0:
        return [str(x) for x in pw_list]
    single = st.secrets.get("ADMIN_PASSWORD", "")
    return [str(single)] if single else []


def _secure_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def require_login():
    """
    Gate all app content behind a password stored in secrets.toml.
    """
    st.session_state.setdefault("auth_ok", False)
    st.session_state.setdefault("auth_tries", 0)
    st.session_state.setdefault("auth_locked_until", 0.0)

    # already authed
    if st.session_state["auth_ok"]:
        return True

    # lockout
    now = time.time()
    if now < st.session_state["auth_locked_until"]:
        wait_s = int(st.session_state["auth_locked_until"] - now)
        st.error(f"尝试次数过多，请 {wait_s}s 后再试。")
        st.stop()

    allowed = _get_allowed_passwords()
    if not allowed:
        st.error("未配置管理员密码：请在 .streamlit/secrets.toml 设置 ADMIN_PASSWORD 或 ADMIN_PASSWORDS。")
        st.stop()

    # Login UI (no set_page_config here to avoid duplicates)
    st.markdown("## AuraInsight 工具登录")
    st.caption("请输入管理员设置的密码后进入。")

    pw = st.text_input("密码", type="password")
    if st.button("登录", type="primary"):
        ok = any(_secure_compare(pw, x) for x in allowed)
        if ok:
            st.session_state["auth_ok"] = True
            st.session_state["auth_tries"] = 0
            st.success("登录成功。")
            st.rerun()
        else:
            st.session_state["auth_tries"] += 1
            st.error("密码错误。")
            if st.session_state["auth_tries"] >= 5:
                st.session_state["auth_locked_until"] = time.time() + 60
                st.session_state["auth_tries"] = 0
                st.warning("已暂时锁定 60 秒。")

    st.stop()


def logout_button():
    if st.button("登出"):
        st.session_state["auth_ok"] = False
        st.session_state["auth_locked_until"] = 0.0
        st.session_state["auth_tries"] = 0
        st.rerun()


# ======================================
# Config
# ======================================
APP_TITLE = "AuraInsights商圈分析报告生成器（Trade Area & Growth Diagnostic）"
OUTPUT_DIR = "output"
ASSETS_DIR = "assets"
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

# Hidden from users (fixed assets)
BG_COVER = os.path.join(ASSETS_DIR, "bg_cover.png")
BG_CONTENT = os.path.join(ASSETS_DIR, "bg_content.png")

# Content header drawing hidden (fixed)
DRAW_CONTENT_HEADER = False

# Static fonts
FONT_NOTO_REG = os.path.join(FONTS_DIR, "NotoSansSC-Regular.ttf")
FONT_NOTO_BOLD = os.path.join(FONTS_DIR, "NotoSansSC-Bold.ttf")
FONT_ROBOTO_REG = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
FONT_ROBOTO_BOLD = os.path.join(FONTS_DIR, "Roboto-Bold.ttf")
FONT_ROBOTO_ITALIC = os.path.join(FONTS_DIR, "Roboto-Italic.ttf")

PAGE_W, PAGE_H = letter  # 612x792


# ======================================
# Data Models
# ======================================
@dataclass
class ReportInputs:
    report_date: str
    restaurant_cn: str
    restaurant_en: str
    address: str
    radius_miles: float
    platform_links: Dict[str, str]
    competitors: List[Dict[str, str]]
    competitor_menu_notes: str
    order_upload_meta: Dict[str, Any]
    extra_business_context: str


# ======================================
# Fonts / Typography
# ======================================
def register_aurainsight_fonts():
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


def is_ascii_line(s: str) -> bool:
    s = s.strip()
    return bool(s) and all(ord(ch) < 128 for ch in s)


# ======================================
# Google Places
# ======================================
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
    params = {"location": f"{lat},{lng}", "radius": radius_m, "type": "restaurant", "key": api_key}

    for _ in range(3):
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            break
        results.extend(data.get("results", []))
        token = data.get("next_page_token")
        if not token:
            break
        time.sleep(2)
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


# ======================================
# Census ACS (Demographics)
# ======================================
def census_tract_from_latlng(lat: float, lng: float) -> Optional[Dict[str, str]]:
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {
        "x": lng,
        "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    try:
        geos = data["result"]["geographies"]
        tract = geos["Census Tracts"][0]
        return {
            "STATE": tract["STATE"],
            "COUNTY": tract["COUNTY"],
            "TRACT": tract["TRACT"],
            "NAME": tract.get("NAME", "")
        }
    except Exception:
        return None


def acs_5y_profile(state: str, county: str, tract: str, year: int = 2023) -> Optional[Dict[str, Any]]:
    vars_map = {
        "pop_total": "B01003_001E",
        "median_income": "B19013_001E",
        "median_age": "B01002_001E",
        "white": "B02001_002E",
        "black": "B02001_003E",
        "asian": "B02001_005E",
        "other": "B02001_007E",
        "hispanic": "B03003_003E",
        "housing_units": "B25001_001E",
        "owner_occ": "B25003_002E",
        "renter_occ": "B25003_003E",
        "avg_hh_size": "B25010_001E",
    }

    get_vars = ",".join(["NAME"] + list(vars_map.values()))
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": get_vars, "for": f"tract:{tract}", "in": f"state:{state} county:{county}"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data or len(data) < 2:
        return None

    headers = data[0]
    values = data[1]
    row = dict(zip(headers, values))

    def to_num(x):
        try:
            return float(x)
        except Exception:
            return None

    out = {"year": year, "name": row.get("NAME", ""), "state": state, "county": county, "tract": tract}
    for k, v in vars_map.items():
        out[k] = to_num(row.get(v))

    pop = out.get("pop_total") or 0

    def pct(x):
        if pop <= 0 or x is None:
            return None
        return x / pop

    out["pct_asian"] = pct(out.get("asian"))
    out["pct_white"] = pct(out.get("white"))
    out["pct_black"] = pct(out.get("black"))
    out["pct_hispanic"] = pct(out.get("hispanic"))

    owner = out.get("owner_occ")
    renter = out.get("renter_occ")
    occ_total = (owner or 0) + (renter or 0)
    out["pct_owner"] = ((owner or 0) / occ_total) if occ_total > 0 else None
    out["pct_renter"] = ((renter or 0) / occ_total) if occ_total > 0 else None

    return out


# ======================================
# OpenAI (Responses API)
# ======================================
def openai_generate(prompt: str, api_key: str, model: str) -> str:
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt, "temperature": 0.35}
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    out = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))

    return "\n".join(out).strip()


# ======================================
# Text Sanitization
# ======================================
def sanitize_text(text: str) -> str:
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.M)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$', '', text, flags=re.M)
    text = text.replace("```", "").replace("`", "")
    text = text.replace("•", "-")
    return text.strip()


# ======================================
# Prompt Builder (No Markdown)
# ======================================
def build_prompt(
    place: Dict[str, Any],
    inputs: ReportInputs,
    competitor_places: List[Dict[str, Any]],
    acs: Optional[Dict[str, Any]],
) -> str:
    def safe(d, k, default=None):
        if not isinstance(d, dict):
            return default
        return d.get(k, default)

    reviews = safe(place, "reviews", []) or []
    reviews_sample = []
    for rv in reviews[:10]:
        reviews_sample.append({
            "rating": rv.get("rating"),
            "time": rv.get("relative_time_description"),
            "text": (rv.get("text") or "")[:260]
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
            "approximation_note": "demographics use tract-level ACS near restaurant coordinate (proxy for trade area)",
        },
        "demographics_acs": acs or {"note": "ACS not available; propose data collection plan"},
        "platform_links": inputs.platform_links,
        "competitors_google": comp_brief,
        "competitors_user_input": inputs.competitors,
        "competitor_menu_notes": inputs.competitor_menu_notes,
        "order_upload_meta": inputs.order_upload_meta,
        "extra_business_context": inputs.extra_business_context,
    }

    return f"""
你是 AuraInsight 的咨询顾问。请基于输入 JSON，输出一份“麦肯锡风格”的《门店商圈与增长诊断报告》文本，中文为主，允许少量英文标题。
必须遵守：
A) 严禁输出 Markdown 语法（不要出现：#、##、**、|---|、```、[]()）。
B) 章节标题用： 【章节标题】；列表用： - 文字；小表格用： 表格: 列1,列2,列3（最多6行）。
C) 每章开头给 2–4 条 Key Takeaways（尽量带数字）。
D) 每条建议必须包含：【动作】【原因】【预期影响】【KPI】【2周验证方法】。
E) 必须应用：STP、JTBD、Menu Engineering、Anchoring、ERRC，并解释适配性。

报告信息：
- 报告日期：{inputs.report_date}
- 商家中文名：{inputs.restaurant_cn}
- 商家英文名：{inputs.restaurant_en}
- 地址：{inputs.address}
- 配送半径：{inputs.radius_miles} miles

章节顺序：
【Executive Summary】
【1. Trade Area & Demographics】
【2. Customer Segments & JTBD】
【3. Platform Ecosystem Strategy】
【4. Competitive Landscape & Differentiation】
【5. Pricing, Anchors & Promo Economics】
【6. Menu Architecture & Menu Engineering】
【7. Operating Playbook & 30/60/90 Roadmap】
【Data Gaps & How to Collect】

输入 JSON：
{json.dumps(data_blob, ensure_ascii=False, indent=2)}
""".strip()


# ======================================
# PDF Rendering
# ======================================
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
    c.setFillColor(colors.HexColor("#7A7A7A"))
    c.setFont(f_en(False), 8)
    c.drawString(0.75 * inch, 0.55 * inch, f"Confidential | Generated by AuraInsight | {report_date}")
    c.drawRightString(PAGE_W - 0.75 * inch, 0.55 * inch, f"Page {page_num}")
    c.setFillColor(colors.black)


def parse_sections(text: str) -> List[Tuple[str, str]]:
    text = text.strip()
    pattern = r'(【[^【】]+】)'
    parts = re.split(pattern, text)
    sections = []
    cur_title = None
    cur_body = []
    for p in parts:
        if not p:
            continue
        if p.startswith("【") and p.endswith("】"):
            if cur_title is not None:
                sections.append((cur_title, "\n".join(cur_body).strip()))
            cur_title = p.replace("【", "").replace("】", "").strip()
            cur_body = []
        else:
            cur_body.append(p)
    if cur_title is not None:
        sections.append((cur_title, "\n".join(cur_body).strip()))
    return sections


def render_pdf(report_text: str, inputs: ReportInputs) -> str:
    register_aurainsight_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_name = "".join([ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in inputs.restaurant_en]).strip()
    safe_name = safe_name.replace(" ", "_") or "Restaurant"
    filename = f"AuraInsight_{safe_name}_{inputs.report_date.replace('/','-')}.pdf"
    out_path = os.path.join(OUTPUT_DIR, filename)

    c = canvas.Canvas(out_path, pagesize=letter)

    # ---- Cover ----
    draw_bg(c, BG_COVER)
    c.setFillColor(colors.HexColor("#1F2A33"))
    c.setFont(f_en(True), 26)
    c.drawCentredString(PAGE_W / 2, 315, "AuraInsight")

    c.setFillColor(colors.black)
    c.setFont(f_cn(True), 18)
    c.drawCentredString(PAGE_W / 2, 285, "【门店分析报告】")

    c.setFillColor(colors.HexColor("#333333"))
    c.setFont(f_en(False), 11)
    c.drawCentredString(PAGE_W / 2, 260, inputs.report_date)

    c.setFillColor(colors.black)
    c.setFont(f_cn(False), 15)
    c.drawCentredString(PAGE_W / 2, 165, inputs.restaurant_cn or inputs.restaurant_en)

    c.setFillColor(colors.HexColor("#333333"))
    c.setFont(f_en(False), 12)
    c.drawCentredString(PAGE_W / 2, 144, inputs.restaurant_en)

    c.setFont(f_en(False), 10)
    c.drawCentredString(PAGE_W / 2, 124, inputs.address)

    c.showPage()

    # ---- Content pages ----
    draw_bg(c, BG_CONTENT)
    page_num = 1

    left = 0.85 * inch
    top = PAGE_H - 1.05 * inch
    y = top - 10  # avoid any pre-printed header

    def new_page():
        nonlocal y, page_num
        draw_footer(c, inputs.report_date, page_num)
        c.showPage()
        page_num += 1
        draw_bg(c, BG_CONTENT)
        y = top - 10

    def draw_heading(title: str):
        nonlocal y
        if y < 1.6 * inch:
            new_page()
        c.setFillColor(colors.black)
        font = f_cn(True) if any("\u4e00" <= ch <= "\u9fff" for ch in title) else f_en(True)
        c.setFont(font, 13)
        c.drawString(left, y, title[:120])
        y -= 18

    def draw_body(text: str):
        nonlocal y
        max_chars = 110
        for line in wrap_lines(text, max_chars):
            if y < 1.2 * inch:
                new_page()
            font = f_en(False) if is_ascii_line(line) else f_cn(False)
            c.setFillColor(colors.black)
            c.setFont(font, 10)
            c.drawString(left, y, line)
            y -= 14
        y -= 8

    sections = parse_sections(report_text)
    if not sections:
        draw_body(report_text)
    else:
        for title, body in sections:
            draw_heading(title)
            if body:
                draw_body(body)

    draw_footer(c, inputs.report_date, page_num)
    c.save()
    return out_path


# ======================================
# Order Upload
# ======================================
def summarize_uploaded_orders(files: List[Any]) -> Dict[str, Any]:
    meta = {"files": [], "notes": "Provide platform exports (CSV). System summarizes schema for analysis."}
    for f in files:
        try:
            df = pd.read_csv(f)
            cols = list(df.columns)[:50]
            meta["files"].append({
                "name": getattr(f, "name", "uploaded.csv"),
                "rows": int(df.shape[0]),
                "cols_sample": cols,
                "date_col_guess": next((c for c in cols if "date" in c.lower() or "time" in c.lower()), None),
            })
        except Exception as e:
            meta["files"].append({"name": getattr(f, "name", "uploaded"), "error": str(e)[:200]})
    return meta


# ======================================
# Streamlit UI (LOGIN FIRST)
# ======================================
st.set_page_config(page_title=APP_TITLE, layout="wide")

# MUST BE HERE: gate everything
require_login()

st.title(APP_TITLE)

google_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
openai_key = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("配置")
    model = st.selectbox("OpenAI 模型", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    radius_miles = st.slider("商圈半径（miles）", 1.0, 6.0, 4.0, 0.5)
    nearby_radius_m = st.slider("Google Nearby 搜索半径（米）", 300, 3000, 1200, 100)

    st.divider()
    logout_button()

    st.divider()
    # Build by c8geek + LinkedIn link (replace URL with your actual LinkedIn)
    st.caption("Build by c8geek")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/your-linkedin-here/)")

if not google_key:
    st.warning("未检测到 GOOGLE_MAPS_API_KEY，请在 .streamlit/secrets.toml 配置。")
if not openai_key:
    st.warning("未检测到 OPENAI_API_KEY，请在 .streamlit/secrets.toml 配置。")


# -------------------------
# Step 1: Address -> Nearby -> Select restaurant
# -------------------------
st.subheader("Step 1｜输入地址 → 搜索附近餐厅")
address_input = st.text_input("输入地址（用于定位并搜索附近餐厅）", value="2406 19th Ave, San Francisco, CA 94116")

colA, colB = st.columns([1, 1])
with colA:
    if st.button("搜索附近餐厅", type="primary", disabled=not google_key):
        geo = google_geocode(address_input, google_key)
        if not geo:
            st.error("无法解析地址，请输入更完整地址（含城市/州）。")
        else:
            lat, lng = geo
            places = google_nearby_restaurants(lat, lng, google_key, radius_m=nearby_radius_m)
            st.session_state["geo"] = (lat, lng)
            st.session_state["places"] = places
            st.success(f"已找到 {len(places)} 家附近餐厅。")

places = st.session_state.get("places", [])
selected_place_id = None

if places:
    options, id_map = [], {}
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
                st.success("已拉取餐厅详情。")

place_details = st.session_state.get("place_details", {})

# -------------------------
# Step 2: Demographics (ACS) + Competitors links + uploads
# -------------------------
if place_details:
    st.subheader("Step 2｜自动补齐商圈人口/收入等 + 竞对平台链接 + 数据补录入口")

    rest_lat = None
    rest_lng = None
    try:
        loc = place_details.get("geometry", {}).get("location", {})
        rest_lat = float(loc.get("lat"))
        rest_lng = float(loc.get("lng"))
    except Exception:
        pass

    acs_data = None
    tract_info = None

    col1, col2 = st.columns([1, 1])
    with col1:
        restaurant_en = st.text_input("餐厅英文名", value=place_details.get("name", ""))
        restaurant_cn = st.text_input("餐厅中文名（可选）", value="")
        formatted_address = st.text_input("餐厅地址", value=place_details.get("formatted_address", address_input))

        rating = place_details.get("rating", "")
        total = place_details.get("user_ratings_total", "")
        st.caption(f"Google 数据：⭐{rating}（{total} reviews）")

        extra_context = st.text_area(
            "补充业务背景（可选）",
            value="例如：经营年限、主打菜、目标客群、当前痛点（单量/评分/利润/人手等）。",
            height=110
        )

    with col2:
        st.markdown("### 平台链接（门店自身）")
        direct_url = st.text_input("Direct / order.online", value="")
        uber_url = st.text_input("Uber Eats", value="")
        doordash_url = st.text_input("DoorDash（可选）", value="")
        fantuan_url = st.text_input("饭团 Fantuan", value="")
        panda_url = st.text_input("HungryPanda 熊猫", value="")

    with st.expander("自动获取商圈人口/收入/年龄/族裔/租住比例（US Census ACS）", expanded=True):
        if rest_lat and rest_lng:
            if st.button("获取 ACS 商圈画像（自动）"):
                tract_info = census_tract_from_latlng(rest_lat, rest_lng)
                if not tract_info:
                    st.warning("无法获取 tract 信息（Census geocoder）。")
                else:
                    acs_data = acs_5y_profile(
                        state=tract_info["STATE"],
                        county=tract_info["COUNTY"],
                        tract=tract_info["TRACT"],
                        year=2023
                    )
                    st.session_state["tract_info"] = tract_info
                    st.session_state["acs_data"] = acs_data
                    st.success("已获取 ACS 数据（tract 级别代理）。" if acs_data else "ACS 数据返回为空。")
        else:
            st.info("未能从 Google Place Details 获取坐标，无法调用 ACS。")

        tract_info = st.session_state.get("tract_info", tract_info)
        acs_data = st.session_state.get("acs_data", acs_data)

        if acs_data:
            pop = acs_data.get("pop_total")
            inc = acs_data.get("median_income")
            age = acs_data.get("median_age")
            pct_asian = acs_data.get("pct_asian")
            pct_renter = acs_data.get("pct_renter")
            st.write({
                "ACS Year": acs_data.get("year"),
                "Geography": acs_data.get("name"),
                "Population (tract)": None if pop is None else int(pop),
                "Median HH Income": None if inc is None else f"${int(inc):,}",
                "Median Age": age,
                "% Asian (proxy)": None if pct_asian is None else f"{pct_asian*100:.1f}%",
                "% Renter (proxy)": None if pct_renter is None else f"{pct_renter*100:.1f}%",
                "Note": "ACS 为 tract 级别代理，作为 3–4 miles 商圈的近似画像；报告中会明确该假设。"
            })

    st.markdown("### 竞对信息（可增删行：用于差异化与菜单策略）")
    default_comp = pd.DataFrame([
        {"name": "Smile House Cafe", "ubereats": "", "doordash": "", "fantuan": "", "hungrypanda": "", "website": "", "notes": ""},
        {"name": "凤凰聚会", "ubereats": "", "doordash": "", "fantuan": "", "hungrypanda": "", "website": "", "notes": ""},
        {"name": "大家乐", "ubereats": "", "doordash": "", "fantuan": "", "hungrypanda": "", "website": "", "notes": ""},
    ])
    comp_df = st.data_editor(default_comp, num_rows="dynamic", use_container_width=True, key="comp_editor")
    competitors = comp_df.fillna("").to_dict("records")

    competitor_places = st.session_state.get("competitor_places", [])
    colx, coly = st.columns([1, 1])
    with colx:
        if st.button("（可选）拉取竞对 Google 数据", disabled=not google_key):
            pulled = []
            for row in competitors[:8]:
                nm = (row.get("name") or "").strip()
                if not nm:
                    continue
                pid = google_textsearch_place_id(f"{nm} San Francisco", google_key)
                if pid:
                    pulled.append(google_place_details(pid, google_key))
            st.session_state["competitor_places"] = pulled
            competitor_places = pulled
            st.success(f"已拉取 {len(pulled)} 家竞对 Google 详情。")
    with coly:
        st.caption("提示：Google 数据用于“信任资产/评价量/营业时间对比”；菜单/价格差异需平台链接或手动补录。")

    with st.expander("竞对菜单/价格补录（强烈建议，提升差异化分析深度）", expanded=False):
        comp_menu_notes = st.text_area(
            "你可以粘贴：竞对 Top 20 菜品+价格 / 套餐结构 / 主推品类 / 活动信息（任意格式均可）",
            value="",
            height=160
        )
        st.caption("建议：每个竞对至少给 10–20 个菜品与价格（或截图转文字），差异化建议会明显更“硬”。")

    with st.expander("上传订单报表（CSV，可选：用于时段/客单/热销/KPI）", expanded=False):
        uploads = st.file_uploader("上传平台订单导出 CSV（可多选）", type=["csv"], accept_multiple_files=True)
        order_meta = summarize_uploaded_orders(uploads) if uploads else {"files": [], "note": "No uploads"}
        if uploads:
            st.json(order_meta)

    st.subheader("Step 3｜生成深度分析报告（咨询级）")
    report_date = dt.datetime.now().strftime("%m/%d/%Y")

    platform_links = {
        "direct": direct_url.strip(),
        "uber_eats": uber_url.strip(),
        "doordash": doordash_url.strip(),
        "fantuan": fantuan_url.strip(),
        "hungrypanda": panda_url.strip(),
    }

    if st.button("生成报告内容", type="primary", disabled=not openai_key):
        inputs = ReportInputs(
            report_date=report_date,
            restaurant_cn=(restaurant_cn.strip() or restaurant_en.strip()),
            restaurant_en=restaurant_en.strip(),
            address=formatted_address.strip(),
            radius_miles=float(st.session_state.get("radius_miles", radius_miles)),
            platform_links=platform_links,
            competitors=competitors,
            competitor_menu_notes=comp_menu_notes.strip() if "comp_menu_notes" in locals() else "",
            order_upload_meta=order_meta,
            extra_business_context=extra_context.strip(),
        )

        prompt = build_prompt(place_details, inputs, competitor_places, st.session_state.get("acs_data", None))

        with st.spinner("正在生成咨询级报告（更深的商圈画像 + 菜单/定价/蓝海策略）..."):
            report_text = sanitize_text(openai_generate(prompt, openai_key, model=model))

        st.session_state["report_text"] = report_text
        st.session_state["report_inputs"] = inputs
        st.success("报告内容已生成。")

# -------------------------
# Preview + PDF
# -------------------------
report_text = st.session_state.get("report_text", "")
report_inputs: Optional[ReportInputs] = st.session_state.get("report_inputs", None)

if report_text and report_inputs:
    st.subheader("预览（可编辑）")
    edited = st.text_area("报告正文（你可以直接修改）", value=report_text, height=520)
    st.session_state["report_text"] = sanitize_text(edited)

    st.subheader("Step 4｜生成PDF报告")

    if not os.path.exists(BG_COVER):
        st.warning(f"封面背景图不存在：{BG_COVER}")
    if not os.path.exists(BG_CONTENT):
        st.warning(f"内容页背景图不存在：{BG_CONTENT}")

    if st.button("生成 PDF", type="primary"):
        with st.spinner("正在生成 PDF..."):
            pdf_path = render_pdf(st.session_state["report_text"], report_inputs)
        st.success("PDF 生成完成。")
        with open(pdf_path, "rb") as f:
            st.download_button(
                "下载 PDF",
                f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf"
            )
        st.caption(f"输出路径：{pdf_path}")
else:
    st.info("完成餐厅选择并生成报告后，这里会显示预览与 PDF 下载。")
