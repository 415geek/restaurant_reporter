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

# =========================================================
# Login (FIXED: no nested set_page_config; enforced at entry)
# =========================================================
def _get_allowed_passwords():
    pw_list = st.secrets.get("ADMIN_PASSWORDS", None)
    if pw_list and isinstance(pw_list, (list, tuple)) and len(pw_list) > 0:
        return [str(x) for x in pw_list]
    single = st.secrets.get("ADMIN_PASSWORD", "")
    return [str(single)] if single else []

def _secure_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))

def require_login():
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False
    if "auth_tries" not in st.session_state:
        st.session_state.auth_tries = 0
    if "auth_locked_until" not in st.session_state:
        st.session_state.auth_locked_until = 0.0

    if st.session_state.auth_ok:
        return True

    now = time.time()
    if now < st.session_state.auth_locked_until:
        wait_s = int(st.session_state.auth_locked_until - now)
        st.error(f"Too many attempts. Please try again in {wait_s}s.")
        st.stop()

    allowed = _get_allowed_passwords()
    if not allowed:
        st.error("Admin password not configured. Set ADMIN_PASSWORD or ADMIN_PASSWORDS in .streamlit/secrets.toml.")
        st.stop()

    st.markdown("## AuraInsight Access")
    st.caption("Enter the admin password to use this tool.")

    pw = st.text_input("Password", type="password")
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Login", type="primary"):
            ok = any(_secure_compare(pw, x) for x in allowed)
            if ok:
                st.session_state.auth_ok = True
                st.session_state.auth_tries = 0
                st.rerun()
            else:
                st.session_state.auth_tries += 1
                st.error("Invalid password.")
                if st.session_state.auth_tries >= 5:
                    st.session_state.auth_locked_until = time.time() + 60
                    st.session_state.auth_tries = 0
                    st.warning("Locked for 60 seconds.")
    with col2:
        if st.button("Clear"):
            st.session_state["__pw_clear__"] = True
            st.rerun()

    st.stop()

def logout_button():
    if st.button("Logout"):
        st.session_state.auth_ok = False
        st.session_state.auth_locked_until = 0.0
        st.session_state.auth_tries = 0
        st.rerun()

# ======================================
# Config
# ======================================
APP_TITLE = "AuraInsight Report Generator (Trade Area & Growth Diagnostic)"
OUTPUT_DIR = "output"
ASSETS_DIR = "assets"
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

BG_COVER_DEFAULT = os.path.join(ASSETS_DIR, "bg_cover.png")
BG_CONTENT_DEFAULT = os.path.join(ASSETS_DIR, "bg_content.png")

# Static fonts
FONT_NOTO_REG = os.path.join(FONTS_DIR, "NotoSansSC-Regular.ttf")
FONT_NOTO_BOLD = os.path.join(FONTS_DIR, "NotoSansSC-Bold.ttf")
FONT_ROBOTO_REG = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
FONT_ROBOTO_BOLD = os.path.join(FONTS_DIR, "Roboto-Bold.ttf")
FONT_ROBOTO_ITALIC = os.path.join(FONTS_DIR, "Roboto-Italic.ttf")

PAGE_W, PAGE_H = letter  # 612x792

# Footer / Branding
BUILT_BY_NAME = "Built by c8geek"
LINKEDIN_TEXT = "LinkedIn"
LINKEDIN_URL = "https://www.linkedin.com/in/maxwelllai/"  # change if needed

# Cover style: avoid duplicated title vs background
# "minimal" => only restaurant/date/address on cover
COVER_STYLE = "minimal"

# Content header: keep off by default to avoid overlap with background header design
DRAW_CONTENT_HEADER = False

# Typography
FONT_SIZE_BODY = 10
FONT_SIZE_H1 = 13
LINE_HEIGHT_CN = 16
LINE_HEIGHT_EN = 14
PARA_SPACING = 14

CONTENT_LEFT = 0.85 * inch
CONTENT_RIGHT = 0.85 * inch
CONTENT_TOP = PAGE_H - 1.05 * inch
CONTENT_BOTTOM = 1.15 * inch

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
    restaurant_menu_notes: str
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

def has_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)

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
        "x": lng, "y": lat,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json"
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    try:
        tract = data["result"]["geographies"]["Census Tracts"][0]
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
    params = {
        "get": get_vars,
        "for": f"tract:{tract}",
        "in": f"state:{state} county:{county}",
    }
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
    if occ_total > 0:
        out["pct_owner"] = (owner or 0) / occ_total
        out["pct_renter"] = (renter or 0) / occ_total
    else:
        out["pct_owner"] = None
        out["pct_renter"] = None

    return out

# ======================================
# OpenAI (Responses API)
# ======================================
def openai_generate(prompt: str, api_key: str, model: str) -> str:
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "input": prompt, "temperature": 0.30}
    r = requests.post(url, headers=headers, json=payload, timeout=180)
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
# Prompt Builder (Upgraded: SKU-level actions required)
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
    for cp in competitor_places[:10]:
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
            "approximation_note": "demographics use tract-level ACS near restaurant coordinate (proxy for trade area); must state assumptions"
        },
        "demographics_acs": acs or {"note": "ACS not available; propose data collection plan"},
        "platform_links": inputs.platform_links,
        "competitors_google": comp_brief,
        "competitors_user_input": inputs.competitors,
        "restaurant_menu_notes": inputs.restaurant_menu_notes,
        "competitor_menu_notes": inputs.competitor_menu_notes,
        "order_upload_meta": inputs.order_upload_meta,
        "extra_business_context": inputs.extra_business_context,
    }

    return f"""
You are AuraInsight's strategy consultant. Based on the input JSON, produce a consulting-grade “Trade Area & Growth Diagnostic Report” for a legacy Hong Kong-style restaurant in San Francisco,also prodece a specific deail step by step plan for improvement.

STRICT FORMAT RULES (failure if violated):
A) Do NOT output Markdown. Do not use: #, ##, **, |---|, ``` , []().
B) Use ONLY:
   - Section titles: 【Title】
   - Bullets: - text
   - Small tables: start with “Table:” and then output CSV lines (max 8 lines).
C) Each section must begin with 2–4 Key Takeaways. Prefer numbers.
D) Every recommendation must include:
   【Action】【Reason】【Expected Impact】【KPI】【2-week Test Plan】.
E) You MUST apply and explain: STP, JTBD, Menu Engineering (Stars/Cash Cows/Puzzles/Dogs), Anchoring, and Blue Ocean ERRC.
F) The report must include SKU-level pricing and bundle proposals. No vague advice.

Business context:
- Report Date: {inputs.report_date}
- Restaurant (CN): {inputs.restaurant_cn}
- Restaurant (EN): {inputs.restaurant_en}
- Address: {inputs.address}
- Delivery Radius: {inputs.radius_miles} miles
- Positioning: authentic Hong Kong cha chaan teng / bing sutt; 27-year legacy.

OUTPUT SECTIONS (exact order and exact titles):
【Executive Summary】
【1. Trade Area & Demographics】
【2. Customer Segments & JTBD】
【3. Platform Ecosystem Strategy】
【4. Competitive Landscape & Differentiation】
【5. Pricing, Anchors & Promo Economics】
【6. Menu Architecture & Menu Engineering】
【7. Operating Playbook & 30/60/90 Roadmap】
【8. SKU-level Menu, Pricing & Bundle Recommendations】
【Data Gaps & How to Collect】

SKU-LEVEL REQUIREMENTS (Section 8):
- Provide at least 12 concrete SKU recommendations.
- For EACH SKU output:
  - Item (CN/EN)
  - Current price by platform (if unknown, estimate using competitor medians and state “estimated”)
  - Recommended new price by platform (Uber/DD/Fantuan/Panda can differ)
  - Price logic (anchoring / psychological pricing / competitive parity / margin)
  - Bundle inclusion (yes/no) + exact bundle composition
  - Expected impact (AOV, conversion, margin proxy) with numeric range
  - 2-week experiment design (which platform first, what metric, stop rule)
- Also provide a “Bundle Matrix”:
  Table: BundleName, Includes, TargetCustomer, Price, PrimaryPlatform, UpsellPath

Virtual Brand requirement:
- Propose at least 1 virtual brand concept (“Wah Kee Bing Sutt / 华记冰室”), with:
  - Differentiated hero category
  - 8-item starter menu
  - Platform mapping and risk control (avoid cannibalization)

INPUT JSON:
{json.dumps(data_blob, ensure_ascii=False, indent=2)}
""".strip()

# ======================================
# PDF Rendering (FIXED: real width wrapping + spacing)
# ======================================
def draw_bg(c: canvas.Canvas, bg_path: str):
    if bg_path and os.path.exists(bg_path):
        c.drawImage(bg_path, 0, 0, width=PAGE_W, height=PAGE_H, mask="auto")

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

def wrap_by_width(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """
    Real wrap by measuring string width (ReportLab stringWidth).
    Supports CJK + EN mixed lines.
    """
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            out_lines.append("")
            continue

        # normalize spaces
        line = re.sub(r"\s+", " ", line).strip()

        words = []
        if has_cjk(line):
            # CJK: wrap by character but keep punctuation attached
            # Use a safe char iteration with basic grouping
            words = list(line)
        else:
            # EN: wrap by words
            words = line.split(" ")

        cur = ""
        for w in words:
            candidate = (cur + w) if has_cjk(line) else (w if cur == "" else cur + " " + w)
            if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
                cur = candidate
            else:
                if cur:
                    out_lines.append(cur)
                # if single token too wide, hard split
                if pdfmetrics.stringWidth(w, font_name, font_size) > max_width and not has_cjk(line):
                    chunks = textwrap.wrap(w, width=20, break_long_words=True)
                    for ch in chunks[:-1]:
                        out_lines.append(ch)
                    cur = chunks[-1] if chunks else ""
                else:
                    cur = w
        if cur:
            out_lines.append(cur)
    return out_lines

def draw_footer(c: canvas.Canvas, report_date: str, page_num: int):
    c.setFillColor(colors.HexColor("#7A7A7A"))
    c.setFont(f_en(False), 8)
    left_x = 0.75 * inch

    footer_text = f"Confidential | Generated by AuraInsight | {report_date} | {BUILT_BY_NAME}"
    c.drawString(left_x, 0.55 * inch, footer_text)

    # clickable LinkedIn
    link_w = pdfmetrics.stringWidth(LINKEDIN_TEXT, f_en(False), 8)
    link_x = PAGE_W - 0.75 * inch - link_w - 45
    c.drawString(link_x, 0.55 * inch, LINKEDIN_TEXT)
    c.linkURL(LINKEDIN_URL, (link_x, 0.50 * inch, link_x + link_w, 0.62 * inch), relative=0)

    c.drawRightString(PAGE_W - 0.75 * inch, 0.55 * inch, f"Page {page_num}")
    c.setFillColor(colors.black)

def render_pdf(
    report_text: str,
    inputs: ReportInputs,
    bg_cover: str,
    bg_content: str,
) -> str:
    register_aurainsight_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_name = "".join([ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in inputs.restaurant_en]).strip()
    safe_name = safe_name.replace(" ", "_") or "Restaurant"
    filename = f"AuraInsight_{safe_name}_{inputs.report_date.replace('/','-')}.pdf"
    out_path = os.path.join(OUTPUT_DIR, filename)

    c = canvas.Canvas(out_path, pagesize=letter)

    # ---------- Cover ----------
    draw_bg(c, bg_cover)

    # MINIMAL cover: avoid duplicated “AuraInsight/门店分析报告” if background already implies it
    c.setFillColor(colors.HexColor("#1F2A33"))
    if COVER_STYLE != "minimal":
        c.setFont(f_en(True), 26)
        c.drawCentredString(PAGE_W / 2, 315, "AuraInsight")
        c.setFillColor(colors.black)
        c.setFont(f_cn(True), 18)
        c.drawCentredString(PAGE_W / 2, 285, "【门店分析报告】")

    # Always show date + restaurant + address
    c.setFillColor(colors.HexColor("#333333"))
    c.setFont(f_en(False), 11)
    c.drawCentredString(PAGE_W / 2, 260, inputs.report_date)

    c.setFillColor(colors.black)
    c.setFont(f_cn(True), 16)
    c.drawCentredString(PAGE_W / 2, 165, inputs.restaurant_cn or inputs.restaurant_en)

    c.setFillColor(colors.HexColor("#333333"))
    c.setFont(f_en(False), 12)
    c.drawCentredString(PAGE_W / 2, 144, inputs.restaurant_en)

    c.setFont(f_en(False), 10)
    c.drawCentredString(PAGE_W / 2, 124, inputs.address)

    c.showPage()

    # ---------- Content pages ----------
    draw_bg(c, bg_content)

    page_num = 1
    x_left = CONTENT_LEFT
    x_right = PAGE_W - CONTENT_RIGHT
    max_width = x_right - x_left
    y = CONTENT_TOP

    def new_page():
        nonlocal y, page_num
        draw_footer(c, inputs.report_date, page_num)
        c.showPage()
        page_num += 1
        draw_bg(c, bg_content)
        y = CONTENT_TOP

    def draw_heading(title: str):
        nonlocal y
        if y < (CONTENT_BOTTOM + 40):
            new_page()
        c.setFillColor(colors.black)
        font = f_cn(True) if has_cjk(title) else f_en(True)
        c.setFont(font, FONT_SIZE_H1)
        c.drawString(x_left, y, title[:140])
        y -= 18

    def draw_paragraph(text: str):
        nonlocal y
        # Split by lines; for each line choose font and wrap by width
        for raw in text.splitlines():
            raw = raw.rstrip()
            if not raw.strip():
                y -= 10
                continue

            # Choose font: if line has CJK => Noto; else Roboto
            font = f_cn(False) if has_cjk(raw) else f_en(False)
            font_size = FONT_SIZE_BODY
            lines = wrap_by_width(raw, font, font_size, max_width)

            for line in lines:
                if y < CONTENT_BOTTOM:
                    new_page()
                c.setFont(font, font_size)
                c.setFillColor(colors.black)
                c.drawString(x_left, y, line)

                if has_cjk(line):
                    y -= LINE_HEIGHT_CN
                else:
                    y -= LINE_HEIGHT_EN

        y -= PARA_SPACING  # paragraph spacing

    sections = parse_sections(report_text)
    if not sections:
        draw_paragraph(report_text)
    else:
        for title, body in sections:
            draw_heading(title)
            if body:
                draw_paragraph(body)

    draw_footer(c, inputs.report_date, page_num)
    c.save()
    return out_path

# ======================================
# Order Upload (meta only)
# ======================================
def summarize_uploaded_orders(files: List[Any]) -> Dict[str, Any]:
    meta = {"files": [], "notes": "CSV schema summary for analysis."}
    for f in files:
        try:
            df = pd.read_csv(f)
            cols = list(df.columns)[:60]
            meta["files"].append({
                "name": getattr(f, "name", "uploaded.csv"),
                "rows": int(df.shape[0]),
                "cols_sample": cols,
                "date_col_guess": next((c for c in cols if "date" in c.lower() or "time" in c.lower()), None),
            })
        except Exception as e:
            meta["files"].append({
                "name": getattr(f, "name", "uploaded"),
                "error": str(e)[:200]
            })
    return meta

# ======================================
# Streamlit UI
# ======================================
st.set_page_config(page_title=APP_TITLE, layout="wide")
require_login()

st.title(APP_TITLE)

google_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
openai_key = st.secrets.get("OPENAI_API_KEY", "")

# Hardcode backgrounds (hide from sidebar as requested)
bg_cover = BG_COVER_DEFAULT
bg_content = BG_CONTENT_DEFAULT

with st.sidebar:
    st.header("Settings")
    model = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    radius_miles = st.slider("Trade area radius (miles)", 1.0, 6.0, 4.0, 0.5)
    nearby_radius_m = st.slider("Google Nearby radius (meters)", 300, 3000, 1200, 100)

    st.divider()
    st.caption("Brand")
    st.markdown(f"- {BUILT_BY_NAME}")
    st.markdown(f"- [{LINKEDIN_TEXT}]({LINKEDIN_URL})")
    logout_button()

if not google_key:
    st.warning("Missing GOOGLE_MAPS_API_KEY in .streamlit/secrets.toml")
if not openai_key:
    st.warning("Missing OPENAI_API_KEY in .streamlit/secrets.toml")

# Step 1
st.subheader("Step 1 — Address → Find nearby restaurants (Google)")
address_input = st.text_input("Address", value="2406 19th Ave, San Francisco, CA 94116")

if st.button("Search nearby restaurants", type="primary", disabled=not google_key):
    geo = google_geocode(address_input, google_key)
    if not geo:
        st.error("Unable to geocode address. Please enter a complete address.")
    else:
        lat, lng = geo
        places = google_nearby_restaurants(lat, lng, google_key, radius_m=nearby_radius_m)
        st.session_state["geo"] = (lat, lng)
        st.session_state["places"] = places
        st.success(f"Found {len(places)} restaurants.")

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

    selected_label = st.selectbox("Select restaurant", options)
    selected_place_id = id_map.get(selected_label)

    if st.button("Fetch Place Details", disabled=not google_key):
        details = google_place_details(selected_place_id, google_key) if selected_place_id else {}
        if not details:
            st.error("Failed to fetch details.")
        else:
            st.session_state["place_details"] = details
            st.success("Place details loaded.")

place_details = st.session_state.get("place_details", {})

# Step 2
if place_details:
    st.subheader("Step 2 — Trade area demographics + platform links + competitors + menu notes")

    # Coordinates
    rest_lat = None
    rest_lng = None
    try:
        loc = place_details.get("geometry", {}).get("location", {})
        rest_lat = float(loc.get("lat"))
        rest_lng = float(loc.get("lng"))
    except Exception:
        pass

    col1, col2 = st.columns([1, 1])

    with col1:
        restaurant_en = st.text_input("Restaurant (EN)", value=place_details.get("name", ""))
        restaurant_cn = st.text_input("Restaurant (CN, optional)", value="")
        formatted_address = st.text_input("Restaurant address", value=place_details.get("formatted_address", address_input))
        st.caption(f"Google rating: ⭐{place_details.get('rating','')} ({place_details.get('user_ratings_total','')} reviews)")

        extra_context = st.text_area(
            "Business context (optional)",
            value="Example: 27-year legacy HK-style cha chaan teng; current pain points; staffing; best sellers; margin constraints.",
            height=110
        )

    with col2:
        st.markdown("### Your platform links")
        direct_url = st.text_input("Direct / order.online", value="")
        uber_url = st.text_input("Uber Eats", value="")
        doordash_url = st.text_input("DoorDash", value="")
        fantuan_url = st.text_input("Fantuan", value="")
        panda_url = st.text_input("HungryPanda", value="")

    # ACS
    with st.expander("Auto-fetch Demographics (US Census ACS, tract proxy)", expanded=True):
        if rest_lat and rest_lng:
            if st.button("Fetch ACS profile"):
                tract_info = census_tract_from_latlng(rest_lat, rest_lng)
                if not tract_info:
                    st.warning("Unable to locate census tract.")
                else:
                    acs_data = acs_5y_profile(tract_info["STATE"], tract_info["COUNTY"], tract_info["TRACT"], year=2023)
                    st.session_state["tract_info"] = tract_info
                    st.session_state["acs_data"] = acs_data
                    st.success("ACS data loaded (tract-level proxy).")
        else:
            st.info("No coordinates available from Google Place Details.")

        acs_data = st.session_state.get("acs_data", None)
        if acs_data:
            pop = acs_data.get("pop_total")
            inc = acs_data.get("median_income")
            st.write({
                "ACS Year": acs_data.get("year"),
                "Geography": acs_data.get("name"),
                "Population (tract proxy)": None if pop is None else int(pop),
                "Median HH Income": None if inc is None else f"${int(inc):,}",
                "Median Age": acs_data.get("median_age"),
                "% Asian (proxy)": None if acs_data.get("pct_asian") is None else f"{acs_data.get('pct_asian')*100:.1f}%",
                "% Renter (proxy)": None if acs_data.get("pct_renter") is None else f"{acs_data.get('pct_renter')*100:.1f}%",
                "Note": "ACS tract is a proxy; report will clearly state assumptions for 3–4 mile trade area."
            })

    # Competitors + platform links
    st.markdown("### Competitors (add platform links for deeper pricing/menu differentiation)")
    default_comp = pd.DataFrame([
        {"name": "Smile House Cafe", "ubereats": "", "doordash": "", "fantuan": "", "hungrypanda": "", "website": "", "notes": ""},
        {"name": "凤凰聚会餐厅", "ubereats": "", "doordash": "", "fantuan": "", "hungrypanda": "", "website": "", "notes": ""},
        {"name": "大家乐餐厅", "ubereats": "", "doordash": "", "fantuan": "", "hungrypanda": "", "website": "", "notes": ""},
    ])
    comp_df = st.data_editor(default_comp, num_rows="dynamic", use_container_width=True, key="comp_editor")
    competitors = comp_df.fillna("").to_dict("records")

    competitor_places = st.session_state.get("competitor_places", [])
    if st.button("Fetch competitor Google profiles (optional)", disabled=not google_key):
        pulled = []
        for row in competitors[:10]:
            nm = (row.get("name") or "").strip()
            if not nm:
                continue
            pid = google_textsearch_place_id(f"{nm} San Francisco", google_key)
            if pid:
                pulled.append(google_place_details(pid, google_key))
        st.session_state["competitor_places"] = pulled
        competitor_places = pulled
        st.success(f"Loaded {len(pulled)} competitor profiles.")

    # Menu notes (KEY: to enable SKU-level recommendations)
    st.markdown("### Menu inputs (critical for SKU-level pricing & bundle proposals)")
    restaurant_menu_notes = st.text_area(
        "Your menu snapshot (paste top items + prices; any format)",
        value="",
        height=140
    )
    competitor_menu_notes = st.text_area(
        "Competitor menu snapshot (paste top items + prices / promos; any format)",
        value="",
        height=140
    )
    st.caption("Tip: If you paste 20–40 lines of menu+price for your store and competitors, the report becomes ‘hard’ and actionable.")

    # Upload optional order exports
    with st.expander("Upload order exports (CSV, optional)", expanded=False):
        uploads = st.file_uploader("Upload CSV exports (multi)", type=["csv"], accept_multiple_files=True)
        order_meta = summarize_uploaded_orders(uploads) if uploads else {"files": [], "note": "No uploads"}
        if uploads:
            st.json(order_meta)

    # Step 3
    st.subheader("Step 3 — Generate consulting-grade report")
    report_date = dt.datetime.now().strftime("%m/%d/%Y")

    platform_links = {
        "direct": direct_url.strip(),
        "uber_eats": uber_url.strip(),
        "doordash": doordash_url.strip(),
        "fantuan": fantuan_url.strip(),
        "hungrypanda": panda_url.strip(),
    }

    if st.button("Generate Report Text", type="primary", disabled=not openai_key):
        inputs = ReportInputs(
            report_date=report_date,
            restaurant_cn=(restaurant_cn.strip() or restaurant_en.strip()),
            restaurant_en=restaurant_en.strip(),
            address=formatted_address.strip(),
            radius_miles=radius_miles,
            platform_links=platform_links,
            competitors=competitors,
            restaurant_menu_notes=restaurant_menu_notes.strip(),
            competitor_menu_notes=competitor_menu_notes.strip(),
            order_upload_meta=order_meta,
            extra_business_context=extra_context.strip(),
        )

        prompt = build_prompt(
            place=place_details,
            inputs=inputs,
            competitor_places=competitor_places,
            acs=st.session_state.get("acs_data", None),
        )

        with st.spinner("Generating report (SKU-level pricing + bundles + platform strategy)…"):
            report_text = openai_generate(prompt, openai_key, model=model)
            report_text = sanitize_text(report_text)

        st.session_state["report_text"] = report_text
        st.session_state["report_inputs"] = inputs
        st.success("Report generated.")

# Preview + PDF
report_text = st.session_state.get("report_text", "")
report_inputs: Optional[ReportInputs] = st.session_state.get("report_inputs", None)

if report_text and report_inputs:
    st.subheader("Preview / Edit")
    edited = st.text_area("Report text (editable)", value=report_text, height=520)
    st.session_state["report_text"] = sanitize_text(edited)

    st.subheader("Step 4 — Generate PDF")
    if not os.path.exists(bg_cover):
        st.warning(f"Missing cover background: {bg_cover}")
    if not os.path.exists(bg_content):
        st.warning(f"Missing content background: {bg_content}")

    if st.button("Generate PDF", type="primary"):
        with st.spinner("Rendering PDF…"):
            pdf_path = render_pdf(
                report_text=st.session_state["report_text"],
                inputs=report_inputs,
                bg_cover=bg_cover,
                bg_content=bg_content,
            )
        st.success("PDF ready.")
        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
        st.caption(f"Saved to: {pdf_path}")
else:
    st.info("Select a restaurant and generate the report to preview and export PDF.")
