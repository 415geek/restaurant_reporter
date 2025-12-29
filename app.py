# app.py
# AuraInsight 报告生成器（Trade Area & Growth Diagnostic）
# NOTE: This is a single-file Streamlit app assembled from the code you provided,
# with Tab 3 replaced by the upgraded "粘贴/上传/链接" 菜单智能调整老板端体验。

import os
import re
import io
import json
import time
import math
import hmac
import base64
import datetime as dt
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# =========================================================
# Page Config (MUST be called once)
# =========================================================
APP_TITLE = "AuraInsight 报告生成器（Trade Area & Growth Diagnostic）"
st.set_page_config(page_title=APP_TITLE, layout="wide")

OUTPUT_DIR = "output"
ASSETS_DIR = "assets"
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

BG_COVER = os.path.join(ASSETS_DIR, "bg_cover.png")
BG_CONTENT = os.path.join(ASSETS_DIR, "bg_content.png")

FONT_NOTO_REG = os.path.join(FONTS_DIR, "NotoSansSC-Regular.ttf")
FONT_NOTO_BOLD = os.path.join(FONTS_DIR, "NotoSansSC-Bold.ttf")
FONT_NOTO_VAR = os.path.join(FONTS_DIR, "NotoSansSC-VariableFont_wght.ttf")

FONT_ROBOTO_REG = os.path.join(FONTS_DIR, "Roboto-Regular.ttf")
FONT_ROBOTO_BOLD = os.path.join(FONTS_DIR, "Roboto-Bold.ttf")
FONT_ROBOTO_ITALIC = os.path.join(FONTS_DIR, "Roboto-Italic.ttf")
FONT_ROBOTO_VAR = os.path.join(FONTS_DIR, "Roboto-VariableFont_wdth,wght.ttf")
FONT_ROBOTO_VAR_ITALIC = os.path.join(FONTS_DIR, "Roboto-Italic-VariableFont_wdth,wght.ttf")

PAGE_W, PAGE_H = letter


# =========================================================
# Auth
# =========================================================
def _get_allowed_passwords() -> List[str]:
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
        return

    now = time.time()
    if now < st.session_state.auth_locked_until:
        wait_s = int(st.session_state.auth_locked_until - now)
        st.error(f"尝试次数过多，请 {wait_s}s 后再试。")
        st.stop()

    allowed = _get_allowed_passwords()
    if not allowed:
        st.error("未配置管理员密码：请在 .streamlit/secrets.toml 设置 ADMIN_PASSWORD 或 ADMIN_PASSWORDS。")
        st.stop()

    st.title("AuraInsight 登录")
    st.caption("请输入管理员设置的密码后进入。")
    pw = st.text_input("密码", type="password")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("登录", type="primary"):
            ok = any(_secure_compare(pw, x) for x in allowed)
            if ok:
                st.session_state.auth_ok = True
                st.session_state.auth_tries = 0
                st.success("登录成功。")
                st.rerun()
            else:
                st.session_state.auth_tries += 1
                st.error("密码错误。")
                if st.session_state.auth_tries >= 5:
                    st.session_state.auth_locked_until = time.time() + 60
                    st.session_state.auth_tries = 0
                    st.warning("已暂时锁定 60 秒。")
    with c2:
        if st.button("清空"):
            st.rerun()

    st.stop()

def logout_button():
    if st.button("登出"):
        st.session_state.auth_ok = False
        st.session_state.auth_locked_until = 0.0
        st.session_state.auth_tries = 0
        st.rerun()


# =========================================================
# Models
# =========================================================
@dataclass
class CompetitorInput:
    name_or_address: str
    notes: str
    menu_files_meta: Dict[str, Any]
    google: Dict[str, Any]
    yelp: Dict[str, Any]

@dataclass
class ReportInputs:
    report_date: str
    restaurant_cn: str
    restaurant_en: str
    address: str
    radius_miles: float

    own_menu_meta: Dict[str, Any]
    orders_meta: Dict[str, Any]

    competitors: List[CompetitorInput]
    extra_business_context: str

    acs: Optional[Dict[str, Any]]
    tract_info: Optional[Dict[str, Any]]
    restaurant_google: Dict[str, Any]

    charts: Dict[str, bytes]


# =========================================================
# Fonts
# =========================================================
def register_fonts():
    if os.path.exists(FONT_NOTO_REG):
        pdfmetrics.registerFont(TTFont("Noto", FONT_NOTO_REG))
    elif os.path.exists(FONT_NOTO_VAR):
        pdfmetrics.registerFont(TTFont("Noto", FONT_NOTO_VAR))
    if os.path.exists(FONT_NOTO_BOLD):
        pdfmetrics.registerFont(TTFont("Noto-Bold", FONT_NOTO_BOLD))

    if os.path.exists(FONT_ROBOTO_REG):
        pdfmetrics.registerFont(TTFont("Roboto", FONT_ROBOTO_REG))
    elif os.path.exists(FONT_ROBOTO_VAR):
        pdfmetrics.registerFont(TTFont("Roboto", FONT_ROBOTO_VAR))
    if os.path.exists(FONT_ROBOTO_BOLD):
        pdfmetrics.registerFont(TTFont("Roboto-Bold", FONT_ROBOTO_BOLD))
    if os.path.exists(FONT_ROBOTO_ITALIC):
        pdfmetrics.registerFont(TTFont("Roboto-Italic", FONT_ROBOTO_ITALIC))
    elif os.path.exists(FONT_ROBOTO_VAR_ITALIC):
        pdfmetrics.registerFont(TTFont("Roboto-Italic", FONT_ROBOTO_VAR_ITALIC))

def f_cn(bold=False):
    if bold and "Noto-Bold" in pdfmetrics.getRegisteredFontNames():
        return "Noto-Bold"
    if "Noto" in pdfmetrics.getRegisteredFontNames():
        return "Noto"
    return "Helvetica-Bold" if bold else "Helvetica"

def f_en(bold=False, italic=False):
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


# =========================================================
# Utils
# =========================================================
def draw_bg(c: canvas.Canvas, bg_path: str):
    if bg_path and os.path.exists(bg_path):
        c.drawImage(bg_path, 0, 0, width=PAGE_W, height=PAGE_H, mask="auto")

def sanitize_text(text: str) -> str:
    text = text.replace("```", "").replace("`", "")
    text = text.replace("•", "-")
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.M)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    return text.strip()

def wrap_lines_by_pdf_width(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    """
    Wrap text to fit PDF width using reportlab stringWidth.
    - English: prefer space wrapping, but will hard-break very long tokens (urls/long words).
    - Chinese/No-space lines: char-by-char safe wrapping.
    """
    out: List[str] = []
    if not text:
        return out

    def _string_w(s: str) -> float:
        try:
            return pdfmetrics.stringWidth(s, font_name, font_size)
        except Exception:
            return len(s) * font_size * 0.55

    def _hard_break_token(token: str) -> List[str]:
        cur = ""
        lines = []
        for ch in token:
            cand = cur + ch
            if _string_w(cand) <= max_width:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    def _wrap_one_line(line: str) -> List[str]:
        line = line.rstrip("\n")
        if not line.strip():
            return [""]

        if _string_w(line) <= max_width:
            return [line]

        if " " in line:
            words = line.split(" ")
            cur = ""
            lines = []
            for w in words:
                candidate = (cur + " " + w).strip() if cur else w
                if _string_w(candidate) <= max_width:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                        cur = w
                    else:
                        lines.extend(_hard_break_token(w))
                        cur = ""
            if cur:
                lines.append(cur)
            return lines

        # no spaces -> char wrap
        chars = list(line)
        cur = ""
        lines = []
        for ch in chars:
            candidate = cur + ch
            if _string_w(candidate) <= max_width:
                cur = candidate
            else:
                if cur:
                    lines.append(cur)
                    cur = ch
                else:
                    lines.append(ch)
                    cur = ""
        if cur:
            lines.append(cur)
        return lines

    for para in text.splitlines():
        if not para.strip():
            out.append("")
            continue
        out.extend(_wrap_one_line(para))

    return out

def parse_sections(text: str) -> List[Tuple[str, str]]:
    text = text.strip()
    if not text:
        return []
    sections = []
    cur_title = None
    cur_body = []

    def flush():
        nonlocal cur_title, cur_body
        if cur_title is not None:
            sections.append((cur_title.strip(), "\n".join(cur_body).strip()))
        cur_title = None
        cur_body = []

    for ln in text.splitlines():
        ln_stripped = ln.strip()
        is_bracket = ln_stripped.startswith("【") and ln_stripped.endswith("】") and len(ln_stripped) >= 4
        is_num = bool(re.match(r'^\d+\.\s+\S+', ln_stripped))
        if is_bracket or is_num:
            flush()
            cur_title = ln_stripped.replace("【", "").replace("】", "") if is_bracket else ln_stripped
        else:
            cur_body.append(ln)

    flush()
    if len(sections) == 1 and sections[0][0] and sections[0][1] == "":
        return []
    return sections

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def df_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")


# =========================================================
# Google Places
# =========================================================
def google_geocode(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    r = requests.get(url, params={"address": address, "key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK" or not data.get("results"):
        return None
    loc = data["results"][0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

def google_nearby_places(lat: float, lng: float, api_key: str, radius_m: int = 1200, place_type: str = "restaurant") -> List[Dict[str, Any]]:
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    results: List[Dict[str, Any]] = []
    params = {"location": f"{lat},{lng}", "radius": radius_m, "type": place_type, "key": api_key}
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
        "name","formatted_address","rating","user_ratings_total","types",
        "url","website","formatted_phone_number","opening_hours","reviews","geometry"
    ])
    r = requests.get(url, params={"place_id": place_id, "fields": fields, "key": api_key}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
        return {}
    return data.get("result", {})


# =========================================================
# Census ACS
# =========================================================
def census_tract_from_latlng(lat: float, lng: float) -> Optional[Dict[str, str]]:
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {"x": lng, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current", "format": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    try:
        tract = data["result"]["geographies"]["Census Tracts"][0]
        return {"STATE": tract["STATE"], "COUNTY": tract["COUNTY"], "TRACT": tract["TRACT"], "NAME": tract.get("NAME","")}
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
        "hispanic": "B03003_003E",
        "owner_occ": "B25003_002E",
        "renter_occ": "B25003_003E",
    }
    get_vars = ",".join(["NAME"] + list(vars_map.values()))
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": get_vars, "for": f"tract:{tract}", "in": f"state:{state} county:{county}"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data or len(data) < 2:
        return None
    headers, values = data[0], data[1]
    row = dict(zip(headers, values))

    def to_num(x):
        try:
            return float(x)
        except Exception:
            return None

    out = {"year": year, "name": row.get("NAME",""), "state": state, "county": county, "tract": tract}
    for k, v in vars_map.items():
        out[k] = to_num(row.get(v))

    pop = out.get("pop_total") or 0.0
    if pop > 0:
        out["pct_asian"] = (out.get("asian") or 0.0) / pop
        out["pct_white"] = (out.get("white") or 0.0) / pop
        out["pct_black"] = (out.get("black") or 0.0) / pop
        out["pct_hispanic"] = (out.get("hispanic") or 0.0) / pop
    else:
        out["pct_asian"] = out["pct_white"] = out["pct_black"] = out["pct_hispanic"] = None

    owner = out.get("owner_occ") or 0.0
    renter = out.get("renter_occ") or 0.0
    occ_total = owner + renter
    out["pct_owner"] = (owner / occ_total) if occ_total > 0 else None
    out["pct_renter"] = (renter / occ_total) if occ_total > 0 else None
    return out


# =========================================================
# OpenAI Responses API (minimal wrapper)
# =========================================================
def openai_responses(api_key: str, payload: Dict[str, Any], timeout: int = 240) -> Dict[str, Any]:
    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def openai_text(prompt: str, api_key: str, model: str, temperature: float = 0.25) -> str:
    payload = {"model": model, "input": prompt, "temperature": temperature}
    data = openai_responses(api_key, payload, timeout=240)
    out = []
    for item in data.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                out.append(c.get("text", ""))
    return "\n".join(out).strip()

def _file_to_text_summary(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".txt"):
        try:
            return raw.decode("utf-8", errors="ignore")[:50000]
        except Exception:
            return str(raw[:2000])

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            return df.head(300).to_csv(index=False)[:50000]
        except Exception:
            uploaded_file.seek(0)
            return "CSV读取失败。"

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(uploaded_file)
            uploaded_file.seek(0)
            return df.head(300).to_csv(index=False)[:50000]
        except Exception:
            uploaded_file.seek(0)
            return "Excel读取失败。"

    return ""

def extract_menu_with_openai(files: List[Any], api_key: str, model: str, label: str) -> Dict[str, Any]:
    if not files:
        return {"label": label, "files": [], "extracted": {"note": "no files", "items": [], "promos": [], "notes": []}}

    extracted_items = []
    promos = []
    notes = []

    for f in files:
        fname = f.name
        lower = fname.lower()

        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            b = f.read()
            f.seek(0)
            b64 = base64.b64encode(b).decode("utf-8")
            mime = "image/png" if lower.endswith(".png") else "image/jpeg"
            data_url = f"data:{mime};base64,{b64}"

            prompt = (
                "你是餐厅外卖菜单解析器。请从菜单图片中识别：\n"
                "1) 菜品名称（尽量抓中英文）\n"
                "2) 价格（保留$或货币符号）\n"
                "3) 分类/栏目（如果能推断）\n"
                "4) 加价项/套餐结构/大小份（如有）\n"
                "5) 促销与营销文案（买一送一/满减/折扣/免配送等）\n"
                "只输出JSON，不要输出任何额外文字。\n"
                "JSON结构："
                "{\"items\":[{\"name\":\"\",\"price\":\"\",\"category\":\"\",\"notes\":\"\"}],"
                "\"promos\":[\"\"],\"platform_hints\":[\"\"],\"quality_flags\":[\"\"]}"
            )

            payload = {
                "model": model,
                "temperature": 0.2,
                "input": [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
            }
            try:
                resp = openai_responses(api_key, payload, timeout=240)
                text_out = ""
                for item in resp.get("output", []):
                    for c in item.get("content", []):
                        if c.get("type") == "output_text":
                            text_out += c.get("text", "")
                text_out = text_out.strip()

                m = re.search(r"\{.*\}", text_out, flags=re.S)
                if not m:
                    notes.append(f"{fname}: vision输出无法解析为JSON。")
                    continue
                obj = json.loads(m.group(0))
                extracted_items.extend(obj.get("items", []))
                promos.extend(obj.get("promos", []))
            except Exception as e:
                notes.append(f"{fname}: vision解析失败: {str(e)[:200]}")
            continue

        if lower.endswith((".txt", ".csv", ".xlsx", ".xls")):
            try:
                text_blob = _file_to_text_summary(f)
                f.seek(0)
            except Exception:
                text_blob = ""
                try:
                    f.seek(0)
                except Exception:
                    pass

            if not text_blob.strip():
                notes.append(f"{fname}: 无法提取文本内容。")
                continue

            prompt = (
                "你是餐厅外卖菜单解析器。以下是菜单文本（可能来自CSV/Excel/TXT）。\n"
                "提取：菜品、价格、分类、加价项/套餐结构、促销信息。\n"
                "只输出JSON，不要输出任何额外文字。\n"
                "JSON结构："
                "{\"items\":[{\"name\":\"\",\"price\":\"\",\"category\":\"\",\"notes\":\"\"}],"
                "\"promos\":[\"\"],\"platform_hints\":[\"\"],\"quality_flags\":[\"\"]}\n\n"
                f"菜单原文开始：\n{text_blob}\n菜单原文结束。"
            )
            try:
                text_out = openai_text(prompt, api_key, model=model, temperature=0.2)
                m = re.search(r"\{.*\}", text_out, flags=re.S)
                if not m:
                    notes.append(f"{fname}: 文本解析输出无法解析为JSON。")
                    continue
                obj = json.loads(m.group(0))
                extracted_items.extend(obj.get("items", []))
                promos.extend(obj.get("promos", []))
            except Exception as e:
                notes.append(f"{fname}: 文本解析失败: {str(e)[:200]}")
            continue

        notes.append(f"{fname}: 不支持的文件类型（建议 png/jpg/txt/csv/xlsx）。")

    extracted_items = extracted_items[:2000]
    promos = promos[:200]

    return {
        "label": label,
        "files": [{"name": f.name, "type": getattr(f, "type", "")} for f in files],
        "extracted": {
            "items": extracted_items,
            "promos": promos,
            "notes": notes[:120],
        }
    }


# =========================================================
# NEW: Pasted text intake + link intake helper
# =========================================================
def extract_menu_from_pasted_text(menu_text: str, api_key: str, model: str, label: str, platform_hint: str = "") -> Dict[str, Any]:
    """
    将用户粘贴的菜单文本，直接用 OpenAI 转成和 extract_menu_with_openai 一致的结构：
    {"label":..., "files":..., "extracted":{"items":[...], "promos":[...], "notes":[...]}}
    """
    menu_text = (menu_text or "").strip()
    if not menu_text:
        return {"label": label, "files": [], "extracted": {"note": "empty text", "items": [], "promos": [], "notes": ["empty menu_text"]}}

    prompt = (
        "你是餐厅外卖菜单解析器。下面是用户粘贴的菜单文本（可能混合中英文、带价格、带分类）。\n"
        "请提取：\n"
        "1) 菜品名称 name（尽量保留中英文原名）\n"
        "2) 价格 price（保留原货币符号或数值）\n"
        "3) 分类 category（如果没有就填空字符串）\n"
        "4) 备注 notes（如：辣度/大小份/套餐/加价项/招牌等）\n"
        "5) 促销 promos（如买一送一/折扣/满减等）\n\n"
        f"平台线索(platform_hint) = {platform_hint}\n"
        "只输出 JSON，不要输出任何额外文字。\n"
        "JSON结构："
        "{\"items\":[{\"name\":\"\",\"price\":\"\",\"category\":\"\",\"notes\":\"\"}],"
        "\"promos\":[\"\"],\"platform_hints\":[\"\"],\"quality_flags\":[\"\"]}\n\n"
        "菜单原文开始：\n"
        f"{menu_text[:65000]}\n"
        "菜单原文结束。"
    )

    notes = []
    try:
        text_out = openai_text(prompt, api_key, model=model, temperature=0.2)
        m = re.search(r"\{.*\}", text_out, flags=re.S)
        if not m:
            notes.append("粘贴文本解析：AI 输出无法解析为 JSON。")
            return {"label": label, "files": [], "extracted": {"note": "parse_failed", "items": [], "promos": [], "notes": notes}}

        obj = json.loads(m.group(0))
        items = obj.get("items", []) or []
        promos = obj.get("promos", []) or []
        items = items[:2000]
        promos = promos[:200]

        return {
            "label": label,
            "files": [{"name": "pasted_menu.txt", "type": "text/plain"}],
            "extracted": {
                "items": items,
                "promos": promos,
                "notes": notes + (obj.get("quality_flags", []) or []),
            }
        }
    except Exception as e:
        notes.append(f"粘贴文本解析失败: {str(e)[:200]}")
        return {"label": label, "files": [], "extracted": {"note": "exception", "items": [], "promos": [], "notes": notes}}


def build_link_intake_help(link: str) -> Dict[str, Any]:
    """
    链接入口不直接爬虫（稳定&合规），用于：
    - 识别平台
    - 指引用户如何从平台复制菜单文本/导出
    """
    link = (link or "").strip()
    s = link.lower()
    platform = "Unknown"
    if "doordash" in s:
        platform = "DoorDash"
    elif "ubereats" in s or "ubereat" in s:
        platform = "UberEats"
    elif "grubhub" in s:
        platform = "Grubhub"
    elif "fantuan" in s:
        platform = "Fantuan"
    elif "hungrypanda" in s or "panda" in s:
        platform = "Panda"

    tips = [
        f"识别平台：{platform}",
        "建议导入（任选其一）：",
        "A) 打开链接 → 在平台后台/菜单页把菜品列表复制成文本 → 粘贴到【方式1：粘贴菜单文本】",
        "B) 从平台后台导出菜单（CSV/Excel）→ 用【方式2：上传 Excel】上传",
        "C) 如果你能拿到截图（菜单页截图）→ 直接上传图片（png/jpg）也可识别"
    ]
    return {"platform": platform, "tips": tips}


# =========================================================
# Menu Stats + Charts
# =========================================================
def _to_price(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x)
    m = re.search(r'(\d+(\.\d+)?)', s.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def menu_to_df(menu_meta: Dict[str, Any]) -> pd.DataFrame:
    items = (((menu_meta or {}).get("extracted") or {}).get("items") or [])
    rows = []
    for it in items:
        name = (it.get("name") or "").strip()
        cat = (it.get("category") or "").strip() or "Uncategorized"
        price = _to_price(it.get("price"))
        notes = (it.get("notes") or "").strip()
        if not name and price is None:
            continue
        rows.append({"name": name, "category": cat, "price": price, "notes": notes})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["category"] = df["category"].fillna("Uncategorized")
    return df

def make_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def build_charts(own_df: pd.DataFrame, comps: List[Tuple[str, pd.DataFrame]]) -> Dict[str, bytes]:
    charts: Dict[str, bytes] = {}
    if own_df is None or own_df.empty:
        return charts

    fig = plt.figure()
    own_df["price"].dropna().plot(kind="hist", bins=18)
    plt.title("Own Menu Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    charts["chart_own_price_hist"] = make_png(fig)

    fig = plt.figure()
    own_df["category"].value_counts().head(12).plot(kind="bar")
    plt.title("Own Menu Category Mix (Top 12)")
    plt.xlabel("Category")
    plt.ylabel("Items")
    charts["chart_own_category_bar"] = make_png(fig)

    bins = [0, 8, 12, 16, 20, 25, 35, 999]
    labels = ["<$8", "$8-12", "$12-16", "$16-20", "$20-25", "$25-35", "$35+"]
    tier = pd.cut(own_df["price"], bins=bins, labels=labels, include_lowest=True)
    fig = plt.figure()
    tier.value_counts().reindex(labels).fillna(0).plot(kind="bar")
    plt.title("Own Menu Price Tiers")
    plt.xlabel("Tier")
    plt.ylabel("Items")
    charts["chart_own_price_tiers"] = make_png(fig)

    comp_rows = []
    for name, dfc in comps:
        if dfc is not None and not dfc.empty and dfc["price"].dropna().shape[0] >= 5:
            comp_rows.append({"competitor": name, "median_price": float(dfc["price"].median())})
    if comp_rows:
        comp_df = pd.DataFrame(comp_rows).sort_values("median_price")
        fig = plt.figure()
        comp_df.set_index("competitor")["median_price"].plot(kind="bar")
        plt.title("Competitor Median Price (from uploaded menus)")
        plt.xlabel("Competitor")
        plt.ylabel("Median Price")
        charts["chart_comp_median_price"] = make_png(fig)

    return charts


# =========================================================
# Menu Optimizer
# =========================================================
def summarize_market_prices(own_df: pd.DataFrame, comp_dfs: List[pd.DataFrame]) -> Dict[str, Any]:
    all_comp = pd.concat([df for df in comp_dfs if df is not None and not df.empty], ignore_index=True) if comp_dfs else pd.DataFrame()
    out = {"own": {}, "market": {}, "notes": []}

    if own_df is None or own_df.empty:
        out["notes"].append("own_df empty")
        return out

    own_prices = own_df["price"].dropna()
    out["own"]["median"] = float(own_prices.median()) if not own_prices.empty else None
    out["own"]["p25"] = float(own_prices.quantile(0.25)) if not own_prices.empty else None
    out["own"]["p75"] = float(own_prices.quantile(0.75)) if not own_prices.empty else None
    out["own"]["count_price"] = int(own_prices.shape[0])

    if all_comp.empty or all_comp["price"].dropna().empty:
        out["notes"].append("no competitor price data")
        return out

    mkt_prices = all_comp["price"].dropna()
    out["market"]["median"] = float(mkt_prices.median())
    out["market"]["p25"] = float(mkt_prices.quantile(0.25))
    out["market"]["p75"] = float(mkt_prices.quantile(0.75))
    out["market"]["count_price"] = int(mkt_prices.shape[0])

    cat_med = (all_comp.dropna(subset=["price"])
               .groupby("category")["price"].median()
               .sort_values(ascending=False))
    out["market"]["category_median_top"] = cat_med.head(20).to_dict()

    return out

def build_menu_optimizer_prompt(
    restaurant_name: str,
    address: str,
    season_hint: str,
    own_menu_meta: Dict[str, Any],
    competitor_menu_metas: List[Dict[str, Any]],
    market_summary: Dict[str, Any],
    lang: str,
    objective_mode: str,
    commission_rate: float,
    price_layering: bool,
) -> str:
    blob = {
        "restaurant_name": restaurant_name,
        "address": address,
        "season_hint": season_hint,
        "objective_mode": objective_mode,
        "platform_commission_rate": commission_rate,
        "price_layering": price_layering,
        "own_menu": own_menu_meta,
        "competitors": competitor_menu_metas,
        "market_summary": market_summary,
        "pricing_rules": {
            "psych": "用心理定价(9/9.5/8.9/18.95/19.95等) + 价格锚点(引流/主推/高毛利/形象款)构建价格梯度",
            "step_c": "若无法识别套餐或整体偏高(对比market_summary)，先把价格调到合理价带，再组套餐，最后生成完整菜单",
            "bundle_policy": "套餐必须有清晰节省感(省$2-$5)；要明确适用菜品、限制规则、不可叠加；给出可执行文案",
            "commission": "外卖价必须考虑平台抽佣，避免净到手过低；堂食价可以更亲民，外卖价用于覆盖抽佣成本",
            "objective": "若objective_mode=acquisition，则要给出1-2个引流爆款(低毛利高转化)+1个高毛利补贴；若profit则提高高毛利主推占比并减少过激折扣"
        },
        "output_schema": {
            "rows": [
                {
                    "section": "Chef_Signature / Best_Sellers / Value_Combos / Drinks / Classic_Menu ...",
                    "name_cn": "",
                    "name_en": "",
                    "name": "",
                    "raw_category": "",
                    "price_dine_in": 0.0,
                    "price_delivery": 0.0,
                    "price_old": None,
                    "commission_rate": 0.0,
                    "commission_fee_est": 0.0,
                    "net_after_commission_est": 0.0,
                    "notes": ""
                }
            ]
        }
    }

    if lang == "English":
        instruction = (
            "You are a US delivery-market menu engineering expert.\n"
            "Task: rebuild a FULL menu structure from extracted items.\n"
            "Critical Step C: If bundles are missing OR overall pricing is overpriced vs market_summary, "
            "first adjust to a reasonable price band using anchor & psychological pricing, then create bundles, "
            "then output the final full menu.\n"
            "You MUST incorporate competitor menus + market_summary.\n"
            "Commission-aware: delivery price MUST account for platform commission_rate.\n"
            "If price_layering=true, output both dine-in and delivery prices; otherwise set them equal.\n"
            "Return JSON ONLY. No extra text.\n"
        )
    else:
        instruction = (
            "你是北美外卖市场的菜单工程专家。\n"
            "任务：根据门店菜单 + 竞对菜单 + market_summary（同行价带），重做一份可直接上架的完整菜单结构。\n"
            "关键 Step C：如果没识别到套餐/或整体定价偏高（对比 market_summary），必须先调价到合理区间（锚点+心理定价），再组套餐，最后输出完整菜单。\n"
            "必须考虑平台抽佣：外卖价要覆盖抽佣，避免净到手过低。\n"
            "若 price_layering=true：输出堂食价+外卖价；否则两者相同。\n"
            "只输出JSON，不要输出任何额外文字。\n"
        )

    return f"{instruction}\n输入JSON：\n{json.dumps(blob, ensure_ascii=False, indent=2)}"

def generate_optimized_menu_df(
    api_key: str,
    model: str,
    restaurant_name: str,
    address: str,
    own_menu_meta: Dict[str, Any],
    competitor_menu_metas: List[Dict[str, Any]],
    market_summary: Dict[str, Any],
    lang: str,
    objective_mode: str,
    commission_rate: float,
    price_layering: bool,
) -> pd.DataFrame:
    season_hint = "按北美当前季节与下个季节给上新建议，并体现到菜单分区或命名中。"
    prompt = build_menu_optimizer_prompt(
        restaurant_name=restaurant_name,
        address=address,
        season_hint=season_hint,
        own_menu_meta=own_menu_meta,
        competitor_menu_metas=competitor_menu_metas,
        market_summary=market_summary,
        lang=lang,
        objective_mode=objective_mode,
        commission_rate=commission_rate,
        price_layering=price_layering,
    )
    text_out = openai_text(prompt, api_key, model=model, temperature=0.2)

    m = re.search(r"\[.*\]", text_out, flags=re.S)
    if not m:
        m = re.search(r"\{.*\}", text_out, flags=re.S)
    if not m:
        raise ValueError("AI 输出无法解析为 JSON。")

    obj = json.loads(m.group(0))
    rows = obj.get("rows", obj if isinstance(obj, list) else [])
    df = pd.DataFrame(rows)

    cols = [
        "section","name_cn","name_en","name","raw_category",
        "price_dine_in","price_delivery","price_old",
        "commission_rate","commission_fee_est","net_after_commission_est",
        "notes"
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None

    df["price_dine_in"] = pd.to_numeric(df["price_dine_in"], errors="coerce")
    df["price_delivery"] = pd.to_numeric(df["price_delivery"], errors="coerce")
    df["commission_rate"] = pd.to_numeric(df["commission_rate"], errors="coerce")
    df["commission_fee_est"] = pd.to_numeric(df["commission_fee_est"], errors="coerce")
    df["net_after_commission_est"] = pd.to_numeric(df["net_after_commission_est"], errors="coerce")

    return df


# =========================================================
# Orders Meta
# =========================================================
def summarize_orders(files: List[Any]) -> Dict[str, Any]:
    meta = {"files": [], "note": "No uploads"}
    if not files:
        return meta
    meta["note"] = "Uploaded"
    for f in files:
        try:
            df = pd.read_csv(f)
            cols = list(df.columns)[:80]
            meta["files"].append({
                "name": getattr(f, "name", "orders.csv"),
                "rows": int(df.shape[0]),
                "cols_sample": cols,
                "date_col_guess": next((c for c in cols if "date" in c.lower() or "time" in c.lower()), None),
                "amount_col_guess": next((c for c in cols if "total" in c.lower() or "amount" in c.lower() or "subtotal" in c.lower()), None),
            })
        except Exception as e:
            meta["files"].append({"name": getattr(f, "name", "orders"), "error": str(e)[:200]})
    return meta


# =========================================================
# Prompt (餐饮业态分析报告) + language switch
# =========================================================
RESTAURANT_SYSTEM_STYLE = """
你是一个餐饮行业的数据驱动增长咨询顾问，擅长对餐饮门店与餐饮品牌进行商圈诊断、平台与堂食结构分析、菜单与价格优化、虚拟品牌设计，并输出可落地执行的增长方案。报告风格参考专业餐饮咨询公司的 trade area & growth diagnostic 报告，有深度、有数据支撑，并给出具体、可拆解的执行步骤与 KPI。

请严格围绕「餐饮门店/餐饮品牌」输出报告，按照以下结构与逻辑编写（可调整小标题措辞，但保留模块和思路）：

0. 报告基本信息与摘要（Executive Summary）
- 说明：门店/品牌名称、地址或城市、业态（如：港式茶餐厅、中餐快餐、火锅、咖啡馆、甜品、轻食等）、主要就餐场景（堂食/外卖/自提比例）。
- 用 3–6 句给出高层摘要：
  - 所处商圈类型：例如“稳定型社区商圈”“办公室/商务型商圈”“学校/学生型商圈”“景区/高流动商圈”等。
  - 1–3 个结构性优势（如：老字号品牌资产、多平台覆盖、厨房产能充足等）。
  - 1–3 个关键问题/增长瓶颈（如：线上结构未最大化、价格带与客群错位、某时段空档严重等）。

1. 商圈与客群结构（Trade Area & Demand Fundamentals）
1.1 商圈界定与人口结构
- 说明门店核心商圈半径（例如：社区店 3–4 英里，商圈步行 10–15 分钟等），并结合可用信息描述：
  - 常住人口大致区间(根据google数据做参考）；
  - 族裔结构、年龄层分布；
  - 家庭住户占比 vs 单身/学生占比；
  - 人口流动性（低/中/高）及其原因。
1.2 餐饮消费行为特征
- 判断：该商圈是高流量即食型（路过客流为主），还是信任驱动型（熟客复购为主），或者混合型。说明判断依据。
- 说明影响复购的关键变量：例如品牌经营年限、出品稳定性、口味正宗度、家庭适配度、价格敏感度等。
- 要求：每个判断后，用 1–2 句说明“哪些数据或事实支撑这个判断”，而不是空泛描述。

2. 门店与品牌资产结构（Store & Brand Assets）
2.1 门店基本盘
- 门店开业年限、历史定位（如：社区老字号、新概念品牌、网红店等）。
- 当前客源构成：附近居民、周边公司白领、学生、游客等的大致比例。
2.2 品牌信任与复购结构
- 说明：在核心客群中的信任度与口碑特征，是“老字号高信任高复购”，还是“新店处于试错期”。
- 判断该门店处于哪个阶段：
  - “结构正确但效率未最大化”；
  - “新店需要快速验证定位与客群”；
  - “老品牌线上化/多平台化转型期”。

3. 渠道与平台生态（Dine-in, Takeout & Platform Ecosystem）
3.1 渠道构成概览
- 拆解：堂食；自提（电话/官网/Google）；第三方外卖平台（DoorDash/UberEats/Grubhub/饭团/HungryPanda 等）。
- 对每个渠道给出：订单占比大致区间、人均消费估计、典型用户类型与消费动机。
3.2 平台角色拆解（Role-based View）
- 对每一个平台说明：商圈特征+平台主要客群；用户动机；战略意义（利润锚点/新客入口/订单基盘/活跃度来源）。
- 强调：“多平台结构不是分散，而是分工”。给出判断：当前结构是“基础结构正确但效率未最大化”还是“过度依赖单一平台存在风险”。

4. 竞争格局与相对位置（Competitive Landscape）
4.1 竞品选择与对比
- 选出 3–5 家主要竞品（同品类、同价格带、同平台、同商圈）。
- 用表格对比：经营年限、平台覆盖、品牌信任资产、用户结构、价格带与主打菜、核心风险点。
4.2 关键洞察
- 说明竞争本质不在“菜系”，而在“平台结构与运营结构是否更优”。明确本店优势与短板各 2–3 条。

5. 价格带与订单经济（Pricing & Order Economics）
5.1 有效成交价格带：总结高频薄利区/主流成交区/高价可接受但需价值支撑/过高阻力区。
5.2 价格敏感机制：强调价格变化的“解释空间”。
5.3 优化方向：主攻区间、引流/高毛利/套餐化菜品、平台差异化标价策略。

6. 时段与场景需求结构（Time-based & Occasion-based Demand）
6.1 时段贡献：早餐/午餐/下午茶/晚餐/夜宵，订单与收入贡献、客单差异、用户类型差异。
6.2 场景适配：工作日vs周末、堂食vs外卖、自提vs平台；识别放量场景/利润场景/获客心智场景。

7. 菜单架构与虚拟品牌策略（Menu Architecture & Virtual Brand）
7.1 平台区分菜单架构：不同平台不能完全同菜单；给出 SKU 数量、主打品类、活动策略、定价策略建议。
7.2 虚拟品牌/子品牌：给 1–2 个方向；明确平台优先级、核心菜品组合与价格带、与主品牌区隔逻辑。
7.3 虚拟品牌 KPI：订单占比、新客占比、对主品牌干扰、差评机制的纠偏动作。

8. 战略结论与执行路线图（Strategic Implications & Execution Roadmap）
8.1 关键战略判断：3–6 条有力度判断句。
8.2 3–5 条核心增长举措：适用对象、动作内容、量化目标（转化/客单/时段收入占比/平台占比调整）。
8.3 执行节奏与监测机制：3–6 个月节奏；平台漏斗、品类结构、时段结构；预警信号与触发调整。

输出风格要求
- 必须分章节分小节，结构类似专业「门店分析 & 增长诊断」报告。
- 多用表格展示：平台角色对比、竞品结构对比、价格带 vs 成交表现、时段 vs 收入贡献。
- 语言：简洁、有判断力，少空洞形容词；多用“结构/分工/分层/杠杆/解释空间/订单基盘/利润锚点/存在感”等术语。
- 所有建议尽量落到可执行细节：谁执行、在哪个平台/时段执行、做什么动作、期望看到哪些数据变化。
""".strip()

def build_prompt(inputs: ReportInputs, lang: str) -> str:
    blob = {
        "report_date": inputs.report_date,
        "restaurant": {
            "cn": inputs.restaurant_cn,
            "en": inputs.restaurant_en,
            "address": inputs.address,
            "radius_miles": inputs.radius_miles,
            "google": inputs.restaurant_google,
        },
        "trade_area": {
            "tract_info": inputs.tract_info,
            "acs": inputs.acs,
            "assumption_note": "ACS 为 tract 级别代理，作为商圈画像的方向性参考；报告必须明确这个假设并给出风险提示。"
        },
        "own_menu": inputs.own_menu_meta,
        "orders_meta": inputs.orders_meta,
        "competitors": [
            {
                "name_or_address": c.name_or_address,
                "notes": c.notes,
                "google": c.google,
                "yelp": c.yelp,
                "menu": c.menu_files_meta,
            } for c in inputs.competitors
        ],
        "extra_business_context": inputs.extra_business_context,
        "charts_available": list(inputs.charts.keys()),
        "current_date": dt.datetime.now().strftime("%Y-%m-%d"),
    }

    if lang == "English":
        lang_rule = "Output language MUST be English."
    else:
        lang_rule = "输出语言必须是中文（简体），除非专有名词/菜名需要英文。"

    return f"""
{RESTAURANT_SYSTEM_STYLE}

You are AuraInsight's restaurant growth consultant.
{lang_rule}

Hard requirements:
1) Do NOT output Markdown.
2) Headings must be numbered like "0. ...", "1. ...", "1.1 ...".
3) Each major chapter starts with 3-6 Key Takeaways.
4) Every recommendation must include: Action, Reason, Expected Impact, KPI, 2-week Validation method.
5) Must interpret charts by name (chart_own_price_hist, chart_own_category_bar, chart_own_price_tiers, chart_comp_median_price) when available.
6) Must include Data Gaps & How to Collect (as an explicit section).
7) Must include tables where appropriate (use plain text tables, not Markdown).

Input JSON:
{json.dumps(blob, ensure_ascii=False, indent=2)}

Start writing now:
""".strip()

def ensure_long_enough(report_text: str, api_key: str, model: str, lang: str, min_chars: int = 16000) -> str:
    t = sanitize_text(report_text)
    if len(t) >= min_chars:
        return t

    if lang == "English":
        expand_prompt = f"""
You will receive a report. Expand it significantly without changing chapter order.
No Markdown. Output full report.
Original:
{t}
""".strip()
    else:
        expand_prompt = f"""
