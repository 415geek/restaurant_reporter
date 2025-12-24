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

def wrap_lines_by_chars(text: str, max_chars: int) -> List[str]:
    lines: List[str] = []
    for para in text.splitlines():
        para = para.rstrip()
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=max_chars, break_long_words=False, replace_whitespace=False))
    return lines

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
# OpenAI Responses API
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
# Menu Optimizer (anchor + market + bundles + commission + price layers)
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
  - 常住人口大致区间；
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
你将收到一份报告正文。请在不改变章节标题顺序的前提下，显著扩写，使其更细、更能执行。
严禁输出Markdown。只输出完整正文（包含所有章节）。
原文开始：
{t}
原文结束。
""".strip()

    try:
        t2 = openai_text(expand_prompt, api_key, model=model, temperature=0.25)
        t2 = sanitize_text(t2)
        if len(t2) > len(t):
            t = t2
    except Exception:
        pass
    return t


# =========================================================
# Prompt (多业态快速诊断报告)
# =========================================================
MULTI_SYSTEM_STYLE = """
你是一个数据驱动的商业分析与增长咨询顾问，擅长把定性洞察和定量数据结合，输出结构清晰、逻辑严谨、可执行的业务诊断与增长方案报告。报告风格参考专业咨询公司（如 trade area & growth diagnostic 形式），要求有深度、有数据支撑、有清晰的执行路径和量化目标。

请按以下原则和结构输出报告（可根据业务类型微调章节标题，但必须保留这些逻辑模块）：

0. 报告基本信息
- 简要说明：业务名称、行业/品类、所在城市/区域、分析时间范围。
- 用 3–5 句给出高层摘要（Executive Summary）：
  - 当前业务所处的结构性位置（例如：稳定型社区商圈、高流动商旅型、线上获客型、平台依赖型等）。
  - 1–3 个关键结构性优势。
  - 1–3 个关键增长瓶颈。

1. 需求与客群结构（Trade Area / Customer Fundamentals）
1.1 区域/目标客群画像
- 结合可用数据描述目标客群，使用“区间 + 相对占比”表达。
1.2 消费/使用行为特征
- 判断业务属于：高流量冲动型/低流动高复购型/项目制决策型/长期合同型等。
- 每个判断后必须给出 1–2 句支撑依据（数据或事实），写出因果逻辑。

2. 品牌与渠道/平台结构（Brand & Channel Ecosystem）
2.1 品牌资产与信任结构：经营年限、认知度、口碑、复购/续约趋势；判断阶段（新品牌试错/老品牌效率优化）。
2.2 渠道/平台角色拆解（Role-based View）
- 列出主要获客与成交渠道，并说明：用户动机、战略意义（利润锚点/新客入口/曝光入口/订单基盘/活跃度来源）。
- 强调“多渠道不是分散，而是分工”，给出结构判断并说明理由。

3. 竞争格局与相对位置（Competitive Landscape）
3.1 主要竞对列表：3–5 个典型竞品（同城同品类/同平台垂类/同客群替代）。
- 建议用表格对比：经营年限、渠道覆盖、品牌信任、价格带、目标客群结构、优势与风险点。
3.2 关键洞察：竞争本质是“结构更优”，明确自身优势与短板各 2–3 点。

4. 价格 / 产品结构与单笔经济（Pricing, Offering & Unit Economics）
4.1 有效成交区间：多个价格带；评估成交频次、利润水平、对应客群。
4.2 产品/方案结构：是否与支付意愿匹配；强调价格“解释空间”。
4.3 竞对价格带监测：设计简单雷达机制，保持核心产品处于“可解释区间”。

5. 时间 / 场景维度需求结构（Time / Scenario-based Demand）
5.1 时间维度：一天/一周/项目周期；哪些时段决定规模/利润弹性。
5.2 场景维度：工作日vs周末、线上vs线下、首次vs复购、试点vs全量 rollout；标注各场景战略意义。

6. 战略结论（Strategic Implications）
- 3–5 条有力度的判断句：阶段、增长来源“不是X而是Y”、哪些资产尚未转译为增长杠杆；每条判断要解释逻辑来源。

7. 增长举措与执行路线图（Strategic Initiatives & Execution Roadmap）
7.1 3–5 条核心增长策略：适用渠道/客群/场景；动作逻辑；预期影响。
7.2 渠道分工下的产品/内容架构：不同渠道按动机分层设计；给出清晰结构建议。
7.3 量化 KPI 与监测节奏：给出可量化目标 + 监测周期 + 预警信号。
7.4 预期整体影响：3–6个月、6–12个月在收入/盈利/风险分散/品牌资产放大上的影响。

输出风格与格式要求
- 报告必须分章节、分小节，使用清晰标题。
- 尽量使用表格展示对比（竞品、渠道角色、价格带 vs 成交表现）。
- 语言：简洁、有判断力，避免空洞形容词；多用“结构/逻辑/角色/分层/杠杆/解释空间”等概念。
- 每条建议尽量落到：谁来做、在哪个渠道/场景做、做什么动作、期望看到哪些数据变化。
""".strip()

def build_quick_prompt(place_details: Dict[str, Any],
                       acs_data: Optional[Dict[str, Any]],
                       tract_info: Optional[Dict[str, Any]],
                       extra_context: str,
                       report_date: str,
                       lang: str) -> str:
    blob = {
        "report_date": report_date,
        "business": {
            "name": place_details.get("name", ""),
            "address": place_details.get("formatted_address", ""),
            "types": place_details.get("types", []),
            "rating": place_details.get("rating", None),
            "user_ratings_total": place_details.get("user_ratings_total", None),
            "phone": place_details.get("formatted_phone_number", ""),
            "website": place_details.get("website", ""),
            "google_url": place_details.get("url", ""),
            "opening_hours": place_details.get("opening_hours", {}),
            "reviews_sample": (place_details.get("reviews", []) or [])[:6],
            "geo": (place_details.get("geometry", {}) or {}).get("location", {}),
        },
        "trade_area": {
            "tract_info": tract_info,
            "acs": acs_data,
            "assumption_note": "ACS 为 tract 级别代理，作为区域画像方向性参考。"
        },
        "extra_business_context": extra_context.strip(),
        "current_date": dt.datetime.now().strftime("%Y-%m-%d"),
    }

    if lang == "English":
        lang_rule = "Output language MUST be English."
    else:
        lang_rule = "输出语言必须是中文（简体），除非专有名词需要英文。"

    return f"""
{MULTI_SYSTEM_STYLE}

{lang_rule}

Hard requirements:
1) Do NOT output Markdown.
2) Headings must be numbered like "0. ...", "1. ...", "1.1 ...".
3) Each major chapter starts with 3-5 Key Takeaways.
4) Every recommendation must include: Action, Reason, Expected Impact, KPI, 2-week Validation method.
5) Must include explicit section: Data Gaps & How to Collect.
6) Must include at least 2 plain text tables (not Markdown).

Input JSON:
{json.dumps(blob, ensure_ascii=False, indent=2)}

Start writing now:
""".strip()


# =========================================================
# PDF Render (safer)
# =========================================================
def draw_footer(c: canvas.Canvas, report_date: str, page_num: int):
    c.setFillColor(colors.HexColor("#7A7A7A"))
    c.setFont(f_en(False), 8)
    c.drawString(0.75 * inch, 0.55 * inch, f"Confidential | Generated by AuraInsight | {report_date}")
    c.drawRightString(PAGE_W - 0.75 * inch, 0.55 * inch, f"Page {page_num}")
    c.setFillColor(colors.black)

def render_pdf(report_text: str, inputs: ReportInputs) -> str:
    register_fonts()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    safe_name = "".join([ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in inputs.restaurant_en]).strip()
    safe_name = safe_name.replace(" ", "_") or "Report"
    filename = f"AuraInsight_{safe_name}_{inputs.report_date.replace('/','-')}.pdf"
    out_path = os.path.join(OUTPUT_DIR, filename)

    c = canvas.Canvas(out_path, pagesize=letter)

    # Cover
    draw_bg(c, BG_COVER)
    c.setFillColor(colors.HexColor("#111111"))

    y_base = 210
    c.setFont(f_cn(True), 16)
    c.drawCentredString(PAGE_W / 2, y_base, inputs.restaurant_cn or inputs.restaurant_en)
    c.setFont(f_en(False), 12)
    c.setFillColor(colors.HexColor("#333333"))
    c.drawCentredString(PAGE_W / 2, y_base - 22, inputs.restaurant_en)
    c.setFont(f_en(False), 10)
    c.drawCentredString(PAGE_W / 2, y_base - 42, inputs.address)
    c.setFont(f_en(False), 10)
    c.drawCentredString(PAGE_W / 2, y_base - 62, inputs.report_date)
    c.showPage()

    # Content pages
    page_num = 1
    left = 0.90 * inch
    top = PAGE_H - 1.55 * inch
    y = top - 0.45 * inch

    body_font_size = 10
    line_gap = 14
    para_gap = 10
    heading_gap = 18
    bottom_margin = 1.35 * inch

    def new_page():
        nonlocal y, page_num
        draw_footer(c, inputs.report_date, page_num)
        c.showPage()
        page_num += 1
        draw_bg(c, BG_CONTENT)
        y = top - 0.45 * inch

    def draw_heading(title: str):
        nonlocal y
        if y < (bottom_margin + 40):
            new_page()
        c.setFillColor(colors.black)
        font = f_cn(True) if any("\u4e00" <= ch <= "\u9fff" for ch in title) else f_en(True)
        c.setFont(font, 13)
        c.drawString(left, y, title[:180])
        y -= heading_gap

    def draw_body(text: str):
        nonlocal y
        available_w = PAGE_W - left - 0.90 * inch  # 右边留白对称

        for raw in text.splitlines():
            if not raw.strip():
                if y < bottom_margin:
                    new_page()
                y -= line_gap
                continue

            font0 = f_en(False) if is_ascii_line(raw) else f_cn(False)
            c.setFont(font0, body_font_size)

            wrapped = wrap_lines_by_pdf_width(
                raw,
                font_name=font0,
                font_size=body_font_size,
                max_width=available_w
            )

            for line in wrapped:
                if y < bottom_margin:
                    new_page()
                    c.setFont(font0, body_font_size)
                c.setFillColor(colors.black)
                c.drawString(left, y, line)
                y -= line_gap

            y -= para_gap

    draw_bg(c, BG_CONTENT)
    sections = parse_sections(report_text)
    if not sections:
        draw_body(report_text)
    else:
        for title, body in sections:
            draw_heading(title)
            if body.strip():
                draw_body(body)

    # Chart pages
    if inputs.charts:
        for chart_name, png_bytes in inputs.charts.items():
            if not isinstance(png_bytes, (bytes, bytearray)) or len(png_bytes) < 50:
                continue
            new_page()
            draw_bg(c, BG_CONTENT)

            c.setFont(f_en(True), 13)
            c.setFillColor(colors.black)
            c.drawString(left, top - 0.25 * inch, f"Chart: {chart_name}")

            try:
                img = ImageReader(io.BytesIO(png_bytes))
                img_w = PAGE_W - 2.0 * inch
                img_h = PAGE_H - 3.0 * inch
                c.drawImage(img, left, 1.4 * inch, width=img_w, height=img_h,
                            preserveAspectRatio=True, anchor='c')
            except Exception:
                c.setFont(f_en(False), 10)
                c.drawString(left, top - 0.55 * inch, f"[Chart render skipped: {chart_name}]")

    draw_footer(c, inputs.report_date, page_num)
    c.save()
    return out_path


# =========================================================
# UI
# =========================================================
require_login()

google_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
openai_key = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("配置")
    model = st.selectbox("OpenAI 模型", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    report_lang = st.selectbox("报告语言 / Report Language", ["中文", "English"], index=0)

    show_advanced = st.checkbox("显示高级设置", value=False)
    if show_advanced:
        radius_miles = st.slider("商圈半径（miles）", 1.0, 6.0, 4.0, 0.5)
        nearby_radius_m = st.slider("Google Nearby 搜索半径（米）", 300, 3000, 1200, 100)
    else:
        radius_miles = 4.0
        nearby_radius_m = 1200

    st.divider()
    logout_button()
    st.divider()
    st.caption("Built by c8geek")
    st.markdown("[LinkedIn](https://www.linkedin.com/)")

st.title(APP_TITLE)

if not google_key:
    st.warning("未检测到 GOOGLE_MAPS_API_KEY，请在 .streamlit/secrets.toml 配置。")
if not openai_key:
    st.warning("未检测到 OPENAI_API_KEY，请在 .streamlit/secrets.toml 配置。")

tab_food, tab_quick, tab_menu = st.tabs(["餐饮业态分析报告", "多业态快速诊断报告", "菜单智能调整"])


# =========================================================
# Tab 1: 餐饮业态分析报告（原：商圈分析报告）
# =========================================================
with tab_food:
    st.subheader("Step 1｜输入地址 → 搜索附近餐厅")
    address_input = st.text_input("输入地址（用于定位并搜索附近餐厅）", value="2406 19th Ave, San Francisco, CA 94116", key="addr_search_food")

    if st.button("搜索附近餐厅", type="primary", disabled=not google_key, key="btn_search_nearby_food"):
        geo = google_geocode(address_input, google_key)
        if not geo:
            st.error("无法解析地址，请输入更完整地址（含城市/州）。")
        else:
            lat, lng = geo
            places = google_nearby_places(lat, lng, google_key, radius_m=nearby_radius_m, place_type="restaurant")
            st.session_state["food_geo"] = (lat, lng)
            st.session_state["food_places"] = places
            st.success(f"已找到 {len(places)} 家附近餐厅。")

    places = st.session_state.get("food_places", [])
    place_details = st.session_state.get("food_place_details", {})

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

        selected_label = st.selectbox("选择目标餐厅（Google Nearby）", options, key="sel_place_food")
        selected_place_id = id_map.get(selected_label)

        if st.button("拉取餐厅详情（Google Place Details）", disabled=not google_key, key="btn_details_food"):
            if not selected_place_id:
                st.error("请先选择一个餐厅。")
            else:
                details = google_place_details(selected_place_id, google_key)
                if not details:
                    st.error("拉取详情失败。")
                else:
                    st.session_state["food_place_details"] = details
                    st.success("已拉取餐厅详情。")
                    place_details = details

    if place_details:
        st.subheader("Step 2｜上传菜单 + 自动商圈画像（ACS）")
        loc = (place_details.get("geometry", {}) or {}).get("location", {}) or {}
        rest_lat = float(loc.get("lat")) if loc.get("lat") is not None else None
        rest_lng = float(loc.get("lng")) if loc.get("lng") is not None else None

        col1, col2 = st.columns([1, 1])
        with col1:
            restaurant_en = st.text_input("餐厅英文名", value=place_details.get("name", ""), key="food_r_en")
            restaurant_cn = st.text_input("餐厅中文名（可选）", value="", key="food_r_cn")
            formatted_address = st.text_input("餐厅地址", value=place_details.get("formatted_address", address_input), key="food_r_addr")
            st.caption(f"Google：⭐{place_details.get('rating','')}（{place_details.get('user_ratings_total','')} reviews）")
            extra_context = st.text_area("补充业务背景（可选）", value="", height=120, key="food_r_ctx")

        with col2:
            st.markdown("### 门店外卖菜单上传")
            own_menu_files = st.file_uploader(
                "上传门店菜单（png/jpg/txt/csv/xlsx，多文件）",
                type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
                accept_multiple_files=True,
                key="food_own_menu_files"
            )

        with st.expander("自动获取商圈人口/收入/年龄/族裔/租住比例（US Census ACS）", expanded=True):
            if rest_lat and rest_lng:
                if st.button("获取 ACS 商圈画像（自动）", key="food_btn_acs"):
                    tract_info = census_tract_from_latlng(rest_lat, rest_lng)
                    if not tract_info:
                        st.warning("无法获取 tract 信息（Census geocoder）。")
                    else:
                        acs_data = acs_5y_profile(tract_info["STATE"], tract_info["COUNTY"], tract_info["TRACT"], year=2023)
                        st.session_state["food_tract_info"] = tract_info
                        st.session_state["food_acs_data"] = acs_data
                        st.success("已获取 ACS 数据（tract 级别代理）。")
            else:
                st.info("未能从 Google Place Details 获取坐标，无法调用 ACS。")

            tract_info = st.session_state.get("food_tract_info", None)
            acs_data = st.session_state.get("food_acs_data", None)
            if acs_data:
                st.write({
                    "ACS Year": acs_data.get("year"),
                    "Geography": acs_data.get("name"),
                    "Population (tract)": None if acs_data.get("pop_total") is None else int(acs_data.get("pop_total")),
                    "Median HH Income": None if acs_data.get("median_income") is None else f"${int(acs_data.get('median_income')):,}",
                    "Median Age": acs_data.get("median_age"),
                    "% Asian (proxy)": None if acs_data.get("pct_asian") is None else f"{acs_data.get('pct_asian')*100:.1f}%",
                    "% Renter (proxy)": None if acs_data.get("pct_renter") is None else f"{acs_data.get('pct_renter')*100:.1f}%",
                    "Note": "ACS 为 tract 级别代理，作为商圈画像方向性参考。"
                })

        with st.expander("上传订单报表（CSV，可选：用于时段/客单/热销/KPI）", expanded=False):
            order_files = st.file_uploader("上传平台订单导出 CSV（可多选）", type=["csv"], accept_multiple_files=True, key="food_orders_report")
            orders_meta = summarize_orders(order_files or [])
            if order_files:
                st.json(orders_meta)
        if "orders_meta" not in locals():
            orders_meta = {"files": [], "note": "No uploads"}

        st.subheader("Step 3｜生成报告内容（先预览可编辑，再生成 PDF）")
        if st.button("生成报告内容", type="primary", disabled=not openai_key, key="food_btn_gen_report"):
            progress = st.progress(0)
            status = st.empty()

            def step(pct: int, msg: str):
                progress.progress(pct)
                status.info(msg)

            report_date = dt.datetime.now().strftime("%m/%d/%Y")

            step(5, "解析门店菜单…")
            own_menu_meta = extract_menu_with_openai(own_menu_files or [], openai_key, model, label="OWN_MENU")

            step(35, "生成图表…")
            own_df = menu_to_df(own_menu_meta)
            charts = build_charts(own_df, [])

            step(60, "生成报告正文…")
            tract_info = st.session_state.get("food_tract_info", None)
            acs_data = st.session_state.get("food_acs_data", None)

            inputs = ReportInputs(
                report_date=report_date,
                restaurant_cn=(restaurant_cn.strip() or restaurant_en.strip()),
                restaurant_en=restaurant_en.strip(),
                address=formatted_address.strip(),
                radius_miles=radius_miles,
                own_menu_meta=own_menu_meta,
                orders_meta=orders_meta,
                competitors=[],
                extra_business_context=extra_context.strip(),
                acs=acs_data,
                tract_info=tract_info,
                restaurant_google=place_details,
                charts=charts,
            )

            prompt = build_prompt(inputs, report_lang)
            report_text = openai_text(prompt, openai_key, model=model, temperature=0.25)
            report_text = sanitize_text(report_text)

            step(80, "扩写补全（确保足够细）…")
            report_text = ensure_long_enough(report_text, openai_key, model=model, lang=report_lang, min_chars=12000)

            st.session_state["food_report_text"] = report_text
            st.session_state["food_report_inputs"] = inputs

            step(100, "完成：可预览编辑，再生成PDF。")
            status.success("报告已生成。")

        report_text = st.session_state.get("food_report_text", "")
        report_inputs: Optional[ReportInputs] = st.session_state.get("food_report_inputs", None)

        if report_text and report_inputs:
            st.subheader("报告预览（可编辑）")
            edited = st.text_area("报告正文", value=report_text, height=520, key="food_report_editor")
            st.session_state["food_report_text"] = sanitize_text(edited)

            st.subheader("图表预览（将自动附在 PDF 后面）")
            if report_inputs.charts:
                cols = st.columns(2)
                idx = 0
                for k, v in report_inputs.charts.items():
                    with cols[idx % 2]:
                        st.caption(k)
                        if isinstance(v, (bytes, bytearray)) and len(v) > 50:
                            st.image(v, use_container_width=True)
                        else:
                            st.warning(f"{k} 图表数据缺失，已跳过。")
                    idx += 1
            else:
                st.info("暂无图表（通常是菜单价格识别不足导致）。")

            st.subheader("Step 4｜生成 PDF（含图表页）")
            if st.button("生成 PDF", type="primary", key="food_btn_pdf"):
                with st.spinner("正在生成 PDF..."):
                    pdf_path = render_pdf(st.session_state["food_report_text"], report_inputs)
                st.success("PDF 生成完成。")
                with open(pdf_path, "rb") as f:
                    st.download_button("下载 PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
                st.caption(f"输出路径：{pdf_path}")
        else:
            st.info("先生成报告内容后，这里会出现预览与PDF下载。")


# =========================================================
# Tab 2: 多业态快速诊断报告
# =========================================================
with tab_quick:
    st.subheader("Step 1｜输入地址 → 搜索附近商家（可用于餐厅/头疗/美容/零售/健身等）")
    address_input = st.text_input("输入地址（用于定位并搜索附近商家）", value="San Bruno, CA", key="addr_search_quick")

    place_type = st.selectbox(
        "选择要搜索的商家类型（Google Places type）",
        ["establishment", "restaurant", "beauty_salon", "spa", "hair_care", "gym", "store", "cafe", "doctor", "dentist"],
        index=0,
        key="quick_place_type"
    )

    if st.button("搜索附近商家", type="primary", disabled=not google_key, key="btn_search_nearby_quick"):
        geo = google_geocode(address_input, google_key)
        if not geo:
            st.error("无法解析地址，请输入更完整地址（含城市/州）。")
        else:
            lat, lng = geo
            places = google_nearby_places(lat, lng, google_key, radius_m=nearby_radius_m, place_type=place_type)
            st.session_state["quick_geo"] = (lat, lng)
            st.session_state["quick_places"] = places
            st.success(f"已找到 {len(places)} 个附近商家。")

    places = st.session_state.get("quick_places", [])
    q_place_details = st.session_state.get("quick_place_details", {})

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

        selected_label = st.selectbox("选择目标商家（Google Nearby）", options, key="sel_place_quick")
        selected_place_id = id_map.get(selected_label)

        if st.button("拉取商家详情（Google Place Details）", disabled=not google_key, key="btn_details_quick"):
            if not selected_place_id:
                st.error("请先选择一个商家。")
            else:
                details = google_place_details(selected_place_id, google_key)
                if not details:
                    st.error("拉取详情失败。")
                else:
                    st.session_state["quick_place_details"] = details
                    st.success("已拉取商家详情。")
                    q_place_details = details

    if q_place_details:
        st.subheader("Step 2｜自动商圈画像（ACS）+ 补充背景")
        loc = (q_place_details.get("geometry", {}) or {}).get("location", {}) or {}
        q_lat = float(loc.get("lat")) if loc.get("lat") is not None else None
        q_lng = float(loc.get("lng")) if loc.get("lng") is not None else None

        col1, col2 = st.columns([1, 1])
        with col1:
            business_name = st.text_input("业务名称（可手动改）", value=q_place_details.get("name", ""), key="quick_name")
            business_address = st.text_input("业务地址", value=q_place_details.get("formatted_address", ""), key="quick_addr")
            st.caption(f"Google：⭐{q_place_details.get('rating','')}（{q_place_details.get('user_ratings_total','')} reviews）")
        with col2:
            extra_context = st.text_area("补充业务背景（可选）", value="", height=140, key="quick_ctx")

        with st.expander("自动获取区域画像（US Census ACS）", expanded=True):
            if q_lat and q_lng:
                if st.button("获取 ACS 区域画像（自动）", key="quick_btn_acs"):
                    tract_info = census_tract_from_latlng(q_lat, q_lng)
                    if not tract_info:
                        st.warning("无法获取 tract 信息（Census geocoder）。")
                    else:
                        acs_data = acs_5y_profile(tract_info["STATE"], tract_info["COUNTY"], tract_info["TRACT"], year=2023)
                        st.session_state["quick_tract_info"] = tract_info
                        st.session_state["quick_acs_data"] = acs_data
                        st.success("已获取 ACS 数据（tract 级别代理）。")
            else:
                st.info("未能从 Google Place Details 获取坐标，无法调用 ACS。")

            tract_info = st.session_state.get("quick_tract_info", None)
            acs_data = st.session_state.get("quick_acs_data", None)
            if acs_data:
                st.write({
                    "ACS Year": acs_data.get("year"),
                    "Geography": acs_data.get("name"),
                    "Population (tract)": None if acs_data.get("pop_total") is None else int(acs_data.get("pop_total")),
                    "Median HH Income": None if acs_data.get("median_income") is None else f"${int(acs_data.get('median_income')):,}",
                    "Median Age": acs_data.get("median_age"),
                    "% Asian (proxy)": None if acs_data.get("pct_asian") is None else f"{acs_data.get('pct_asian')*100:.1f}%",
                    "% Renter (proxy)": None if acs_data.get("pct_renter") is None else f"{acs_data.get('pct_renter')*100:.1f}%",
                    "Note": "ACS 为 tract 级别代理，作为区域画像方向性参考。"
                })

        st.subheader("Step 3｜生成快速诊断报告内容（先预览可编辑，再生成 PDF）")
        if st.button("生成报告内容", type="primary", disabled=not openai_key, key="quick_btn_gen"):
            progress = st.progress(0)
            status = st.empty()

            def step(pct: int, msg: str):
                progress.progress(pct)
                status.info(msg)

            report_date = dt.datetime.now().strftime("%m/%d/%Y")
            step(25, "整理输入数据…")
            tract_info = st.session_state.get("quick_tract_info", None)
            acs_data = st.session_state.get("quick_acs_data", None)

            # 强制覆盖 name/address（用户可手动改）
            q_place_details2 = dict(q_place_details)
            q_place_details2["name"] = business_name.strip()
            q_place_details2["formatted_address"] = business_address.strip()

            step(60, "AI 生成快速诊断正文…")
            prompt = build_quick_prompt(
                place_details=q_place_details2,
                acs_data=acs_data,
                tract_info=tract_info,
                extra_context=extra_context,
                report_date=report_date,
                lang=report_lang
            )
            text_out = openai_text(prompt, openai_key, model=model, temperature=0.25)
            text_out = sanitize_text(text_out)

            step(85, "扩写补全（确保足够细）…")
            text_out = ensure_long_enough(text_out, openai_key, model=model, lang=report_lang, min_chars=9000)

            # 复用 ReportInputs 作为 PDF 封面/页脚信息
            inputs = ReportInputs(
                report_date=report_date,
                restaurant_cn=business_name.strip(),
                restaurant_en=business_name.strip(),
                address=business_address.strip(),
                radius_miles=radius_miles,
                own_menu_meta={"label": "N/A", "files": [], "extracted": {"note": "N/A", "items": [], "promos": [], "notes": []}},
                orders_meta={"files": [], "note": "N/A"},
                competitors=[],
                extra_business_context=extra_context.strip(),
                acs=acs_data,
                tract_info=tract_info,
                restaurant_google=q_place_details2,
                charts={},
            )

            st.session_state["quick_report_text"] = text_out
            st.session_state["quick_report_inputs"] = inputs

            step(100, "完成：可预览编辑，再生成PDF。")
            status.success("快速诊断报告已生成。")

        q_text = st.session_state.get("quick_report_text", "")
        q_inputs: Optional[ReportInputs] = st.session_state.get("quick_report_inputs", None)

        if q_text and q_inputs:
            st.subheader("报告预览（可编辑）")
            edited = st.text_area("报告正文", value=q_text, height=520, key="quick_editor")
            st.session_state["quick_report_text"] = sanitize_text(edited)

            st.subheader("Step 4｜生成 PDF")
            if st.button("生成 PDF", type="primary", key="quick_btn_pdf"):
                with st.spinner("正在生成 PDF..."):
                    pdf_path = render_pdf(st.session_state["quick_report_text"], q_inputs)
                st.success("PDF 生成完成。")
                with open(pdf_path, "rb") as f:
                    st.download_button("下载 PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
                st.caption(f"输出路径：{pdf_path}")
        else:
            st.info("先生成报告内容后，这里会出现预览与PDF下载。")


# =========================================================
# Tab 3: Menu Optimizer (保持原功能)
# =========================================================
with tab_menu:
    st.subheader("菜单智能调整（价格锚点 + 心理定价 + 同行对标 + 套餐重构 + 抽佣测算）")
    st.caption("流程：识别菜单 → 引入竞对菜单 → 判断价带偏差 → 先调价 → 再组套餐 → 输出可直接上架的菜单结构（含堂食/外卖分层与抽佣净到手估算）。")

    colA, colB = st.columns([1, 1])
    with colA:
        menu_rest_name = st.text_input("门店名称（用于菜单输出）", value="My Restaurant", key="menu_rest_name")
        menu_rest_addr = st.text_input("门店地址（用于市场对标提示）", value="San Francisco, CA", key="menu_rest_addr")
        menu_lang = st.selectbox("输出菜单语言", ["中文", "English"], index=0, key="menu_lang")

        st.markdown("### 策略目标（新增）")
        objective_ui = st.radio(
            "选择一个主目标",
            ["优先引流品（低毛利高转化）", "优先利润最大化"],
            index=0,
            key="objective_ui"
        )
        objective_mode = "acquisition" if "引流" in objective_ui else "profit"

        st.markdown("### 价格分层（新增）")
        price_layering = st.checkbox(
            "启用堂食价/外卖价分层（推荐）",
            value=True,
            key="price_layering"
        )

        st.markdown("### 平台抽佣（新增）")
        commission_choice = st.selectbox("平台抽佣（用于外卖价测算）", ["25%", "30%", "自定义"], index=1, key="commission_choice")
        if commission_choice == "自定义":
            commission_rate = st.number_input("自定义抽佣比例（0~0.6）", min_value=0.0, max_value=0.6, value=0.30, step=0.01, key="commission_custom")
        else:
            commission_rate = 0.25 if commission_choice == "25%" else 0.30

        st.caption("提示：如果你想更贴近现实，可在外卖价里把 packaging / delivery fee / promo cost 作为 notes 提示，后续可扩展为成本模型。")

    with colB:
        own_menu_files2 = st.file_uploader(
            "上传门店菜单（png/jpg/txt/csv/xlsx，多文件）",
            type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="own_menu_files_optimizer"
        )
        st.markdown("**竞对菜单上传（必备）**")
        comp_menu_files2 = st.file_uploader(
            "上传竞对菜单（可多家多文件）",
            type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="comp_menu_files_optimizer"
        )
        st.caption("建议：至少 1-2 家竞对菜单；如果没有竞对数据，AI 会提示“数据不足”，并采取更保守的价带策略。")

    if st.button("生成智能菜单结构", type="primary", disabled=not openai_key, key="btn_opt_menu"):
        progress = st.progress(0)
        status = st.empty()

        def step(pct: int, msg: str):
            progress.progress(pct)
            status.info(msg)

        step(10, "解析门店菜单…")
        own_meta2 = extract_menu_with_openai(own_menu_files2 or [], openai_key, model, label="OWN_MENU_OPT")

        step(30, "解析竞对菜单…")
        comp_meta2 = extract_menu_with_openai(comp_menu_files2 or [], openai_key, model, label="COMP_MENU_OPT")

        own_df2 = menu_to_df(own_meta2)
        comp_df2 = menu_to_df(comp_meta2)

        step(55, "计算同行价带与偏高判断…")
        market_summary = summarize_market_prices(own_df2, [comp_df2])

        step(75, "AI 进行：调价 → 组套餐 → 分层定价 → 抽佣净到手估算…")
        try:
            optimized_df = generate_optimized_menu_df(
                api_key=openai_key,
                model=model,
                restaurant_name=menu_rest_name,
                address=menu_rest_addr,
                own_menu_meta=own_meta2,
                competitor_menu_metas=[comp_meta2],
                market_summary=market_summary,
                lang=menu_lang,
                objective_mode=objective_mode,
                commission_rate=float(commission_rate),
                price_layering=bool(price_layering),
            )
        except Exception as e:
            st.error(f"生成失败：{str(e)[:300]}")
            st.stop()

        st.session_state["optimized_menu_df"] = optimized_df
        st.session_state["optimized_market_summary"] = market_summary
        st.session_state["optimized_raw_metas"] = {"own": own_meta2, "comp": comp_meta2}
        st.session_state["optimized_settings"] = {
            "objective_mode": objective_mode,
            "commission_rate": float(commission_rate),
            "price_layering": bool(price_layering),
            "lang": menu_lang,
        }

        step(100, "完成。")
        status.success("已生成智能菜单结构（含：价格锚点/心理定价/同行对标/套餐/堂食外卖分层/抽佣净到手估算）。")

    optimized_df = st.session_state.get("optimized_menu_df", None)
    if isinstance(optimized_df, pd.DataFrame) and not optimized_df.empty:
        st.subheader("智能菜单预览（可直接下载）")
        st.dataframe(optimized_df, use_container_width=True, height=520)

        settings = st.session_state.get("optimized_settings", {})
        with st.expander("本次策略参数（用于复盘/对齐）", expanded=False):
            st.json(settings)

        market_summary = st.session_state.get("optimized_market_summary", {})
        with st.expander("同行价带摘要（用于解释为什么要调价/怎么锚点）", expanded=False):
            st.json(market_summary)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "下载菜单结构 CSV",
                data=df_to_csv_bytes(optimized_df),
                file_name="optimized_menu.csv",
                mime="text/csv"
            )
        with c2:
            st.download_button(
                "下载菜单结构 JSON",
                data=df_to_json_bytes(optimized_df),
                file_name="optimized_menu.json",
                mime="application/json"
            )
    else:
        st.info("上传门店菜单 + 竞对菜单，然后点击“生成智能菜单结构”。")
