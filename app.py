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
    menu_files_meta: Dict[str, Any]      # includes extracted items
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

    charts: Dict[str, bytes]  # chart_name -> png bytes


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
    # avoid markdown headings leaking
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

def parse_sections(text: str) -> List[Tuple[str, str]]:
    """
    支持两种标题：
    1) 【Title】
    2) ^\d+\. Title
    """
    text = text.strip()
    if not text:
        return []

    lines = text.splitlines()
    norm = []
    for ln in lines:
        norm.append(ln.rstrip())
    text = "\n".join(norm)

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
        "name","formatted_address","rating","user_ratings_total","types",
        "url","website","formatted_phone_number","opening_hours","reviews","geometry"
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
# Yelp (optional)
# =========================================================
def yelp_search_business(term: str, location: str, api_key: str, limit: int = 3) -> Dict[str, Any]:
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"term": term, "location": location, "limit": limit}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code != 200:
        return {"error": f"Yelp search failed: {r.status_code}", "raw": r.text[:300]}
    return r.json()

def yelp_get_reviews(business_id: str, api_key: str) -> Dict[str, Any]:
    url = f"https://api.yelp.com/v3/businesses/{business_id}/reviews"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code != 200:
        return {"error": f"Yelp reviews failed: {r.status_code}", "raw": r.text[:300]}
    return r.json()


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
        return {"label": label, "files": [], "extracted": {"note": "no files", "items": [], "promos": []}}

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

    extracted_items = extracted_items[:1200]
    promos = promos[:150]

    return {
        "label": label,
        "files": [{"name": f.name, "type": getattr(f, "type", "")} for f in files],
        "extracted": {
            "items": extracted_items,
            "promos": promos,
            "notes": notes[:100],
        }
    }


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
# Menu Stats + Charts (visual)
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
        cat = (it.get("category") or "").strip()
        price = _to_price(it.get("price"))
        notes = (it.get("notes") or "").strip()
        if not name and price is None:
            continue
        rows.append({"name": name, "category": cat if cat else "Uncategorized", "price": price, "notes": notes})
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

    # 1) Price distribution
    fig = plt.figure()
    own_df["price"].dropna().plot(kind="hist", bins=18)
    plt.title("Own Menu Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    charts["chart_own_price_hist"] = make_png(fig)

    # 2) Category count
    fig = plt.figure()
    own_df["category"].value_counts().head(12).plot(kind="bar")
    plt.title("Own Menu Category Mix (Top 12)")
    plt.xlabel("Category")
    plt.ylabel("Items")
    charts["chart_own_category_bar"] = make_png(fig)

    # 3) Price tiers matrix (count by tier)
    bins = [0, 8, 12, 16, 20, 25, 35, 999]
    labels = ["<$8", "$8-12", "$12-16", "$16-20", "$20-25", "$25-35", "$35+"]
    tier = pd.cut(own_df["price"], bins=bins, labels=labels, include_lowest=True)
    fig = plt.figure()
    tier.value_counts().reindex(labels).fillna(0).plot(kind="bar")
    plt.title("Own Menu Price Tiers")
    plt.xlabel("Tier")
    plt.ylabel("Items")
    charts["chart_own_price_tiers"] = make_png(fig)

    # 4) Competitor median price compare (if any comp has prices)
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
# Menu Smart Adjust (NEW)
# =========================================================
DEFAULT_TARGET_MEDIAN_MAIN = 18.95
HIGH_PRICE_THRESHOLD_RATIO = 1.12
MIN_COMBOS_REQUIRED = 4
FRONT_LIMIT = 35

def _has_chinese(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in (s or ""))

def _split_cn_en(name: str) -> Tuple[str, str]:
    """
    尝试从 name 中拆出中英文。无法拆就：中文/英文都填 name（不强行猜）。
    """
    name = (name or "").strip()
    if not name:
        return "", ""
    if _has_chinese(name) and re.search(r"[A-Za-z]", name):
        # 粗暴拆：把中文和英文各自提取
        cn = "".join([ch for ch in name if "\u4e00" <= ch <= "\u9fff" or ch in "（）()·•- "]).strip()
        en = re.sub(r"[\u4e00-\u9fff（）()·•]", " ", name)
        en = re.sub(r"\s+", " ", en).strip()
        return cn, en
    # 仅中文或仅英文
    if _has_chinese(name):
        return name, ""
    return "", name

def normalize_extracted_items(menu_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = (((menu_meta or {}).get("extracted") or {}).get("items") or [])
    out = []
    seen = set()
    for it in rows:
        name = (it.get("name") or "").strip()
        if not name:
            continue
        cn, en = _split_cn_en(name)
        price = _to_price(it.get("price"))
        cat = (it.get("category") or "").strip()
        notes = (it.get("notes") or "").strip()

        key = (cn.lower(), en.lower(), price, cat.lower())
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "name_cn": cn,
            "name_en": en,
            "price": price,
            "raw_category": cat if cat else "Uncategorized",
            "notes": notes,
        })
    return out

def item_fullname(it: Dict[str, Any]) -> str:
    cn = (it.get("name_cn") or "").strip()
    en = (it.get("name_en") or "").strip()
    if cn and en:
        return f"{en} {cn}".strip()
    return cn or en

def classify_item(it: Dict[str, Any]) -> str:
    name = (item_fullname(it) or "").lower()
    cat = (it.get("raw_category") or "").lower()
    notes = (it.get("notes") or "").lower()
    blob = " ".join([name, cat, notes])

    # Combo
    if any(k in blob for k in ["套餐", "超值", "combo", "value", "set", "bundle"]):
        return "combo"
    # Add-on
    if any(k in blob for k in ["加", "加配", "升级", "upgrade", "add-on", "addon", "extra", "+$"]):
        return "addon"
    # Drink
    if any(k in blob for k in [
        "奶茶","milk tea","柠檬","lemon","coffee","咖啡","soda","汽水","coconut","椰子",
        "jasmine","茉莉","chrysanthemum","菊花","prunella","夏枯草","tea","茶"
    ]):
        return "drink"
    # Main (rough)
    if any(k in blob for k in ["饭","rice","河粉","chow fun","炒面","noodle","意粉","spaghetti","粉","面","焗","baked"]):
        return "main"
    return "other"

def split_by_type(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out = {"main": [], "drink": [], "combo": [], "addon": [], "other": []}
    for it in items:
        t = classify_item(it)
        it["type"] = t
        out[t].append(it)
    return out

def median_price(items: List[Dict[str, Any]]) -> Optional[float]:
    prices = [it["price"] for it in items if isinstance(it.get("price"), (int, float)) and it["price"] and it["price"] > 0]
    if not prices:
        return None
    s = sorted(prices)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def detect_combo_system(combos: List[Dict[str, Any]], min_required: int) -> bool:
    return len(combos) >= min_required

def is_menu_overpriced(mains: List[Dict[str, Any]], target_median: float, ratio: float) -> Tuple[bool, Optional[float]]:
    cur = median_price(mains)
    if cur is None:
        return False, None
    return (cur > target_median * ratio), cur

def scale_prices(items: List[Dict[str, Any]], scale: float) -> List[Dict[str, Any]]:
    adjusted = []
    for it in items:
        p = it.get("price")
        it2 = dict(it)
        it2["price_old"] = p
        if isinstance(p, (int, float)) and p > 0:
            it2["price"] = round(float(p) * scale, 2)
        adjusted.append(it2)
    return adjusted

def safe_json_obj(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}

def ai_market_benchmark(
    items: List[Dict[str, Any]],
    location: str,
    cuisine: str,
    target_median_main: float,
    api_key: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    用 AI 估算同地区同品类“主食合理价带”，并给出逐菜 new_price。
    失败则返回原 items + meta(ai_error)。
    """
    mains = [i for i in items if classify_item(i) == "main" and isinstance(i.get("price"), (int, float)) and i["price"] and i["price"] > 0]
    cur_med = median_price(mains) or 0

    prompt = f"""
你是北美外卖定价与菜单工程顾问。
地点：{location}
菜系：{cuisine}

我会给你一份菜品列表（含价格）。请完成：
1) 给出同地区同品类外卖平台“主食”常见合理中位数和合理区间（范围要可执行）。
2) 判断整体是否偏高。
3) 如果偏高，给出每一道菜建议的新价格（只要给出你有把握的，没把握可以不写在 adjusted_items 里）。

只输出严格 JSON，不要输出任何额外文字：
{{
  "market_median_main": 0,
  "market_range_main": [0,0],
  "is_overpriced": true/false,
  "strategy": "一句话策略",
  "adjusted_items": [
    {{"name":"", "new_price":0}}
  ]
}}

当前目标主食中位数倾向：{target_median_main}
当前主食中位数（估算）：{cur_med}

菜品（name - price）：
{json.dumps([{"name": item_fullname(i), "price": i.get("price")} for i in items], ensure_ascii=False)}
""".strip()

    try:
        text_out = openai_text(prompt, api_key, model=model, temperature=0.2)
        obj = safe_json_obj(text_out)
        mapping = {x.get("name", ""): x.get("new_price") for x in (obj.get("adjusted_items") or [])}

        adjusted = []
        for it in items:
            nm = item_fullname(it)
            newp = mapping.get(nm, None)
            it2 = dict(it)
            it2["price_old"] = it.get("price")
            if isinstance(newp, (int, float)) and newp > 0:
                it2["price"] = round(float(newp), 2)
            adjusted.append(it2)
        return adjusted, obj
    except Exception as e:
        return items, {"ai_error": str(e)[:200]}

def build_combos_if_missing(
    items: List[Dict[str, Any]],
    location: str,
    cuisine: str,
    api_key: str,
    model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    typed = split_by_type(items)
    mains = [m for m in typed["main"] if isinstance(m.get("price"), (int, float)) and m["price"] and m["price"] > 0]
    drinks = [d for d in typed["drink"] if isinstance(d.get("price"), (int, float)) and d["price"] and d["price"] > 0]

    # pick candidates
    mains_sorted = sorted(mains, key=lambda x: abs((x.get("price") or 0) - DEFAULT_TARGET_MEDIAN_MAIN))
    mains_pick = mains_sorted[:6]
    drinks_pick = drinks[:4]

    new_combos = []
    for i, main in enumerate(mains_pick):
        if not drinks_pick:
            break
        drink = drinks_pick[i % len(drinks_pick)]
        base = (main["price"] or 0) + (drink["price"] or 0)
        combo_price = round(base - 1.50, 2)

        combo_name_cn = f"超值套餐｜{(main.get('name_cn') or '主食')} + {(drink.get('name_cn') or '饮品')}"
        combo_name_en = f"Value Combo | {(main.get('name_en') or 'Main')} + {(drink.get('name_en') or 'Drink')}"
        new_combos.append({
            "name_cn": combo_name_cn,
            "name_en": combo_name_en,
            "price": combo_price,
            "raw_category": "Value Combos",
            "notes": "Auto-generated combo",
            "type": "combo",
        })

    meta = {"generated_combos": len(new_combos), "ai_combo_optimized": False}

    # AI optimize combo names/prices (best effort)
    if api_key:
        try:
            prompt = f"""
你是外卖套餐工程师。地点：{location}，菜系：{cuisine}
我生成了一批套餐（主食+饮品）。请优化：
1) 套餐命名：更像平台爆品，简短、明确、利于点击
2) 套餐价格：保持“比单点更划算”的感知；不要低到不赚钱
只输出严格JSON：
{{
  "combos":[{{"idx":0,"name_cn":"","name_en":"","price":0}}]
}}
当前套餐：
{json.dumps(new_combos, ensure_ascii=False)}
""".strip()
            out = openai_text(prompt, api_key, model=model, temperature=0.2)
            obj = safe_json_obj(out)
            combos_out = obj.get("combos") or []
            for row in combos_out:
                idx = row.get("idx")
                if isinstance(idx, int) and 0 <= idx < len(new_combos):
                    if row.get("name_cn"): new_combos[idx]["name_cn"] = row["name_cn"]
                    if row.get("name_en"): new_combos[idx]["name_en"] = row["name_en"]
                    if isinstance(row.get("price"), (int, float)) and row["price"] > 0:
                        new_combos[idx]["price"] = round(float(row["price"]), 2)
            meta["ai_combo_optimized"] = True
        except Exception as e:
            meta["ai_error"] = str(e)[:200]

    merged = items + new_combos
    # re-dedup
    seen = set()
    dedup = []
    for it in merged:
        key = (item_fullname(it).lower(), it.get("price"), (it.get("raw_category") or "").lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return dedup, meta

def build_smart_menu(items: List[Dict[str, Any]], front_limit: int = FRONT_LIMIT) -> Dict[str, Any]:
    typed = split_by_type(items)
    mains = [m for m in typed["main"] if isinstance(m.get("price"), (int, float)) and m["price"] and m["price"] > 0]
    drinks = [d for d in typed["drink"] if isinstance(d.get("price"), (int, float)) and d["price"] and d["price"] > 0]
    combos = [c for c in typed["combo"] if isinstance(c.get("price"), (int, float)) and c["price"] and c["price"] > 0]
    addons = [a for a in typed["addon"] if isinstance(a.get("price"), (int, float)) and a["price"] and a["price"] > 0]
    others = typed["other"]

    anchors = sorted(mains, key=lambda x: x["price"], reverse=True)[:6]

    core_band = [m for m in mains if 16.0 <= float(m["price"]) <= 20.5 and m not in anchors]
    best_sellers = sorted(core_band, key=lambda x: abs(float(x["price"]) - DEFAULT_TARGET_MEDIAN_MAIN))[:10]

    front_combos = combos[:8]
    front_addons = addons[:6]
    front_drinks = drinks[:6]

    used = set()
    def mark_used(lst):
        for x in lst:
            used.add((item_fullname(x), x.get("price")))

    mark_used(anchors)
    mark_used(best_sellers)
    mark_used(front_combos)
    mark_used(front_addons)
    mark_used(front_drinks)

    classic = []
    for it in items:
        if (item_fullname(it), it.get("price")) not in used:
            classic.append(it)

    return {
        "Chef_Signature": anchors,
        "Best_Sellers": best_sellers,
        "Value_Combos": front_combos,
        "Make_It_Better": front_addons,
        "Drinks": front_drinks,
        "Classic_Menu": classic,
    }

def menu_to_flat_df(menu: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for section, lst in menu.items():
        for it in lst:
            rows.append({
                "section": section,
                "name_cn": it.get("name_cn",""),
                "name_en": it.get("name_en",""),
                "name": item_fullname(it),
                "price": it.get("price"),
                "price_old": it.get("price_old", None),
                "raw_category": it.get("raw_category",""),
                "type": it.get("type", classify_item(it)),
                "notes": it.get("notes",""),
            })
    return pd.DataFrame(rows)


# =========================================================
# Prompt (forces deep + specific execution)
# =========================================================
def build_prompt(inputs: ReportInputs) -> str:
    own_df = menu_to_df(inputs.own_menu_meta)
    own_stats = {}
    if not own_df.empty:
        own_stats = {
            "items_count": int(own_df.shape[0]),
            "price_count": int(own_df["price"].dropna().shape[0]),
            "median_price": None if own_df["price"].dropna().empty else float(own_df["price"].median()),
            "min_price": None if own_df["price"].dropna().empty else float(own_df["price"].min()),
            "max_price": None if own_df["price"].dropna().empty else float(own_df["price"].max()),
            "top_categories": own_df["category"].value_counts().head(10).to_dict(),
        }

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
        "own_menu_stats": own_stats,
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
        "season_hint": "请按北美常见季节（Winter/Spring/Summer/Fall）与节日节点（圣诞/新年/情人节/复活节/母亲节/暑期/返校/万圣节/感恩节）给具体季节菜建议。",
    }

    return f"""
你是 AuraInsight 的北美餐饮增长咨询顾问（偏外卖 + 商圈增长 + 菜单工程 + 促销经济学）。
你将收到一个 JSON 数据包（含：商圈ACS、门店与竞对的Google/Yelp信息、门店与竞对的外卖菜单识别结果、订单报表字段摘要、以及已经生成的可视化图表列表）。

目标：输出一份“能直接执行”的咨询级《Trade Area & Growth Diagnostic》报告正文，用于生成 PDF。

硬性要求（违反任何一条都算失败）：
1) 严禁输出 Markdown（不要出现：#、##、**、```、|---|、[]()）。
2) 章节标题必须使用 “1. ”这种编号标题（例如：5. Pricing, Anchors & Promo Economics）。
3) 每一章开头必须给 3–6 条 Key Takeaways（尽量带数字、范围或明确阈值）。
4) 所有建议必须包含五件套：【动作】【原因】【预期影响】【KPI】【两周验证方法】。
5) “第5章”必须写得极细：对价格锚点、折扣、买一送一、满减、加价购、组合套餐、免运费对标策略逐条拆解。
   - 买一送一必须指定到具体菜品（至少 3 个候选），并写清楚：为什么是它、风险是什么、如何限制规则、如何防薅羊毛、如何验证。
6) 必须对“全菜单所有菜品”做结构化分析（如果识别不全也要做：标出缺口如何补齐）。
   - 至少输出：价格带矩阵（<$10、$10-15、$15-20、$20-25、$25+），每个价格带应该放什么品类与锚点商品。
   - 至少输出：菜单工程（星/牛/谜/狗）执行方式与落地清单。
7) 运营动作清单必须展开为“具体方案”：季节性上新必须按“当下季节 + 下个季节”分别给建议（每季至少 3 个菜），写：上什么、为什么、怎么拍图、怎么命名、怎么定价、怎么上平台、怎么做首周促销。
8) 报告必须足够长，生成后 PDF 目标至少 6–7 页。内容要“可执行、带表格/清单”，但不要用 Markdown 表格；如果需要表格，请用：
   表格:
   colA,colB,colC
   ...
9) 必须包含“可视化图表解读”：
   - 你不用画图，但必须引用图表名称并解读（例如 chart_own_price_hist、chart_own_category_bar、chart_own_price_tiers、chart_comp_median_price）。
10) 必须包含“Data Gaps & How to Collect”，把缺哪些数据、怎么补、谁负责、需要多久写清楚。

输出章节顺序必须如下（标题一字不差）：
1. Executive Summary
2. Trade Area & Demographics
3. Customer Segments & JTBD
4. Demand, Occasion & Menu Positioning
5. Competitive Landscape (Google + Yelp + Menu)
6. Pricing, Anchors & Promo Economics
7. Menu Architecture & Menu Engineering
8. Platform Growth Playbook (30/60/90)
9. Measurement System & Experiment Design
10. Appendix A: Own Menu Deep Dive
11. Appendix B: Competitor Menu Deep Dive
12. Data Gaps & How to Collect

输入 JSON：
{json.dumps(blob, ensure_ascii=False, indent=2)}

开始输出报告正文：
""".strip()

def ensure_long_enough(report_text: str, api_key: str, model: str, min_chars: int = 16000) -> str:
    t = sanitize_text(report_text)
    if len(t) >= min_chars:
        return t

    expand_prompt = f"""
你将收到一份报告正文。请在不改变章节标题顺序的前提下，显著扩写，使其更细、更能执行。
重点扩写：
- 第6章（Pricing, Anchors & Promo Economics）：买一送一/满减/套餐/加价购/折扣门槛要给具体菜品与规则。
- 第7章（Menu Engineering）：把所有品类按星/牛/谜/狗的执行方法写成“动作清单”，含KPI与两周验证。
- Appendix A/B：列出更多菜单级别的建议（至少 25 条），每条写“当前价(若缺写待补齐)/建议价/理由/竞对对标/风险与对策/验证”。
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
# PDF Render (fix truncation + add chart pages)
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
    safe_name = safe_name.replace(" ", "_") or "Restaurant"
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
    draw_bg(c, BG_CONTENT)

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
        c.drawString(left, y, title[:140])
        y -= heading_gap

    def draw_body(text: str):
        nonlocal y
        max_chars = 100
        for line in wrap_lines_by_chars(text, max_chars):
            if y < bottom_margin:
                new_page()
            font = f_en(False) if is_ascii_line(line) else f_cn(False)
            c.setFillColor(colors.black)
            c.setFont(font, body_font_size)
            c.drawString(left, y, line)
            y -= line_gap
        y -= para_gap

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
            new_page()
            draw_bg(c, BG_CONTENT)

            c.setFont(f_en(True), 13)
            c.setFillColor(colors.black)
            c.drawString(left, top - 0.25 * inch, f"Chart: {chart_name}")

            img = ImageReader(io.BytesIO(png_bytes))
            img_w = PAGE_W - 2.0 * inch
            img_h = PAGE_H - 3.0 * inch
            c.drawImage(img, left, 1.4 * inch, width=img_w, height=img_h, preserveAspectRatio=True, anchor='c')

    draw_footer(c, inputs.report_date, page_num)
    c.save()
    return out_path


# =========================================================
# UI
# =========================================================
require_login()

google_key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
openai_key = st.secrets.get("OPENAI_API_KEY", "")
yelp_key = st.secrets.get("YELP_API_KEY", "")  # optional

with st.sidebar:
    st.header("配置")
    model = st.selectbox("OpenAI 模型", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)

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
if not yelp_key:
    st.info("未检测到 YELP_API_KEY（可选）。竞对 Yelp 维度会缺失。")

# =========================================================
# Tabs: Report vs Menu Smart Adjust
# =========================================================
tab_report, tab_menu = st.tabs(["📊 门店分析报告（PDF）", "🧠 菜单智能调整（先调价→再组套餐→出完整菜单）"])


# =========================================================
# TAB 1: REPORT (your original flow, unchanged)
# =========================================================
with tab_report:

    # =========================================================
    # Step 1: Search restaurant
    # =========================================================
    st.subheader("Step 1｜输入地址 → 搜索附近餐厅")
    address_input = st.text_input("输入地址（用于定位并搜索附近餐厅）", value="2406 19th Ave, San Francisco, CA 94116", key="report_address_input")

    if st.button("搜索附近餐厅", type="primary", disabled=not google_key, key="btn_search_nearby"):
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
    place_details = st.session_state.get("place_details", {})

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

        selected_label = st.selectbox("选择目标餐厅（Google Nearby）", options, key="report_selected_label")
        selected_place_id = id_map.get(selected_label)

        if st.button("拉取餐厅详情（Google Place Details）", disabled=not google_key, key="btn_place_details"):
            if not selected_place_id:
                st.error("请先选择一个餐厅。")
            else:
                details = google_place_details(selected_place_id, google_key)
                if not details:
                    st.error("拉取详情失败。")
                else:
                    st.session_state["place_details"] = details
                    st.success("已拉取餐厅详情。")
                    place_details = details


    # =========================================================
    # Step 2
    # =========================================================
    if place_details:
        st.subheader("Step 2｜上传菜单 + 自动商圈画像（ACS） + 竞对（Google/Yelp + 竞对菜单上传）")

        loc = (place_details.get("geometry", {}) or {}).get("location", {}) or {}
        rest_lat = float(loc.get("lat")) if loc.get("lat") is not None else None
        rest_lng = float(loc.get("lng")) if loc.get("lng") is not None else None

        col1, col2 = st.columns([1, 1])
        with col1:
            restaurant_en = st.text_input("餐厅英文名", value=place_details.get("name", ""), key="report_restaurant_en")
            restaurant_cn = st.text_input("餐厅中文名（可选）", value="", key="report_restaurant_cn")
            formatted_address = st.text_input("餐厅地址", value=place_details.get("formatted_address", address_input), key="report_formatted_address")
            st.caption(f"Google：⭐{place_details.get('rating','')}（{place_details.get('user_ratings_total','')} reviews）")
            extra_context = st.text_area(
                "补充业务背景（可选）",
                value="例如：经营年限、主打菜、目标客群、当前痛点（单量/评分/利润/人手等）。",
                height=120,
                key="report_extra_context"
            )
        with col2:
            st.markdown("### 门店外卖菜单上传（替代平台链接）")
            own_menu_files = st.file_uploader(
                "上传门店菜单（png/jpg/txt/csv/xlsx，支持多文件）",
                type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
                accept_multiple_files=True,
                key="own_menu_files"
            )
            st.caption("系统会识别：菜品、价格、分类、套餐结构、促销文案；最终写入 PDF（Appendix A + 图表页）。")

        # ACS
        with st.expander("自动获取商圈人口/收入/年龄/族裔/租住比例（US Census ACS）", expanded=True):
            if rest_lat and rest_lng:
                if st.button("获取 ACS 商圈画像（自动）", key="btn_get_acs"):
                    tract_info = census_tract_from_latlng(rest_lat, rest_lng)
                    if not tract_info:
                        st.warning("无法获取 tract 信息（Census geocoder）。")
                    else:
                        acs_data = acs_5y_profile(tract_info["STATE"], tract_info["COUNTY"], tract_info["TRACT"], year=2023)
                        st.session_state["tract_info"] = tract_info
                        st.session_state["acs_data"] = acs_data
                        st.success("已获取 ACS 数据（tract 级别代理）。")
            else:
                st.info("未能从 Google Place Details 获取坐标，无法调用 ACS。")

            tract_info = st.session_state.get("tract_info", None)
            acs_data = st.session_state.get("acs_data", None)
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

        # Competitors
        st.markdown("### 竞对信息（可增删行：用于差异化与竞品分析）")
        if "comp_rows" not in st.session_state:
            st.session_state.comp_rows = 3

        cA, cB, cC = st.columns([1, 1, 2])
        with cA:
            if st.button("➕ 添加竞对", key="btn_add_comp"):
                st.session_state.comp_rows += 1
        with cB:
            if st.button("➖ 删除最后一个", disabled=st.session_state.comp_rows <= 1, key="btn_del_comp"):
                st.session_state.comp_rows = max(1, st.session_state.comp_rows - 1)
        with cC:
            st.caption("每个竞对：填写名称/地址 → 拉取 Google/Yelp → 上传竞对菜单文件。")

        comp_inputs: List[CompetitorInput] = []
        comp_summary_rows = []

        for i in range(st.session_state.comp_rows):
            with st.container(border=True):
                st.markdown(f"竞对 #{i+1}")
                cc1, cc2, cc3 = st.columns([2, 2, 2])
                with cc1:
                    comp_name = st.text_input(f"竞对名称或地址（#{i+1}）", value="", key=f"comp_name_{i}")
                with cc2:
                    comp_notes = st.text_input(f"备注（可选 #{i+1}）", value="", key=f"comp_notes_{i}")
                with cc3:
                    comp_menu_files = st.file_uploader(
                        f"上传竞对菜单（#{i+1}）png/jpg/txt/csv/xlsx",
                        type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
                        accept_multiple_files=True,
                        key=f"comp_menu_files_{i}"
                    )

                pull_col1, pull_col2 = st.columns([1, 2])
                with pull_col1:
                    pull = st.button(f"拉取竞对 Google + Yelp（#{i+1}）", key=f"pull_comp_{i}", disabled=not google_key)
                with pull_col2:
                    st.caption("Google: 评分/评论/位置/营业时间；Yelp: 价格带/分类/评论示例（如配置了YELP_API_KEY）。")

                comp_google = st.session_state.get(f"comp_google_{i}", {})
                comp_yelp = st.session_state.get(f"comp_yelp_{i}", {})

                if pull and comp_name.strip():
                    pid = google_textsearch_place_id(f"{comp_name} {formatted_address}", google_key)
                    comp_google = google_place_details(pid, google_key) if pid else {"error": "Google textsearch failed"}
                    st.session_state[f"comp_google_{i}"] = comp_google

                    if yelp_key:
                        ysr = yelp_search_business(comp_name, formatted_address, yelp_key, limit=3)
                        best = None
                        if isinstance(ysr, dict) and ysr.get("businesses"):
                            best = ysr["businesses"][0]
                            rid = best.get("id")
                            rev = yelp_get_reviews(rid, yelp_key) if rid else {}
                            comp_yelp = {"best_match": best, "reviews": rev}
                        else:
                            comp_yelp = {"error": "No Yelp match", "raw": ysr}
                    else:
                        comp_yelp = {"note": "YELP_API_KEY not configured"}
                    st.session_state[f"comp_yelp_{i}"] = comp_yelp

                comp_summary_rows.append({
                    "competitor": comp_name.strip(),
                    "google_rating": comp_google.get("rating", "") if isinstance(comp_google, dict) else "",
                    "google_reviews": comp_google.get("user_ratings_total", "") if isinstance(comp_google, dict) else "",
                    "yelp_rating": (comp_yelp.get("best_match", {}) or {}).get("rating", "") if isinstance(comp_yelp, dict) else "",
                    "yelp_reviews": (comp_yelp.get("best_match", {}) or {}).get("review_count", "") if isinstance(comp_yelp, dict) else "",
                    "menus_uploaded": 0 if not comp_menu_files else len(comp_menu_files),
                })

                comp_inputs.append(
                    CompetitorInput(
                        name_or_address=comp_name.strip(),
                        notes=comp_notes.strip(),
                        menu_files_meta={"label": f"COMP_{i+1}", "files": [{"name": f.name} for f in (comp_menu_files or [])],
                                         "extracted": {"items": [], "promos": [], "notes": []}},
                        google=comp_google if isinstance(comp_google, dict) else {},
                        yelp=comp_yelp if isinstance(comp_yelp, dict) else {},
                    )
                )

        if comp_summary_rows:
            st.dataframe(pd.DataFrame(comp_summary_rows), use_container_width=True)

        # Orders
        with st.expander("上传订单报表（CSV，可选：用于时段/客单/热销/KPI）", expanded=False):
            order_files = st.file_uploader("上传平台订单导出 CSV（可多选）", type=["csv"], accept_multiple_files=True, key="report_order_files")
            orders_meta = summarize_orders(order_files or [])
            if order_files:
                st.json(orders_meta)

        if "orders_meta" not in locals():
            orders_meta = {"files": [], "note": "No uploads"}

        # Step 3
        st.subheader("Step 3｜生成深度分析报告（含可视化图表 + 可执行动作展开）")

        if st.button("生成报告内容", type="primary", disabled=not openai_key, key="btn_generate_report"):
            progress = st.progress(0)
            status = st.empty()

            def step(pct: int, msg: str):
                progress.progress(pct)
                status.info(msg)

            report_date = dt.datetime.now().strftime("%m/%d/%Y")

            step(5, "正在解析门店菜单（识别菜品/价格/促销）...")
            own_menu_meta = extract_menu_with_openai(own_menu_files or [], openai_key, model, label="OWN_MENU")

            step(25, "正在解析竞对菜单（逐个识别菜品/价格/促销）...")
            competitors_full: List[CompetitorInput] = []
            for i in range(st.session_state.comp_rows):
                comp_name = st.session_state.get(f"comp_name_{i}", "").strip()
                comp_notes = st.session_state.get(f"comp_notes_{i}", "").strip()
                comp_google = st.session_state.get(f"comp_google_{i}", {}) or {}
                comp_yelp = st.session_state.get(f"comp_yelp_{i}", {}) or {}
                comp_files = st.session_state.get(f"comp_menu_files_{i}", None)

                if comp_files and isinstance(comp_files, list) and len(comp_files) > 0:
                    comp_menu_meta = extract_menu_with_openai(comp_files, openai_key, model, label=f"COMP_{i+1}")
                else:
                    comp_menu_meta = {"label": f"COMP_{i+1}", "files": [], "extracted": {"items": [], "promos": [], "notes": ["no competitor menu uploaded"]}}

                competitors_full.append(CompetitorInput(
                    name_or_address=comp_name,
                    notes=comp_notes,
                    menu_files_meta=comp_menu_meta,
                    google=comp_google,
                    yelp=comp_yelp
                ))

            step(45, "正在生成菜单可视化图表（价格分布/品类结构/价格带/竞对对比）...")
            own_df = menu_to_df(own_menu_meta)
            comp_dfs = []
            for c in competitors_full:
                dfc = menu_to_df(c.menu_files_meta)
                comp_dfs.append((c.name_or_address or c.menu_files_meta.get("label", "Competitor"), dfc))
            charts = build_charts(own_df, comp_dfs)

            step(60, "正在生成咨询级报告（含：买一送一具体菜品、季节性上新、动作清单展开）...")
            tract_info = st.session_state.get("tract_info", None)
            acs_data = st.session_state.get("acs_data", None)

            inputs = ReportInputs(
                report_date=report_date,
                restaurant_cn=(restaurant_cn.strip() or restaurant_en.strip()),
                restaurant_en=restaurant_en.strip(),
                address=formatted_address.strip(),
                radius_miles=radius_miles,
                own_menu_meta=own_menu_meta,
                orders_meta=orders_meta,
                competitors=competitors_full,
                extra_business_context=extra_context.strip(),
                acs=acs_data,
                tract_info=tract_info,
                restaurant_google=place_details,
                charts=charts,
            )

            prompt = build_prompt(inputs)

            report_text = openai_text(prompt, openai_key, model=model, temperature=0.25)
            report_text = sanitize_text(report_text)

            step(80, "正在自动扩写（确保足够长、足够细、能落地执行）...")
            report_text = ensure_long_enough(report_text, openai_key, model=model, min_chars=16000)

            step(95, "正在完成渲染准备（可编辑预览 + PDF）...")
            st.session_state["report_text"] = report_text
            st.session_state["report_inputs"] = inputs

            step(100, "完成：报告内容已生成。你可以预览编辑并输出 PDF。")
            status.success("报告内容已生成（包含图表页与更完整的第6章定价/促销深度分析）。")

    # Preview + PDF
    report_text = st.session_state.get("report_text", "")
    report_inputs: Optional[ReportInputs] = st.session_state.get("report_inputs", None)

    if report_text and report_inputs:
        st.subheader("预览（可编辑）")
        edited = st.text_area("报告正文（你可以直接修改）", value=report_text, height=520, key="report_editor")
        st.session_state["report_text"] = sanitize_text(edited)

        st.subheader("图表预览（将自动附在 PDF 后面）")
        if report_inputs.charts:
            for k, v in report_inputs.charts.items():
                st.markdown(f"**{k}**")
                st.image(v, use_container_width=True)
        else:
            st.info("暂无图表（通常是菜单价格识别不足导致 price 缺失）。")

        st.subheader("Step 4｜生成 PDF（含图表页）")
        if st.button("生成 PDF", type="primary", key="btn_make_pdf"):
            with st.spinner("正在生成 PDF..."):
                pdf_path = render_pdf(st.session_state["report_text"], report_inputs)
            st.success("PDF 生成完成。")
            with open(pdf_path, "rb") as f:
                st.download_button("下载 PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf", key="btn_dl_pdf")
            st.caption(f"输出路径：{pdf_path}")
    else:
        st.info("完成餐厅选择 → 上传菜单/竞对 → 生成报告后，这里会显示预览与 PDF 下载。")


# =========================================================
# TAB 2: MENU SMART ADJUST (NEW)
# =========================================================
with tab_menu:
    st.subheader("菜单智能调整（按你要求的 Step C）")
    st.caption("逻辑：上传菜单 → AI识别 → 判断“套餐不足/整体偏贵” → 先调价 → 再组套餐 → 输出完整菜单结构（可下载CSV/JSON）。")

    st.divider()

    colL, colR = st.columns([2, 1], gap="large")
    with colR:
        st.markdown("### 参数")
        location = st.text_input("门店/商圈位置（用于同行价带估算）", value="San Francisco Bay Area", key="menu_location")
        cuisine = st.text_input("菜系/定位（用于同行价带估算）", value="港式/东南亚/中餐外卖", key="menu_cuisine")
        target_median = st.number_input("目标主食中位价（偏高判定基准）", value=float(DEFAULT_TARGET_MEDIAN_MAIN), step=0.5, key="menu_target_median")
        high_ratio = st.number_input("偏高阈值倍率（当前中位数 > 目标*倍率）", value=float(HIGH_PRICE_THRESHOLD_RATIO), step=0.01, key="menu_high_ratio")
        min_combos = st.number_input("最低套餐数量（不足则自动组建）", value=int(MIN_COMBOS_REQUIRED), step=1, key="menu_min_combos")
        front_limit = st.number_input("前台建议 SKU 上限（参考）", value=int(FRONT_LIMIT), step=1, key="menu_front_limit")
        st.caption("说明：你没上传竞品菜单也能跑。同行价带由 AI 估算；若 AI 失败，至少能按目标中位数做规则回调。")

    with colL:
        st.markdown("### 上传菜单（图片/CSV/Excel）")
        menu_files = st.file_uploader(
            "上传菜单文件（png/jpg/txt/csv/xlsx，支持多文件）",
            type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="smart_menu_files"
        )
        st.caption("建议：上传“平台菜单页截图”比“后台列表截图”更好识别。后台列表截图也能用，但缺口会多一些。")

        run_smart = st.button("🚀 开始智能调整并生成完整菜单", type="primary", disabled=not openai_key, key="btn_run_smart_menu")

    if run_smart:
        prog = st.progress(0)
        stat = st.empty()

        def step(p, msg):
            prog.progress(p)
            stat.info(msg)

        # Step A: Extract
        step(10, "正在解析菜单（识别菜品/价格/分类/套餐线索）...")
        menu_meta = extract_menu_with_openai(menu_files or [], openai_key, model, label="SMART_MENU")
        items = normalize_extracted_items(menu_meta)

        if not items:
            st.error("未识别到任何菜品。建议换更清晰的菜单图片，或上传 CSV/Excel。")
            st.stop()

        typed0 = split_by_type(items)
        mains0 = [m for m in typed0["main"] if isinstance(m.get("price"), (int, float)) and m["price"] and m["price"] > 0]
        combos0 = typed0["combo"]

        step(25, "Step C：检查套餐 & 检查整体价格是否偏高（同行价带估算）...")

        # Step C-1: overpriced check (rule)
        overpriced, cur_med = is_menu_overpriced(mains0, float(target_median), float(high_ratio)) if mains0 else (False, None)

        meta = {
            "current_median_main": cur_med,
            "target_median_main": float(target_median),
            "overpriced_by_rule": overpriced,
            "combo_count_detected": len(combos0),
        }

        # Step C-2: If overpriced -> first adjust prices (AI first, fallback rule)
        adjusted_items = items
        ai_meta = {}
        if overpriced:
            step(40, "检测到整体偏贵：先进行 AI 同行价带校准（失败则规则缩放回调）...")
            adjusted_items_ai, ai_meta = ai_market_benchmark(
                items=items,
                location=location,
                cuisine=cuisine,
                target_median_main=float(target_median),
                api_key=openai_key,
                model=model
            )

            # If AI provides usable adjustment, apply it; else fallback to rule scaling
            if ai_meta and not ai_meta.get("ai_error"):
                adjusted_items = adjusted_items_ai
                meta["price_adjustment_mode"] = "ai_market_benchmark"
                meta["ai_market_meta"] = ai_meta
            else:
                # fallback: scale to target median
                if cur_med and cur_med > 0:
                    scale = float(target_median) / float(cur_med)
                    adjusted_items = scale_prices(items, scale=scale)
                    meta["price_adjustment_mode"] = "rule_scale"
                    meta["rule_scale"] = round(scale, 4)
                meta["ai_market_meta"] = ai_meta

        # Step C-3: If combo missing -> build combos AFTER pricing adjusted
        typed1 = split_by_type(adjusted_items)
        combos1 = typed1["combo"]
        combo_ok = detect_combo_system(combos1, int(min_combos))

        if not combo_ok:
            step(65, f"套餐不足（当前 {len(combos1)} < {int(min_combos)}）：先调价已完成 → 现在自动组建套餐...")
            adjusted_items, combo_meta = build_combos_if_missing(
                items=adjusted_items,
                location=location,
                cuisine=cuisine,
                api_key=openai_key,
                model=model
            )
            meta["combo_build_meta"] = combo_meta
        else:
            meta["combo_build_meta"] = {"note": "combos sufficient"}

        # Step Final: Build final menu
        step(85, "正在生成完整菜单结构（锚点/爆品/套餐/加购/饮品/隐藏层）...")
        final_menu = build_smart_menu(adjusted_items, front_limit=int(front_limit))
        flat_df = menu_to_flat_df(final_menu)

        step(100, "完成：已生成完整菜单结构。")

        st.success("✅ 菜单智能调整完成（按 Step C：先调价 → 再组套餐 → 输出完整菜单）")
        st.write(meta)

        st.divider()
        st.markdown("## 最终菜单结构预览")
        for section, lst in final_menu.items():
            st.markdown(f"### {section}（{len(lst)}）")
            for it in lst[:80]:
                nm = item_fullname(it)
                p = it.get("price")
                po = it.get("price_old", None)
                if isinstance(po, (int, float)) and isinstance(p, (int, float)) and po != p:
                    st.write(f"- {nm} — ${p:.2f}（原价 ${po:.2f}）")
                elif isinstance(p, (int, float)):
                    st.write(f"- {nm} — ${p:.2f}")
                else:
                    st.write(f"- {nm} — 价格缺失")

            if len(lst) > 80:
                st.caption(f"（此处仅预览前 80 项，完整请下载 CSV/JSON）")

        st.divider()
        st.markdown("## 下载")
        menu_json = json.dumps(final_menu, ensure_ascii=False, indent=2)
        st.download_button("下载菜单结构 JSON", data=menu_json, file_name="smart_menu.json", mime="application/json", key="dl_smart_json")

        st.dataframe(flat_df, use_container_width=True, height=520)
        st.download_button("下载菜单结构 CSV", data=flat_df.to_csv(index=False), file_name="smart_menu.csv", mime="text/csv", key="dl_smart_csv")
