import os
import re
import io
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
# Page Config (MUST be called once, at top-level)
# =========================================================
APP_TITLE = "AuraInsight 报告生成器（Trade Area & Growth Diagnostic）"
st.set_page_config(page_title=APP_TITLE, layout="wide")


# =========================================================
# Auth (Password gate via secrets.toml)
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

    # Login UI
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
# Config / Assets
# =========================================================
OUTPUT_DIR = "output"
ASSETS_DIR = "assets"
FONTS_DIR = os.path.join(ASSETS_DIR, "fonts")

BG_COVER = os.path.join(ASSETS_DIR, "bg_cover.png")
BG_CONTENT = os.path.join(ASSETS_DIR, "bg_content.png")

# Font strategy:
# - Prefer static fonts if you have them.
# - If only variable fonts exist, we still register as a fallback.
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
# Data Models
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
    order_upload_meta: Dict[str, Any]

    competitors: List[CompetitorInput]
    extra_business_context: str

    acs: Optional[Dict[str, Any]]
    tract_info: Optional[Dict[str, Any]]
    restaurant_google: Dict[str, Any]


# =========================================================
# Fonts
# =========================================================
def register_aurainsight_fonts():
    # Chinese
    if os.path.exists(FONT_NOTO_REG):
        pdfmetrics.registerFont(TTFont("Noto", FONT_NOTO_REG))
    elif os.path.exists(FONT_NOTO_VAR):
        pdfmetrics.registerFont(TTFont("Noto", FONT_NOTO_VAR))

    if os.path.exists(FONT_NOTO_BOLD):
        pdfmetrics.registerFont(TTFont("Noto-Bold", FONT_NOTO_BOLD))

    # English
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


# =========================================================
# Helpers
# =========================================================
def draw_bg(c: canvas.Canvas, bg_path: str):
    if bg_path and os.path.exists(bg_path):
        c.drawImage(bg_path, 0, 0, width=PAGE_W, height=PAGE_H, mask="auto")

def wrap_lines_by_chars(text: str, max_chars: int) -> List[str]:
    lines: List[str] = []
    for para in text.splitlines():
        para = para.rstrip()
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=max_chars, break_long_words=False, replace_whitespace=False))
    return lines

def sanitize_text(text: str) -> str:
    # Remove markdown artifacts if leaked
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.M)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = text.replace("```", "").replace("`", "")
    text = text.replace("•", "-")
    return text.strip()

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
                sections.append((cur_title.replace("【", "").replace("】", "").strip(), "\n".join(cur_body).strip()))
            cur_title = p
            cur_body = []
        else:
            cur_body.append(p)
    if cur_title is not None:
        sections.append((cur_title.replace("【", "").replace("】", "").strip(), "\n".join(cur_body).strip()))
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
        "name","formatted_address","rating","user_ratings_total",
        "types","url","website","formatted_phone_number",
        "opening_hours","reviews","geometry"
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
# US Census (ACS)
# =========================================================
def census_tract_from_latlng(lat: float, lng: float) -> Optional[Dict[str, str]]:
    url = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
    params = {"x": lng, "y": lat, "benchmark": "Public_AR_Current", "vintage": "Current_Current", "format": "json"}
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


# =========================================================
# Yelp Fusion API (optional)
# =========================================================
def yelp_search_business(name_or_addr: str, location: str, api_key: str, limit: int = 3) -> Dict[str, Any]:
    # We do a pragmatic search. Yelp may not match perfectly; user can refine competitor text.
    url = "https://api.yelp.com/v3/businesses/search"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"term": name_or_addr, "location": location, "limit": limit}
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
# OpenAI (Responses API) - Text + Vision extraction
# =========================================================
def openai_responses(api_key: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
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
    # Supports txt/csv/xlsx
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".txt"):
        try:
            return raw.decode("utf-8", errors="ignore")[:30000]
        except Exception:
            return str(raw[:2000])

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
            return df.head(200).to_csv(index=False)[:30000]
        except Exception:
            uploaded_file.seek(0)
            return "CSV读取失败。"

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            df = pd.read_excel(uploaded_file)
            uploaded_file.seek(0)
            return df.head(200).to_csv(index=False)[:30000]
        except Exception:
            uploaded_file.seek(0)
            return "Excel读取失败。"

    return ""

def extract_menu_with_openai(
    files: List[Any],
    api_key: str,
    model: str,
    label: str
) -> Dict[str, Any]:
    """
    Extract menu items/prices/promos from uploaded files.
    For images: use OpenAI vision via Responses API.
    For csv/xlsx/txt: send as text.
    Returns: structured dict (best-effort).
    """
    if not files:
        return {"label": label, "files": [], "extracted": {"note": "no files"}}

    extracted_items = []
    promos = []
    categories = set()
    notes = []

    for f in files:
        fname = f.name
        lower = fname.lower()
        file_meta = {"name": fname, "size": getattr(f, "size", None)}
        content_blocks = []

        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            # Use vision: send image as base64 data URL
            b = f.read()
            f.seek(0)
            import base64
            b64 = base64.b64encode(b).decode("utf-8")
            mime = "image/png" if lower.endswith(".png") else "image/jpeg"
            data_url = f"data:{mime};base64,{b64}"

            prompt = (
                f"你是餐厅外卖菜单解析器。请从这张菜单图片中识别：\n"
                f"1) 菜品名称（中英文都要尽量抓到）\n"
                f"2) 价格（货币符号/数字）\n"
                f"3) 套餐结构（如加价项、组合、第二件折扣）\n"
                f"4) 平台营销活动/促销文案（如满减、免配送费、折扣）\n"
                f"输出必须是JSON，结构："
                f'{{"items":[{{"name":"", "price":"", "category":"", "notes":""}}], "promos":[...], "platform_hints":[...], "quality_flags":[...]}}。\n'
                f"不要输出任何非JSON内容。"
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

                # Parse JSON best-effort
                m = re.search(r"\{.*\}", text_out, flags=re.S)
                if not m:
                    notes.append(f"{fname}: vision输出无法解析为JSON。")
                    continue
                obj = json.loads(m.group(0))
                for it in obj.get("items", []):
                    extracted_items.append(it)
                    if it.get("category"):
                        categories.add(str(it.get("category")))
                for p in obj.get("promos", []):
                    promos.append(p)
            except Exception as e:
                notes.append(f"{fname}: vision解析失败: {str(e)[:200]}")
            continue

        # Text-like files
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
                f"你是餐厅外卖菜单解析器。以下是菜单内容（可能来自CSV/Excel/TXT）。\n"
                f"请提取菜品、价格、分类、加价项/套餐结构、促销信息。\n"
                f"输出必须是JSON："
                f'{{"items":[{{"name":"", "price":"", "category":"", "notes":""}}], "promos":[...], "platform_hints":[...], "quality_flags":[...]}}。\n'
                f"不要输出任何非JSON内容。\n\n"
                f"菜单原文开始：\n{text_blob}\n菜单原文结束。"
            )
            try:
                text_out = openai_text(prompt, api_key, model=model, temperature=0.2)
                m = re.search(r"\{.*\}", text_out, flags=re.S)
                if not m:
                    notes.append(f"{fname}: 文本解析输出无法解析为JSON。")
                    continue
                obj = json.loads(m.group(0))
                for it in obj.get("items", []):
                    extracted_items.append(it)
                    if it.get("category"):
                        categories.add(str(it.get("category")))
                for p in obj.get("promos", []):
                    promos.append(p)
            except Exception as e:
                notes.append(f"{fname}: 文本解析失败: {str(e)[:200]}")
            continue

        notes.append(f"{fname}: 不支持的文件类型（建议 png/jpg/txt/csv/xlsx）。")

    # Trim huge payloads
    extracted_items = extracted_items[:600]
    promos = promos[:80]

    return {
        "label": label,
        "files": [{"name": f.name, "type": f.type if hasattr(f, "type") else ""} for f in files],
        "extracted": {
            "items": extracted_items,
            "promos": promos,
            "categories_guess": sorted(list(categories))[:80],
            "notes": notes[:80],
        }
    }


# =========================================================
# Orders upload meta
# =========================================================
def summarize_uploaded_orders(files: List[Any]) -> Dict[str, Any]:
    meta = {"files": [], "notes": "Provide platform exports (CSV). System summarizes schema for analysis."}
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
            meta["files"].append({"name": getattr(f, "name", "uploaded"), "error": str(e)[:200]})
    return meta


# =========================================================
# Report Prompt (forces detailed, long, actionable output)
# =========================================================
def build_prompt(inputs: ReportInputs, model_hint: str = "") -> str:
    # Compact but rich data blob
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
            "assumption_note": "ACS is tract-level proxy; treat as directional for 3–4 miles trade area."
        },
        "own_menu": inputs.own_menu_meta,
        "orders_meta": inputs.order_upload_meta,
        "competitors": [
            {
                "name_or_address": c.name_or_address,
                "notes": c.notes,
                "google": c.google,
                "yelp": c.yelp,
                "menu": c.menu_files_meta,
            }
            for c in inputs.competitors
        ],
        "extra_business_context": inputs.extra_business_context,
    }

    # HARD constraints: no markdown, include pricing tables, bundles, steps, KPIs, 2-week tests.
    return f"""
你是 AuraInsight 的北美餐饮增长咨询顾问（偏外卖+本地商圈增长）。你将收到一个 JSON 数据包（含：商圈ACS、门店与竞对的Google/Yelp信息、门店与竞对的外卖菜单识别结果、订单报表字段摘要）。
你的任务：输出一份“可直接交付给老板执行”的《Trade Area & Growth Diagnostic》报告正文，用于生成 PDF。

必须遵守（违背任何一条都算失败）：
1) 严禁输出 Markdown（不要出现：#、##、**、```、|---|、[]()）。
2) 章节标题只能使用： 【章节标题】 这种格式；列表用 “- ”。
3) 每一章开头先写 3–6 条 Key Takeaways（尽量带数字/范围/逻辑）。
4) 所有建议必须包含五件套：【动作】【原因】【预期影响】【KPI】【两周验证方法】。
5) 必须落地到“菜品与价格”：至少给出 2 个价格锚点策略 + 2 套套餐方案 + 10 个具体菜品改价/改名/上架/下架建议（每条要写当前价/建议价/理由/对标竞对）。
   - 如果 JSON 中没有“当前价”，就写 “当前价：待补齐（来自菜单识别不完整）”，但仍需给出建议价与逻辑。
6) 必须应用并解释为什么适用：STP、JTBD、Menu Engineering（星/牛/谜/狗）、Anchoring（锚点定价）、ERRC（蓝海四动作）、Behavioral Economics（至少3条：损失厌恶/稀缺性/默认选项/社会证明等）。
7) 报告必须足够长：目标生成后 PDF 至少 6–7 页。请写得具体、步骤化、带表格/清单（但不要 Markdown 表格，用“表格:”后CSV格式最多10行/表）。
8) 对“商圈人口/收入/客群”做完整分析：把 ACS 数据转成“可经营决策”（价格带、客单、品类选择、出餐速度、营销渠道、时段策略）。
9) 必须包含“门店菜单深度分析（基于上传菜单识别）”与“竞对菜单深度分析（基于上传竞对菜单识别 + Google/Yelp）”。

输出章节顺序必须如下（标题一字不差）：
【Executive Summary】
【1. Trade Area & Demographics】
【2. Customer Segments & JTBD】
【3. Demand, Occasion & Menu Positioning】
【4. Competitive Landscape (Google + Yelp + Menu)】
【5. Pricing, Anchors & Promo Economics】
【6. Menu Architecture & Menu Engineering】
【7. Platform Growth Playbook (30/60/90)】
【8. Measurement System & Experiment Design】
【Appendix A: Own Menu Deep Dive】
【Appendix B: Competitor Menu Deep Dive】
【Data Gaps & How to Collect】

输入 JSON：
{json.dumps(blob, ensure_ascii=False, indent=2)}

开始输出报告正文：
""".strip()


# =========================================================
# Auto-extend report if too short
# =========================================================
def ensure_long_enough(report_text: str, api_key: str, model: str, min_chars: int = 12000) -> str:
    t = sanitize_text(report_text)
    if len(t) >= min_chars:
        return t

    # Ask model to expand specific parts to increase depth, not fluff
    expand_prompt = f"""
你将收到一份报告正文（无Markdown）。请在不改变既有结构标题的前提下，显著扩写内容，使其更“可执行、数据化、菜单定价更细”。
扩写重点：
- 把【5】【6】【Appendix A】【Appendix B】扩成更细的“当前价/建议价/理由/对标竞对/预计毛利与销量影响/两周实验设计”。
- 增加至少2套套餐的详细组成、定价锚点与心理学理由。
- 增加至少10条“可直接照做”的运营动作清单（含KPI和2周验证）。
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
# PDF Rendering (fix cover duplication + spacing)
# =========================================================
def draw_footer(c: canvas.Canvas, report_date: str, page_num: int):
    c.setFillColor(colors.HexColor("#7A7A7A"))
    c.setFont(f_en(False), 8)
    c.drawString(0.75 * inch, 0.55 * inch, f"Confidential | Generated by AuraInsight | {report_date}")
    c.drawRightString(PAGE_W - 0.75 * inch, 0.55 * inch, f"Page {page_num}")
    c.setFillColor(colors.black)

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

    # IMPORTANT: Avoid brand/title duplication.
    # Only print restaurant info (safe overlay).
    c.setFillColor(colors.HexColor("#111111"))

    # Put restaurant info lower area to avoid background logo/title region.
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

    # ---- Content pages ----
    page_num = 1
    draw_bg(c, BG_CONTENT)

    left = 0.90 * inch

    # Push content DOWN to avoid background header overlap.
    top = PAGE_H - 1.55 * inch
    y = top - 0.45 * inch

    # Spacing tuning
    body_font_size = 10
    line_gap = 15     # more breathing room
    para_gap = 10
    heading_gap = 22

    def new_page():
        nonlocal y, page_num
        draw_footer(c, inputs.report_date, page_num)
        c.showPage()
        page_num += 1
        draw_bg(c, BG_CONTENT)
        y = top - 0.45 * inch

    def draw_heading(title: str):
        nonlocal y
        if y < 1.8 * inch:
            new_page()
        c.setFillColor(colors.black)
        font = f_cn(True) if any("\u4e00" <= ch <= "\u9fff" for ch in title) else f_en(True)
        c.setFont(font, 13)
        c.drawString(left, y, title[:120])
        y -= heading_gap

    def draw_body(text: str):
        nonlocal y
        max_chars = 105
        for line in wrap_lines_by_chars(text, max_chars):
            if y < 1.25 * inch:
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
            if body:
                draw_body(body)

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
    radius_miles = st.slider("商圈半径（miles）", 1.0, 6.0, 4.0, 0.5)
    nearby_radius_m = st.slider("Google Nearby 搜索半径（米）", 300, 3000, 1200, 100)

    st.divider()
    logout_button()

    st.divider()
    st.caption("Built by c8geek")
    # LinkedIn link (edit to your actual profile URL if needed)
    st.markdown("[LinkedIn](https://www.linkedin.com/)")

# Hide background path + content header controls (per your request)
# (No UI controls; we use BG_COVER/BG_CONTENT constants.)

st.title(APP_TITLE)

if not google_key:
    st.warning("未检测到 GOOGLE_MAPS_API_KEY，请在 .streamlit/secrets.toml 配置。")
if not openai_key:
    st.warning("未检测到 OPENAI_API_KEY，请在 .streamlit/secrets.toml 配置。")
if not yelp_key:
    st.info("未检测到 YELP_API_KEY（可选）。不影响生成报告，但竞对 Yelp 维度会缺失。")


# =========================================================
# Step 1: Choose restaurant via Google Nearby
# =========================================================
st.subheader("Step 1｜输入地址 → 搜索附近餐厅")
address_input = st.text_input("输入地址（用于定位并搜索附近餐厅）", value="2406 19th Ave, San Francisco, CA 94116")

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


# =========================================================
# Step 2: Upload menus + ACS + Competitors (Google/Yelp + menus)
# =========================================================
if place_details:
    st.subheader("Step 2｜上传菜单 + 自动商圈画像（ACS） + 竞对（Google/Yelp + 竞对菜单上传）")

    # Restaurant fields
    loc = place_details.get("geometry", {}).get("location", {}) or {}
    rest_lat = float(loc.get("lat")) if loc.get("lat") is not None else None
    rest_lng = float(loc.get("lng")) if loc.get("lng") is not None else None

    col1, col2 = st.columns([1, 1])
    with col1:
        restaurant_en = st.text_input("餐厅英文名", value=place_details.get("name", ""))
        restaurant_cn = st.text_input("餐厅中文名（可选）", value="")
        formatted_address = st.text_input("餐厅地址", value=place_details.get("formatted_address", address_input))
        st.caption(f"Google：⭐{place_details.get('rating','')}（{place_details.get('user_ratings_total','')} reviews）")
        extra_context = st.text_area(
            "补充业务背景（可选）",
            value="例如：经营年限、主打菜、目标客群、当前痛点（单量/评分/利润/人手等）。",
            height=120
        )

    with col2:
        st.markdown("### 门店外卖菜单上传（替代平台链接）")
        own_menu_files = st.file_uploader(
            "上传门店菜单（png/jpg/txt/csv/xlsx，支持多文件）",
            type=["png", "jpg", "jpeg", "webp", "txt", "csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="own_menu_files"
        )
        st.caption("系统会识别：菜品、价格、分类、套餐结构、促销文案；最终写入 PDF 的 Appendix A。")

    # ACS
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
                "Note": "ACS 为 tract 级别代理，作为 3–4 miles 商圈近似画像；报告中会明确该假设。"
            })

    # Competitors table (keep style, but no preset names, no platform columns)
    st.markdown("### 竞对信息（可增删行：用于差异化与竞品分析）")
    if "comp_rows" not in st.session_state:
        st.session_state.comp_rows = 3

    cA, cB, cC = st.columns([1, 1, 2])
    with cA:
        if st.button("➕ 添加竞对"):
            st.session_state.comp_rows += 1
    with cB:
        if st.button("➖ 删除最后一个", disabled=st.session_state.comp_rows <= 1):
            st.session_state.comp_rows = max(1, st.session_state.comp_rows - 1)
    with cC:
        st.caption("每个竞对：填写名称/地址 → 拉取 Google/Yelp → 上传该竞对菜单文件（用于 Appendix B）。")

    # Render competitor inputs + uploads
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

            # Pull Google + Yelp (optional)
            pull_col1, pull_col2 = st.columns([1, 2])
            with pull_col1:
                pull = st.button(f"拉取竞对 Google + Yelp（#{i+1}）", key=f"pull_comp_{i}", disabled=not google_key)
            with pull_col2:
                st.caption("说明：Google 用于位置/评分/营业时间/评论片段；Yelp 用于价格带/分类/评论示例（如配置了YELP_API_KEY）。")

            comp_google = st.session_state.get(f"comp_google_{i}", {})
            comp_yelp = st.session_state.get(f"comp_yelp_{i}", {})

            if pull and comp_name.strip():
                # Google
                pid = google_textsearch_place_id(f"{comp_name} {formatted_address}", google_key)
                if pid:
                    comp_google = google_place_details(pid, google_key)
                else:
                    comp_google = {"error": "Google textsearch failed"}

                st.session_state[f"comp_google_{i}"] = comp_google

                # Yelp (optional)
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

            # Summaries
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
                    menu_files_meta={"label": f"Competitor #{i+1}", "files": [{"name": f.name} for f in (comp_menu_files or [])]},
                    google=comp_google if isinstance(comp_google, dict) else {},
                    yelp=comp_yelp if isinstance(comp_yelp, dict) else {},
                )
            )

    if comp_summary_rows:
        st.dataframe(pd.DataFrame(comp_summary_rows), use_container_width=True)

    # Order exports upload
    with st.expander("上传订单报表（CSV，可选：用于时段/客单/热销/KPI）", expanded=False):
        uploads = st.file_uploader("上传平台订单导出 CSV（可多选）", type=["csv"], accept_multiple_files=True)
        order_meta = summarize_uploaded_orders(uploads) if uploads else {"files": [], "note": "No uploads"}
        if uploads:
            st.json(order_meta)

    # =========================================================
    # Step 3: Generate report text
    # =========================================================
    st.subheader("Step 3｜生成深度分析报告（咨询级，目标≥6–7页）")
    report_date = dt.datetime.now().strftime("%m/%d/%Y")

    if st.button("生成报告内容", type="primary", disabled=not openai_key):
        with st.spinner("正在解析门店菜单（OpenAI 识别菜品/价格/促销）..."):
            own_menu_meta = extract_menu_with_openai(own_menu_files or [], openai_key, model, label="OWN_MENU")

        # For each competitor, parse its menu files (if any)
        parsed_competitors: List[CompetitorInput] = []
        for i, comp in enumerate(comp_inputs):
            # retrieve files again from uploader keys
            files = st.session_state.get(f"comp_menu_files_{i}", None)  # uploader stores in widget, not session_state
            # Streamlit does not store uploaded objects in session_state by default;
            # we re-read via st.file_uploader return value earlier; but here we only have meta.
            # So: we re-run extraction using current widget read:
            # (We cannot access it unless we kept it. We'll re-bind by reading from widget key via st.session_state is unreliable.)
            # Workaround: store extracted meta in session_state at upload-time is complex; simplest: re-run extract by asking user to click after files are present (which they are).
            # We will instead do extraction by scanning the widget's returned value again.
            # To ensure we have it, we re-rendered in loop; Streamlit keeps it accessible via same key:
            comp_files = st.session_state.get(f"_uploaded_{i}", None)

            # Better: just re-call file_uploader value by reading widget state is not stable.
            # So we do: create a new file_uploader? not possible.
            # Practical approach: in this run, use "menus_uploaded" info and skip if not available.
            # We'll ask user to re-run if competitor parsing empty.
            #
            # HOWEVER: Streamlit actually passes the file list to our variable comp_menu_files within loop.
            # We didn't keep it globally. We'll store it during loop next time:
            pass

        # We fix the above by re-extracting right now using already extracted outputs stored in session_state during loop.
        # We'll store per competitor extraction on-demand here by asking OpenAI again using a lightweight prompt:
        with st.spinner("正在解析竞对菜单（如有上传）..."):
            competitors_full: List[CompetitorInput] = []
            for i in range(st.session_state.comp_rows):
                comp_name = st.session_state.get(f"comp_name_{i}", "").strip()
                comp_notes = st.session_state.get(f"comp_notes_{i}", "").strip()
                comp_google = st.session_state.get(f"comp_google_{i}", {}) or {}
                comp_yelp = st.session_state.get(f"comp_yelp_{i}", {}) or {}

                # The actual uploaded files list can be retrieved by re-creating a hidden uploader? Not possible.
                # So we instead re-use the menu extraction only when user uploaded; Streamlit keeps uploaded files accessible
                # via st.session_state for the uploader key in practice in most deployments.
                comp_files = st.session_state.get(f"comp_menu_files_{i}", None)
                # If not found, we'll skip extraction.
                if comp_files and isinstance(comp_files, list) and len(comp_files) > 0:
                    comp_menu_meta = extract_menu_with_openai(comp_files, openai_key, model, label=f"COMP_{i+1}")
                else:
                    comp_menu_meta = {"label": f"COMP_{i+1}", "files": [], "extracted": {"note": "no competitor menu uploaded or not accessible"}}

                competitors_full.append(
                    CompetitorInput(
                        name_or_address=comp_name,
                        notes=comp_notes,
                        menu_files_meta=comp_menu_meta,
                        google=comp_google,
                        yelp=comp_yelp,
                    )
                )

        # Restaurant Google details already
        restaurant_google = place_details

        # ACS
        tract_info = st.session_state.get("tract_info", None)
        acs_data = st.session_state.get("acs_data", None)

        inputs = ReportInputs(
            report_date=report_date,
            restaurant_cn=(restaurant_cn.strip() or restaurant_en.strip()),
            restaurant_en=restaurant_en.strip(),
            address=formatted_address.strip(),
            radius_miles=radius_miles,
            own_menu_meta=own_menu_meta,
            order_upload_meta=order_meta,
            competitors=competitors_full,
            extra_business_context=extra_context.strip(),
            acs=acs_data,
            tract_info=tract_info,
            restaurant_google=restaurant_google,
        )

        prompt = build_prompt(inputs)

        with st.spinner("正在生成咨询级报告（会更长、更细、含定价与套餐方案）..."):
            report_text = openai_text(prompt, openai_key, model=model, temperature=0.25)
            report_text = sanitize_text(report_text)
            report_text = ensure_long_enough(report_text, openai_key, model=model, min_chars=12000)

        st.session_state["report_text"] = report_text
        st.session_state["report_inputs"] = inputs
        st.success("报告内容已生成（已尽量保证≥6–7页的细度）。")


# =========================================================
# Step 4: Preview + PDF
# =========================================================
report_text = st.session_state.get("report_text", "")
report_inputs: Optional[ReportInputs] = st.session_state.get("report_inputs", None)

if report_text and report_inputs:
    st.subheader("预览（可编辑）")
    edited = st.text_area("报告正文（你可以直接修改）", value=report_text, height=520)
    st.session_state["report_text"] = sanitize_text(edited)

    st.subheader("Step 4｜生成 PDF（套用封面/内容页背景图）")

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
    st.info("完成餐厅选择 → 上传菜单/竞对 → 生成报告后，这里会显示预览与 PDF 下载。")
