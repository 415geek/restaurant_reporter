# app.py
# AuraInsight 报告生成器（Trade Area & Growth Diagnostic）
# FINAL: trade area population fix (multi-tract aggregation) + OpenAI streaming output + safer chart display

import os
import re
import io
import json
import time
import math
import hmac
import base64
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterable

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

    # legacy tract level (still kept)
    tract_info: Optional[Dict[str, Any]]
    acs_tract: Optional[Dict[str, Any]]

    # NEW: aggregated trade area
    trade_area_acs: Optional[Dict[str, Any]]
    trade_area_debug: Optional[Dict[str, Any]]

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
    if not text:
        return ""
    text = text.replace("```", "").replace("`", "")
    text = text.replace("•", "-")
    text = re.sub(r'^\s{0,3}#{1,6}\s*', '', text, flags=re.M)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    return text.strip()

def wrap_lines_by_pdf_width(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
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
    text = (text or "").strip()
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
# Geography helpers (NEW)
# =========================================================
EARTH_R_KM = 6371.0088

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))

def destination_point(lat: float, lon: float, bearing_deg: float, dist_km: float) -> Tuple[float, float]:
    # great-circle destination
    br = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    dr = dist_km / EARTH_R_KM
    lat2 = math.asin(math.sin(lat1)*math.cos(dr) + math.cos(lat1)*math.sin(dr)*math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br)*math.sin(dr)*math.cos(lat1), math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return (math.degrees(lat2), (math.degrees(lon2)+540) % 360 - 180)

def sample_points_in_circle(lat: float, lon: float, radius_km: float, rings: int = 4, per_ring: int = 16) -> List[Tuple[float, float]]:
    # include center + rings with evenly distributed bearings
    pts = [(lat, lon)]
    if rings <= 0:
        return pts
    for r in range(1, rings+1):
        frac = r / rings
        dist = radius_km * frac
        n = max(8, int(per_ring * frac))
        for i in range(n):
            b = (360.0 * i) / n
            pts.append(destination_point(lat, lon, b, dist))
    return pts


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
# Census ACS (tract-level fetch + NEW aggregated trade area)
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
    # key counts + medians
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

def aggregate_trade_area_acs(lat: float, lng: float, radius_miles: float, year: int = 2023,
                             rings: int = 4, per_ring: int = 18, request_delay_s: float = 0.0) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    NEW: “实际商圈人口统计”的近似实现
    - 在半径内采样多点 → 每点反查 tract → 去重 tract → 拉取每 tract ACS → 人口加权聚合
    - 注意：median_income / median_age 用人口加权均值（近似），不是严格合并中位数
    """
    debug = {"radius_miles": radius_miles, "year": year, "rings": rings, "per_ring": per_ring,
             "points": 0, "unique_tracts": 0, "tracts": [], "errors": []}

    radius_km = radius_miles * 1.609344
    pts = sample_points_in_circle(lat, lng, radius_km, rings=rings, per_ring=per_ring)
    debug["points"] = len(pts)

    tract_keys = {}
    for (plat, plng) in pts:
        try:
            tract = census_tract_from_latlng(plat, plng)
            if tract and tract.get("STATE") and tract.get("COUNTY") and tract.get("TRACT"):
                key = (tract["STATE"], tract["COUNTY"], tract["TRACT"])
                tract_keys[key] = tract
        except Exception as e:
            debug["errors"].append(f"tract_lookup_failed: {str(e)[:120]}")
        if request_delay_s:
            time.sleep(request_delay_s)

    debug["unique_tracts"] = len(tract_keys)
    debug["tracts"] = [{"state": k[0], "county": k[1], "tract": k[2], "name": v.get("NAME","")} for k, v in tract_keys.items()][:300]

    if not tract_keys:
        return None, debug

    tracts_acs = []
    for key, tract in tract_keys.items():
        try:
            acs = acs_5y_profile(key[0], key[1], key[2], year=year)
            if acs and acs.get("pop_total") is not None and acs.get("pop_total") > 0:
                tracts_acs.append(acs)
        except Exception as e:
            debug["errors"].append(f"acs_failed_{key}: {str(e)[:120]}")
        if request_delay_s:
            time.sleep(request_delay_s)

    if not tracts_acs:
        return None, debug

    # aggregate
    pop_sum = sum([a.get("pop_total", 0.0) or 0.0 for a in tracts_acs])
    if pop_sum <= 0:
        return None, debug

    def sum_field(field: str) -> float:
        return float(sum([(a.get(field, 0.0) or 0.0) for a in tracts_acs]))

    # counts
    white = sum_field("white")
    black = sum_field("black")
    asian = sum_field("asian")
    hispanic = sum_field("hispanic")
    owner = sum_field("owner_occ")
    renter = sum_field("renter_occ")

    # weighted medians (approx)
    def wavg(field: str) -> Optional[float]:
        num = 0.0
        den = 0.0
        for a in tracts_acs:
            v = a.get(field, None)
            p = a.get("pop_total", 0.0) or 0.0
            if v is None or p <= 0:
                continue
            num += float(v) * float(p)
            den += float(p)
        return (num/den) if den > 0 else None

    median_income_w = wavg("median_income")
    median_age_w = wavg("median_age")

    occ_total = owner + renter
    out = {
        "year": year,
        "method": "multi-tract aggregation via sampled points within radius",
        "radius_miles": radius_miles,
        "center": {"lat": lat, "lng": lng},
        "tracts_used": len(tracts_acs),
        "pop_total": pop_sum,
        "white": white,
        "black": black,
        "asian": asian,
        "hispanic": hispanic,
        "pct_white": (white / pop_sum) if pop_sum > 0 else None,
        "pct_black": (black / pop_sum) if pop_sum > 0 else None,
        "pct_asian": (asian / pop_sum) if pop_sum > 0 else None,
        "pct_hispanic": (hispanic / pop_sum) if pop_sum > 0 else None,
        "owner_occ": owner,
        "renter_occ": renter,
        "pct_owner": (owner / occ_total) if occ_total > 0 else None,
        "pct_renter": (renter / occ_total) if occ_total > 0 else None,
        "median_income_wavg": median_income_w,
        "median_age_wavg": median_age_w,
        "notes": [
            "Population / race / owner-renter are aggregated (sum) across tracts discovered within the radius.",
            "Median income / median age are population-weighted averages of tract medians (approximation).",
            "This is much closer to 'trade area' reality than a single tract, but still an approximation."
        ],
    }
    return out, debug


# =========================================================
# OpenAI Responses API (streaming)
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

def openai_text_stream(prompt: str, api_key: str, model: str, temperature: float = 0.25) -> Iterable[str]:
    """
    Stream output_text chunks from Responses API.
    UI 会实时看到生成内容（不是“内部思考”，而是最终输出的实时增量）。
    """
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {"model": model, "input": prompt, "temperature": temperature, "stream": True}

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
        r.raise_for_status()
        buffer = ""
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                obj = json.loads(data_str)
            except Exception:
                continue

            # Try to extract incremental text
            # Different deployments may emit slightly different shapes; we handle common cases.
            # Case A: {type:"response.output_text.delta", delta:"..."}
            t = ""
            if isinstance(obj, dict):
                if obj.get("type") in ("response.output_text.delta", "response.output_text"):
                    t = obj.get("delta") or obj.get("text") or ""
                # Case B: full response snapshots
                if not t and "output" in obj:
                    # might be snapshot; we ignore to avoid duplication
                    t = ""
            if t:
                yield t


# =========================================================
# Menu extraction (unchanged core)
# =========================================================
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
# Menu Stats + Charts (safer)
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

def build_charts(own_df: pd.DataFrame) -> Dict[str, bytes]:
    charts: Dict[str, bytes] = {}
    if own_df is None or own_df.empty:
        return charts

    # Price hist
    fig = plt.figure()
    own_df["price"].dropna().plot(kind="hist", bins=18)
    plt.title("Own Menu Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Count")
    charts["chart_own_price_hist"] = make_png(fig)

    # Category bar
    fig = plt.figure()
    own_df["category"].value_counts().head(12).plot(kind="bar")
    plt.title("Own Menu Category Mix (Top 12)")
    plt.xlabel("Category")
    plt.ylabel("Items")
    charts["chart_own_category_bar"] = make_png(fig)

    # Price tiers
    bins = [0, 8, 12, 16, 20, 25, 35, 999]
    labels = ["<$8", "$8-12", "$12-16", "$16-20", "$20-25", "$25-35", "$35+"]
    tier = pd.cut(own_df["price"], bins=bins, labels=labels, include_lowest=True)
    fig = plt.figure()
    tier.value_counts().reindex(labels).fillna(0).plot(kind="bar")
    plt.title("Own Menu Price Tiers")
    plt.xlabel("Tier")
    plt.ylabel("Items")
    charts["chart_own_price_tiers"] = make_png(fig)

    return charts

def safe_st_image(img_bytes: Any, caption: str = ""):
    # Avoid Streamlit TypeError when bytes is None or unexpected
    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 50:
        st.image(img_bytes, use_container_width=True, caption=caption if caption else None)
    else:
        st.warning("图表数据缺失或格式不支持，已跳过。")


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
# Prompt (餐饮业态分析报告)
# =========================================================
RESTAURANT_SYSTEM_STYLE = """
你是一个餐饮行业的数据驱动增长咨询顾问，擅长对餐饮门店与餐饮品牌进行商圈诊断、平台与堂食结构分析、菜单与价格优化、虚拟品牌设计，并输出可落地执行的增长方案。报告风格参考专业餐饮咨询公司的 trade area & growth diagnostic 报告，有深度、有数据支撑，并给出具体、可拆解的执行步骤与 KPI。

请严格围绕「餐饮门店/餐饮品牌」输出报告，按照以下结构与逻辑编写（可调整小标题措辞，但保留模块和思路）：

0. 报告基本信息与摘要（Executive Summary）
1. 商圈与客群结构（Trade Area & Demand Fundamentals）
2. 门店与品牌资产结构（Store & Brand Assets）
3. 渠道与平台生态（Dine-in, Takeout & Platform Ecosystem）
4. 竞争格局与相对位置（Competitive Landscape）
5. 价格带与订单经济（Pricing & Order Economics）
6. 时段与场景需求结构（Time-based & Occasion-based Demand）
7. 菜单架构与虚拟品牌策略（Menu Architecture & Virtual Brand）
8. 战略结论与执行路线图（Strategic Implications & Execution Roadmap）

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
            "trade_area_acs": inputs.trade_area_acs,
            "trade_area_debug": inputs.trade_area_debug,
            "tract_info_center": inputs.tract_info,
            "acs_tract_center": inputs.acs_tract,
            "assumption_note": (
                "优先使用 trade_area_acs（半径内多 tract 聚合）作为商圈人口统计；"
                "acs_tract_center 仅作为中心点 tract 的对照参考。"
            )
        },
        "own_menu": inputs.own_menu_meta,
        "orders_meta": inputs.orders_meta,
        "competitors": [],
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
5) Must interpret charts by name (chart_own_price_hist, chart_own_category_bar, chart_own_price_tiers) when available.
6) Must include Data Gaps & How to Collect (as an explicit section).
7) Must include tables where appropriate (use plain text tables, not Markdown).
8) For population & demographics, MUST use trade_area.trade_area_acs.pop_total/pct_* as primary if available, and explain methodology briefly.

Input JSON:
{json.dumps(blob, ensure_ascii=False, indent=2)}

Start writing now:
""".strip()

def ensure_long_enough(report_text: str, api_key: str, model: str, lang: str, min_chars: int = 14000) -> str:
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
# PDF Render
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

    safe_name = "".join([ch if ch.isalnum() or ch in ("_", "-", " ") else "_" for ch in (inputs.restaurant_en or "Report")]).strip()
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
        available_w = PAGE_W - left - 0.90 * inch

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
        # NEW trade area sampling
        rings = st.slider("商圈采样环数（越大越准/越慢）", 2, 6, 4, 1)
        per_ring = st.slider("每环采样点基数（越大越准/越慢）", 10, 30, 18, 1)
    else:
        radius_miles = 4.0
        nearby_radius_m = 1200
        rings = 4
        per_ring = 18

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

tab_food = st.tabs(["餐饮业态分析报告"])[0]


# =========================================================
# Tab 1: 餐饮业态分析报告（FINAL）
# =========================================================
with tab_food:
    st.subheader("Step 1｜输入地址 → 搜索附近餐厅")
    address_input = st.text_input("输入地址（用于定位并搜索附近餐厅）", value="1970 Lewelling Blvd, San Leandro, CA 94579", key="addr_search_food")

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
        st.subheader("Step 2｜上传菜单 + 自动商圈人口统计（真实 trade area 聚合）")
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
                cA, cB = st.columns([1, 1])
                with cA:
                    if st.button("获取 Trade Area 人口统计（推荐：按半径聚合）", key="food_btn_trade_area"):
                        with st.spinner("正在计算商圈人口统计（采样 → tract → ACS → 聚合）..."):
                            trade_area_acs, trade_debug = aggregate_trade_area_acs(
                                rest_lat, rest_lng,
                                radius_miles=radius_miles,
                                year=2023,
                                rings=rings,
                                per_ring=per_ring,
                                request_delay_s=0.0
                            )
                        st.session_state["food_trade_area_acs"] = trade_area_acs
                        st.session_state["food_trade_area_debug"] = trade_debug
                        if trade_area_acs:
                            st.success("Trade Area 聚合完成（比单 tract 更接近真实商圈）。")
                        else:
                            st.warning("Trade Area 聚合失败：可能 Census 服务暂时不可用或采样点无法匹配。")

                with cB:
                    if st.button("获取中心点 Tract（仅对照）", key="food_btn_tract_only"):
                        tract_info = census_tract_from_latlng(rest_lat, rest_lng)
                        acs_data = None
                        if tract_info:
                            acs_data = acs_5y_profile(tract_info["STATE"], tract_info["COUNTY"], tract_info["TRACT"], year=2023)
                        st.session_state["food_tract_info"] = tract_info
                        st.session_state["food_acs_tract"] = acs_data
                        st.success("中心点 tract 数据已获取（对照用）。")

            else:
                st.info("未能从 Google Place Details 获取坐标，无法调用 ACS。")

            trade_area_acs = st.session_state.get("food_trade_area_acs", None)
            trade_debug = st.session_state.get("food_trade_area_debug", None)
            if trade_area_acs:
                st.write({
                    "ACS Year": trade_area_acs.get("year"),
                    "Method": trade_area_acs.get("method"),
                    "Radius (miles)": trade_area_acs.get("radius_miles"),
                    "Tracts Used": trade_area_acs.get("tracts_used"),
                    "Population (trade area est.)": int(trade_area_acs.get("pop_total", 0)),
                    "% Asian": None if trade_area_acs.get("pct_asian") is None else f"{trade_area_acs.get('pct_asian')*100:.1f}%",
                    "% White": None if trade_area_acs.get("pct_white") is None else f"{trade_area_acs.get('pct_white')*100:.1f}%",
                    "% Hispanic": None if trade_area_acs.get("pct_hispanic") is None else f"{trade_area_acs.get('pct_hispanic')*100:.1f}%",
                    "% Black": None if trade_area_acs.get("pct_black") is None else f"{trade_area_acs.get('pct_black')*100:.1f}%",
                    "% Renter": None if trade_area_acs.get("pct_renter") is None else f"{trade_area_acs.get('pct_renter')*100:.1f}%",
                    "Median Age (wavg approx)": None if trade_area_acs.get("median_age_wavg") is None else round(trade_area_acs.get("median_age_wavg"), 1),
                    "Median HH Income (wavg approx)": None if trade_area_acs.get("median_income_wavg") is None else f"${int(trade_area_acs.get('median_income_wavg')):,}",
                    "Note": "population/race/owner-renter 为 tract 汇总；中位数为加权近似。"
                })
                with st.expander("Trade Area 计算细节（debug）", expanded=False):
                    st.json(trade_debug)

            tract_info = st.session_state.get("food_tract_info", None)
            acs_tract = st.session_state.get("food_acs_tract", None)
            if acs_tract:
                st.caption("中心点 tract（仅对照）")
                st.write({
                    "Tract Geography": acs_tract.get("name"),
                    "Population (tract)": None if acs_tract.get("pop_total") is None else int(acs_tract.get("pop_total")),
                    "% Asian (proxy)": None if acs_tract.get("pct_asian") is None else f"{acs_tract.get('pct_asian')*100:.1f}%",
                    "% Renter (proxy)": None if acs_tract.get("pct_renter") is None else f"{acs_tract.get('pct_renter')*100:.1f}%",
                })

        with st.expander("上传订单报表（CSV，可选：用于时段/客单/热销/KPI）", expanded=False):
            order_files = st.file_uploader("上传平台订单导出 CSV（可多选）", type=["csv"], accept_multiple_files=True, key="food_orders_report")
            orders_meta = summarize_orders(order_files or [])
            if order_files:
                st.json(orders_meta)
        if "orders_meta" not in locals():
            orders_meta = {"files": [], "note": "No uploads"}

        st.subheader("Step 3｜生成报告内容（实时看到 AI 输出流）")
        st.caption("你会看到 AI 正在生成的内容片段（实时流式输出）。这不是“内部思考链”，但能让你看到生成进度与细节。")

        if st.button("生成报告内容（Streaming）", type="primary", disabled=not openai_key, key="food_btn_gen_report_stream"):
            progress = st.progress(0)
            status = st.empty()
            live_box = st.empty()

            def step(pct: int, msg: str):
                progress.progress(pct)
                status.info(msg)

            report_date = dt.datetime.now().strftime("%m/%d/%Y")

            step(5, "解析门店菜单…")
            own_menu_meta = extract_menu_with_openai(own_menu_files or [], openai_key, model, label="OWN_MENU")

            step(25, "生成图表…")
            own_df = menu_to_df(own_menu_meta)
            charts = build_charts(own_df)

            step(40, "整理商圈人口统计（trade area 聚合优先）…")
            trade_area_acs = st.session_state.get("food_trade_area_acs", None)
            trade_debug = st.session_state.get("food_trade_area_debug", None)
            tract_info = st.session_state.get("food_tract_info", None)
            acs_tract = st.session_state.get("food_acs_tract", None)

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
                tract_info=tract_info,
                acs_tract=acs_tract,
                trade_area_acs=trade_area_acs,
                trade_area_debug=trade_debug,
                restaurant_google=place_details,
                charts=charts,
            )

            step(55, "构建报告提示词…")
            prompt = build_prompt(inputs, report_lang)

            step(65, "调用 AI 生成报告（实时输出中）…")
            partial = ""
            try:
                for delta in openai_text_stream(prompt, openai_key, model=model, temperature=0.25):
                    partial += delta
                    # 只展示末尾一部分，避免 UI 卡顿
                    show_tail = sanitize_text(partial[-12000:])
                    live_box.text_area("AI 实时输出（滚动更新）", value=show_tail, height=420)
                report_text = sanitize_text(partial)
            except Exception as e:
                st.error(f"Streaming 失败：{str(e)[:300]}")
                st.stop()

            step(85, "扩写补全（确保更细 & 可执行）…")
            report_text = ensure_long_enough(report_text, openai_key, model=model, lang=report_lang, min_chars=14000)

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
                        safe_st_image(v)
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
