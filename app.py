"""
Gerador de Propostas Comerciais para Licitações (v7 — com persistência, detecção de esgotado e requisitos parciais)
Streamlit + agentes modulares + busca web inteligente (ddgs) + ReportLab

Secrets:
    GEMINI_API_KEY, GEMINI_MODEL
    GROQ_API_KEY,   GROQ_MODEL
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    JINA_API_KEY (opcional)
    APP_PASSWORD_HASH, APP_PASSWORD_SALT
"""
from __future__ import annotations

import copy
import io
import ipaddress
import json
import re
import socket
import time
import uuid
import hashlib as _hashlib
import hmac as _hmac
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse
from xml.sax.saxutils import escape

import pandas as pd
import requests
import streamlit as st
from PIL import Image as PILImage
import openai
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ---------- persistência de dados do proponente (cookies) ----------
try:
    import st_cookie
    HAS_ST_COOKIE = True
except ImportError:
    HAS_ST_COOKIE = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
PROPOSTA_V = 8
MAX_ITENS = 6
MAX_PAGINAS_POR_ITEM = 8
MAX_IMAGENS_POR_ITEM = 6

NAVY = colors.HexColor("#1F3A5F")
LIGHT = colors.HexColor("#EEF2F7")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
}

JINA_BASE = "https://r.jina.ai/"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
GROQ_BASE = "https://api.groq.com/openai/v1"
OPENAI_BASE = "https://api.openai.com/v1"
DEEPSEEK_BASE = "https://api.deepseek.com"

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

RETRY_WAITS = (2, 4)

BLOCKED_DOMAINS = (
    "youtube.com", "wikipedia.org", "facebook.com", "instagram.com",
    "reddit.com", "pinterest.com", "tiktok.com", "twitter.com", "x.com",
    "linkedin.com",
)
SITE_TARGETS = ("mercadolivre.com.br", "kabum.com.br", "magazineluiza.com.br", "amazon.com.br")
KNOWN_STORES = (
    "mercadolivre", "amazon.com.br", "kabum", "magazineluiza", "magalu",
    "americanas", "casasbahia", "pontofrio", "carrefour", "fastshop",
    "terabyteshop", "pichau", "dell.com", "lenovo.com", "submarino",
    "shoptime", "leroymerlin", "kalunga", "extra.com", "havan",
    "samsung.com", "hp.com", "ferreiracosta", "telhanorte",
)

# ---------- padrões de “esgotado” ----------
OUT_OF_STOCK_PATTERNS = [
    r"esgotado", r"fora de estoque", r"indisponível", r"out of stock",
    r"sold out", r"notify me", r"avise-me", r"waitlist", r"indisponivel",
    r"temporariamente indisponível", r"sem estoque", r"produto esgotado",
]
OUT_OF_STOCK_BUTTON_RE = re.compile(
    r'<button[^>]*disabled[^>]*>.*?(comprar|adicionar|buy|add to cart|comprar agora)',
    re.I | re.S,
)

# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def _hash_password(password: str, salt: str) -> str:
    return _hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 200_000
    ).hex()


def require_login() -> None:
    if st.session_state.get("auth_ok"):
        return

    st.session_state.setdefault("auth_fails", 0)
    st.session_state.setdefault("auth_lock_until", 0.0)

    now = time.time()
    if now < st.session_state["auth_lock_until"]:
        wait = int(st.session_state["auth_lock_until"] - now) + 1
        st.error(f"🔒 Bloqueado por excesso de tentativas. Aguarde {wait}s.")
        st.stop()

    st.title("🔒 Acesso restrito")
    st.caption("Informe a senha para acessar o gerador de propostas.")

    with st.form("login_form"):
        pwd = st.text_input("Senha", type="password", key="login_pwd")
        ok = st.form_submit_button("Entrar", type="primary")

    if ok:
        stored_hash = get_secret("APP_PASSWORD_HASH")
        salt = get_secret("APP_PASSWORD_SALT")
        if not stored_hash or not salt:
            st.error("❌ Credenciais não configuradas. Verifique os Secrets.")
            st.stop()

        candidate = _hash_password(pwd, salt)
        if _hmac.compare_digest(candidate, stored_hash):
            st.session_state["auth_ok"] = True
            st.session_state["auth_fails"] = 0
            st.session_state["auth_lock_until"] = 0.0
            st.rerun()
        else:
            st.session_state["auth_fails"] += 1
            if st.session_state["auth_fails"] >= 5:
                st.session_state["auth_lock_until"] = now + 60
                st.session_state["auth_fails"] = 0
                st.error("❌ Muitas tentativas. Tente novamente em 60 segundos.")
            else:
                restantes = 5 - st.session_state["auth_fails"]
                st.error(f"❌ Senha incorreta. Tentativas restantes: {restantes}.")
    st.stop()


def now_br() -> datetime:
    try:
        return datetime.now(ZoneInfo("America/Sao_Paulo"))
    except Exception:
        return datetime.now(timezone.utc) - timedelta(hours=3)


def to_float(value, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return default
    s = re.sub(r"[^\d,.\-]", "", str(value))
    if not s:
        return default
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return default


def brl(v: float) -> str:
    return "R$ " + f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_qtd(q: float) -> str:
    return str(int(q)) if float(q).is_integer() else f"{q:.2f}".replace(".", ",")


def as_list(x) -> list[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    if isinstance(x, str) and x.strip():
        return [x.strip()]
    return []


def extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("A IA não retornou um JSON válido.")
    return json.loads(text[start:end + 1])


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def parse_models(s: str) -> list[str]:
    return [m.strip() for m in (s or "").split(",") if m.strip()]


# ----------------------------------------------------------------------------
# Detecção de “esgotado”
# ----------------------------------------------------------------------------
def is_out_of_stock(html: str, text: str = "") -> bool:
    """Retorna True se a página apresentar sinais de produto esgotado."""
    combined = (html or "") + " " + (text or "")
    combined_lower = combined.lower()

    for pat in OUT_OF_STOCK_PATTERNS:
        if re.search(pat, combined_lower, re.I):
            return True

    if OUT_OF_STOCK_BUTTON_RE.search(html or ""):
        return True

    return False


# ----------------------------------------------------------------------------
# Busca web (DuckDuckGo)
# ----------------------------------------------------------------------------
def search_web(query: str, n: int = 8) -> list[dict]:
    if DDGS is None:
        raise RuntimeError("Biblioteca de busca não instalada (adicione 'ddgs').")
    last = None
    for attempt in range(2):
        try:
            return list(DDGS().text(query, region="br-pt", safesearch="off", max_results=n))
        except Exception as e:
            if "no results" in str(e).lower():
                return []
            last = e
            if attempt < 1:
                time.sleep(1.0 + attempt * 0.5)
    raise RuntimeError(f"Falha na busca web: {last}")


def build_site_variations(item_name: str) -> list[list[str]]:
    return [
        [f'"{item_name}" site:{s}' for s in SITE_TARGETS],
        [f"{item_name} comprar", f"{item_name} preço", f"{item_name} oferta"],
    ]


# ----------------------------------------------------------------------------
# Segurança URL
# ----------------------------------------------------------------------------
def is_public_url(url: str) -> bool:
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        for info in socket.getaddrinfo(u.hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
            ip = ipaddress.ip_address(info[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# HTTP / Jina
# ----------------------------------------------------------------------------
def fetch_html(url: str, max_bytes: int = 1_500_000) -> tuple[str, str]:
    try:
        for _ in range(4):
            if not is_public_url(url):
                return "", "endereço não permitido"
            r = requests.get(url, headers=HEADERS, timeout=7, stream=True, allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, r.headers.get("location", ""))
                r.close()
                continue
            if r.status_code != 200:
                code = r.status_code
                r.close()
                return "", f"HTTP {code}"
            if "html" not in r.headers.get("content-type", "").lower():
                r.close()
                return "", "não é HTML"
            data = b""
            for chunk in r.iter_content(65536):
                data += chunk
                if len(data) >= max_bytes:
                    break
            r.close()
            return data.decode(r.encoding or "utf-8", errors="ignore"), "ok"
        return "", "redirecionamentos excessivos"
    except Exception as e:
        return "", f"erro de rede ({type(e).__name__})"


def fetch_jina(url: str, api_key: str = "", timeout: int = 20) -> tuple[str, str]:
    headers = {
        "Accept": "text/markdown",
        "Cache-Control": "no-cache",
        "x-no-cache": "true",
        "x-cache-tolerance": "0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = requests.get(f"{JINA_BASE}{url}", headers=headers, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text, "ok"
        return "", f"HTTP {r.status_code}"
    except Exception as e:
        return "", f"erro ({type(e).__name__})"


# ----------------------------------------------------------------------------
# Imagens
# ----------------------------------------------------------------------------
def normalize_image(data: bytes, max_side: int = 700, quality: int = 85) -> Optional[bytes]:
    try:
        im = PILImage.open(io.BytesIO(data))
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGBA")
            bg = PILImage.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[3])
            im = bg
        im.thumbnail((max_side, max_side))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality)
        return out.getvalue()
    except Exception:
        return None


def fetch_image(url: str, max_bytes: int = 3_000_000) -> Optional[bytes]:
    try:
        for _ in range(4):
            if not is_public_url(url):
                return None
            r = requests.get(url, headers=HEADERS, timeout=8, stream=True, allow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, r.headers.get("location", ""))
                r.close()
                continue
            if r.status_code != 200 or not r.headers.get("content-type", "").lower().startswith("image/"):
                r.close()
                return None
            data = b""
            for chunk in r.iter_content(65536):
                data += chunk
                if len(data) > max_bytes:
                    r.close()
                    return None
            r.close()
            return normalize_image(data)
    except Exception:
        return None
    return None


BAD_IMG_RE = re.compile(
    r"(sprite|logo|favicon|avatar|icon|banner|selo|payment|bandeira|"
    r"placeholder|spinner|loading|pixel|beacon|tracking)", re.I
)


def is_product_image(url: str) -> bool:
    if not url or str(url).startswith("data:"):
        return False
    if not re.match(r"^https?://", str(url)):
        return False
    path = str(url).lower().split("?")[0].split("#")[0]
    if path.endswith((".svg", ".ico", ".gif")):
        return False
    if BAD_IMG_RE.search(path):
        return False
    return True


def absolutize_images(urls, base_url: str) -> list[str]:
    out, seen = [], set()
    for u in urls:
        if not u:
            continue
        try:
            full = urljoin(base_url, str(u).strip())
        except Exception:
            continue
        if full in seen:
            continue
        seen.add(full)
        if is_product_image(full):
            out.append(full)
    return out


MD_IMG_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")
_LD_RE = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)
_OG_IMG_RE = re.compile(
    r'<meta[^>]+(?:property|name)=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']',
    re.I,
)
_PRICE_RE = re.compile(r"R\$\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:,\d{2})?)")


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def images_from_markdown(md: str, base_url: str) -> list[str]:
    return absolutize_images(MD_IMG_RE.findall(md or ""), base_url)


def extract_images_from_html(html: str, base_url: str) -> list[str]:
    urls: list[str] = []
    for m in _OG_IMG_RE.finditer(html or ""):
        urls.append(m.group(1))
    for m in _LD_RE.finditer(html or ""):
        try:
            data = json.loads(m.group(1).strip())
        except Exception:
            continue
        for node in _walk(data):
            img = node.get("image")
            if isinstance(img, str):
                urls.append(img)
            elif isinstance(img, list):
                urls.extend(x for x in img if isinstance(x, str))
            elif isinstance(img, dict):
                u = img.get("url") or img.get("contentUrl")
                if u:
                    urls.append(u)
    if HAS_BS4 and html:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for img in soup.find_all("img"):
                src = (img.get("src") or img.get("data-src")
                       or img.get("data-original") or img.get("data-lazy-src"))
                if src:
                    urls.append(src)
        except Exception:
            pass
    return absolutize_images(urls, base_url)


# ----------------------------------------------------------------------------
# Extração de produto
# ----------------------------------------------------------------------------
def _ld_price(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return to_float(v)


def parse_product_page(html: str) -> dict:
    info: dict = {}
    for m in _LD_RE.finditer(html):
        try:
            data = json.loads(m.group(1).strip())
        except Exception:
            continue
        for node in _walk(data):
            offers = node.get("offers")
            if not offers:
                continue
            prices = []
            for o in offers if isinstance(offers, list) else [offers]:
                if isinstance(o, dict) and str(o.get("priceCurrency") or "BRL").upper() == "BRL":
                    p = _ld_price(o.get("price") or o.get("lowPrice"))
                    if p > 0:
                        prices.append(p)
            if prices:
                info["preco"] = min(prices)
                info["nome"] = str(node.get("name") or "")
                rt = node.get("aggregateRating")
                if isinstance(rt, dict):
                    info["nota"] = to_float(rt.get("ratingValue"), 0.0) or None
                    info["n"] = int(to_float(rt.get("reviewCount") or rt.get("ratingCount"), 0)) or None
                return info
    return info


_BAD_CONTEXT = re.compile(
    r"frete|cupom|desconto|\boff\b|economi|cashback|parcel|juros|entrada|acima de|taxa", re.I
)


def price_from_text(text: str) -> Optional[float]:
    for m in _PRICE_RE.finditer(text or ""):
        before = text[max(0, m.start() - 30):m.start()].lower()
        after = text[m.end():m.end() + 14].lower()
        if re.search(r"\d\s*x\s*(de\s*)?$", before) or _BAD_CONTEXT.search(before[-14:] + " " + after):
            continue
        if re.match(r"\s*por\b", after):
            continue
        p = to_float(m.group(1))
        if p > 0:
            return p
    return None


def _known_store(name: str) -> bool:
    norm = re.sub(r"[^a-z0-9.]", "", name.lower())
    return any(s in norm for s in KNOWN_STORES)


def _is_blocked(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


def enrich_result(r: dict, use_jina: bool = True, jina_key: str = "") -> dict:
    url = r.get("href") or ""
    host = host_of(url)
    if not url:
        return {"_erro": "sem URL"}

    html, status = fetch_html(url)
    info: dict = {}
    if html:
        try:
            info = parse_product_page(html)
        except Exception:
            info = {}

    imgs = extract_images_from_html(html, url) if html else []

    if use_jina and (not html or (not info.get("preco") and len(imgs) < 2)):
        md, _ = fetch_jina(url, jina_key)
        if md:
            imgs.extend(images_from_markdown(md, url))
            if not info.get("preco"):
                p = price_from_text(md)
                if p:
                    info["preco"] = p
                    info["_origem"] = "jina markdown"

    imgs = list(dict.fromkeys(imgs))[:MAX_IMAGENS_POR_ITEM]

    preco = info.get("preco")
    origem = info.get("_origem") or ("página do produto (schema)" if preco else None)
    if not preco:
        p = price_from_text(f"{r.get('title', '')} {r.get('body', '')}")
        if p:
            preco = p
            origem = "trecho da busca"
    if preco and not (1 <= preco <= 5_000_000):
        preco = None
        origem = None

    motivo = "" if preco else (f"sem preço legível ({status})" if html else f"falha na leitura: {status}")

    # ---------- detecção de esgotado ----------
    esgotado = is_out_of_stock(html or "", "")

    return {
        "imagens": imgs,
        "titulo": (info.get("nome") or r.get("title") or "")[:160],
        "url": url,
        "loja": host,
        "loja_conhecida": _known_store(host),
        "preco": preco,
        "origem": origem or "",
        "nota": info.get("nota"),
        "n_avaliacoes": info.get("n"),
        "motivo_sem_preco": motivo,
        "esgotado": esgotado,
    }


def collect_quotes(queries_by_item, item_names, avisos, use_jina=True,
                   jina_key="", on_step=lambda m: None):
    cands = {i: [] for i in queries_by_item}
    sem_preco = {i: [] for i in queries_by_item}
    logs = {i: [] for i in queries_by_item}
    seen = {i: set() for i in queries_by_item}

    state = {"fails": 0, "disabled": False}
    MAX_FAILS = 3

    def run_phase(qs_by_item: dict[int, list[str]]) -> None:
        if not qs_by_item or state["disabled"]:
            return
        found: dict[int, list[dict]] = {i: [] for i in qs_by_item}
        for i, qs in qs_by_item.items():
            for q in qs:
                if state["disabled"]:
                    break
                on_step(f"🌐 Pesquisando: “{q}”")
                try:
                    results = search_web(q)
                    state["fails"] = 0
                    logs[i].append({"consulta": q, "resultados": len(results), "erro": ""})
                except Exception as e:
                    state["fails"] += 1
                    avisos.append(f"Busca falhou: {e}")
                    logs[i].append({"consulta": q, "resultados": 0, "erro": str(e)})
                    if state["fails"] >= MAX_FAILS:
                        state["disabled"] = True
                        avisos.append(
                            "⚠️ Muitas falhas consecutivas na busca web — "
                            "encerrando buscas desta execução."
                        )
                        break
                    continue
                for r in results:
                    url = r.get("href") or ""
                    host = host_of(url)
                    if url and url not in seen[i] and not _is_blocked(host):
                        seen[i].add(url)
                        found[i].append(r)
                time.sleep(0.2)
            found[i] = found[i][:MAX_PAGINAS_POR_ITEM]

        if not any(found.values()):
            return

        on_step("🛒 Lendo páginas e extraindo preços e imagens...")
        tasks = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, rs in found.items():
                for r in rs:
                    tasks.append((i, ex.submit(enrich_result, r, use_jina, jina_key)))
            for i, f in tasks:
                try:
                    c = f.result()
                except Exception:
                    continue
                if c.get("_erro"):
                    continue
                if c.get("preco"):
                    c["id"] = f"{i}-{len(cands[i])}"
                    cands[i].append(c)
                else:
                    sem_preco[i].append(c)

    run_phase(queries_by_item)

    for i, name in item_names.items():
        if cands[i] or state["disabled"]:
            continue
        for var in build_site_variations(name):
            if cands[i] or state["disabled"]:
                break
            on_step(f"🔁 Tentativa extra para item {i + 1}: variação de busca...")
            run_phase({i: var})

    return cands, sem_preco, logs


# ----------------------------------------------------------------------------
# Multi-Provider LLM
# ----------------------------------------------------------------------------
class ProviderAuthError(Exception):
    pass


def _err_msg(e: Exception) -> str:
    return str(e).replace("\n", " ")[:160]


@dataclass
class ProviderConfig:
    name: str
    api_key: str
    base_url: str
    model: str
    fallback_models: list[str] = field(default_factory=list)


class MultiProviderLLM:
    def __init__(self, providers: list[ProviderConfig]):
        self.providers = providers
        self.last_used: Optional[tuple[str, str]] = None
        self.error_log: dict[str, list[str]] = {}

    def _client(self, p: ProviderConfig) -> OpenAI:
        return OpenAI(api_key=p.api_key, base_url=p.base_url, max_retries=1, timeout=90)

    def _try_provider(self, p: ProviderConfig, system: str, user: str) -> Optional[str]:
        client = self._client(p)
        models = [p.model] + list(p.fallback_models)
        self.error_log[p.name] = []
        for model in models:
            for attempt in range(len(RETRY_WAITS) + 1):
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=0.4,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    content = resp.choices[0].message.content or ""
                    if not content:
                        self.error_log[p.name].append(f"{model}: resposta vazia")
                        break
                    self.last_used = (p.name, model)
                    return content
                except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
                    raise ProviderAuthError(
                        f"{p.name}: {type(e).__name__} ({_err_msg(e)})"
                    ) from e
                except openai.NotFoundError as e:
                    self.error_log[p.name].append(f"{model}: modelo não encontrado ({_err_msg(e)})")
                    break
                except (openai.RateLimitError, openai.InternalServerError,
                        openai.APIConnectionError, openai.APITimeoutError) as e:
                    if attempt < len(RETRY_WAITS):
                        time.sleep(RETRY_WAITS[attempt])
                    else:
                        self.error_log[p.name].append(f"{model}: {type(e).__name__} ({_err_msg(e)})")
                        break
                except openai.OpenAIError as e:
                    self.error_log[p.name].append(f"{model}: {type(e).__name__} ({_err_msg(e)})")
                    break
        return None

    def _complete(self, system: str, user: str) -> str:
        errors = []
        for p in self.providers:
            try:
                result = self._try_provider(p, system, user)
                if result:
                    return result
                detalhe = " | ".join(self.error_log.get(p.name, [])) or "sem detalhes"
                errors.append(f"{p.name}: sem resposta [{detalhe}]")
            except ProviderAuthError as e:
                errors.append(str(e))
                continue
        raise RuntimeError(
            "Nenhum provedor de IA conseguiu responder. Detalhes: " + "; ".join(errors)
        )

    def json_call(self, system: str, user: str, retries: int = 2) -> dict:
        last = None
        for _ in range(retries + 1):
            try:
                return extract_json(self._complete(system, user))
            except (ValueError, json.JSONDecodeError) as e:
                last = e
        raise RuntimeError(f"Resposta da IA em formato inválido: {last}")


# ----------------------------------------------------------------------------
# Agente
# ----------------------------------------------------------------------------
@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    llm: MultiProviderLLM

    def run(self, task: str) -> dict:
        system = (
            f"Você é {self.role}. Objetivo: {self.goal}. Contexto: {self.backstory} "
            "Responda SOMENTE com um objeto JSON válido, sem markdown e sem texto extra."
        )
        return self.llm.json_call(system, task)


SCHEMA_REQ = (
    '{"itens":[{"nome_produto":"","especificacoes_obrigatorias":[""],'
    '"quantidade":1,"unidade":"un","requisitos_minimos":[""]}]}'
)
SCHEMA_QRY = (
    '{"consultas":[{"indice":0,"queries":["",""],'
    '"faixa_preco_unitario_brl":{"min":0,"max":0}}]}'
)
SCHEMA_VAL = (
    '{"validacoes":[{"indice":0,"validos":["id","id"],"recomendado":"id ou null",'
    '"requisitos_nao_atendidos":["id:req1, req2"],'
    '"descartados":[{"id":"","motivo":""}],"descricao_comercial":"","justificativa":""}]}'
)
SCHEMA_RED = (
    '{"apresentacao":"","itens":[{"indice":0,"descricao_proposta":"","especificacoes":[""]}],'
    '"condicoes":{"validade_dias":60,"prazo_entrega":"","garantia":"","pagamento":""}}'
)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_pipeline(llm, edital, empresa, jina_key="", use_jina=True,
                 on_step=lambda msg: None, run_seed: str = "") -> dict:
    avisos: list[str] = []
    run_seed = run_seed or uuid.uuid4().hex[:8]

    on_step("🔎 Analisando requisitos do edital...")
    analista = Agent(
        "Analista de Requisitos de Licitações",
        "extrair do edital os itens e especificações com precisão",
        "Especialista em licitações públicas (Lei 14.133/2021).",
        llm,
    )
    req = analista.run(
        "Extraia os itens a serem adquiridos do texto do edital. Quantidade padrão é 1.\n"
        f"Formato JSON esperado: {SCHEMA_REQ}\n\nTEXTO:\n<<<\n{edital}\n>>>"
    )
    itens = (req.get("itens") or [])[:MAX_ITENS]
    if not itens:
        raise ValueError("Não foi possível identificar itens no texto fornecido.")
    for i, it in enumerate(itens):
        it["indice"] = i

    on_step("💰 Planejando buscas de mercado...")
    pesquisador = Agent(
        "Pesquisador de Mercado",
        "gerar consultas eficientes e validar ofertas coletadas",
        "Especialista em compras e pesquisa de preços web.",
        llm,
    )
    qry = pesquisador.run(
        f"[run-seed={run_seed}] Use este seed para VARIAR as consultas a cada execução — "
        "evite repetir exatamente as mesmas queries de execuções anteriores.\n"
        "Para cada item, crie 2 consultas de busca eficientes (4 a 8 palavras) para encontrar "
        "o produto em lojas online. Estime também uma faixa de preço unitário razoável "
        "em reais (min e max).\n"
        f"Formato JSON esperado: {SCHEMA_QRY}\n\n"
        f"ITENS:\n{json.dumps(itens, ensure_ascii=False)}"
    )
    queries = {i: [] for i in range(len(itens))}
    faixas: dict[int, tuple[float, float]] = {}
    for c in qry.get("consultas", []):
        idx = int(to_float(c.get("indice"), -1))
        if idx in queries:
            queries[idx] = as_list(c.get("queries"))
            f = c.get("faixa_preco_unitario_brl")
            if isinstance(f, dict):
                lo, hi = to_float(f.get("min")), to_float(f.get("max"))
                if 0 < lo < hi:
                    faixas[idx] = (lo, hi)
    for i, it in enumerate(itens):
        if not queries[i]:
            queries[i] = [f"{it.get('nome_produto', '')} comprar preço"]

    item_names = {i: str(it.get("nome_produto") or f"Item {i + 1}") for i, it in enumerate(itens)}
    cands, sem_preco, logs = collect_quotes(
        queries, item_names, avisos, use_jina=use_jina, jina_key=jina_key, on_step=on_step
    )
    pesquisa_em = now_br().strftime("%d/%m/%Y %H:%M:%S")

    for i, cs in cands.items():
        fx = faixas.get(i)
        for c in cs:
            c["implausivel"] = bool(fx) and not (0.3 * fx[0] <= c["preco"] <= 3 * fx[1])

    validacoes: dict[int, dict] = {}
    payload = [
        {
            "indice": i,
            "item": itens[i].get("nome_produto"),
            "ofertas": [
                {k: c[k] for k in (
                    "id", "titulo", "loja", "loja_conhecida", "preco",
                    "origem", "nota", "n_avaliacoes", "esgotado",
                )}
                for c in cs if not c["implausivel"]
            ],
        }
        for i, cs in cands.items() if any(not c["implausivel"] for c in cs)
    ]
    if payload:
        on_step("🧪 Validando ofertas e selecionando melhor custo-benefício...")
        val = pesquisador.run(
            "Selecione em 'validos' os ids das ofertas compatíveis com o produto, "
            "permitindo até 3 requisitos não atendidos. "
            "Priorize menor preço em lojas confiáveis (loja_conhecida=true). "
            "Indique a melhor em 'recomendado'. "
            "Para ofertas com requisitos não atendidos, liste-os em 'requisitos_nao_atendidos'.\n"
            f"Formato JSON esperado: {SCHEMA_VAL}\n\nDADOS:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        validacoes = {int(to_float(v.get("indice"), -1)): v for v in val.get("validacoes", [])}

    linhas = []
    for i, it in enumerate(itens):
        nome_item = item_names[i]
        v = validacoes.get(i, {})
        cs = cands.get(i, [])
        by_id = {c["id"]: c for c in cs if not c["implausivel"]}
        valid_ids = [x for x in as_list(v.get("validos")) if x in by_id]
        rec = str(v.get("recomendado") or "")

        usadas = [by_id[x] for x in valid_ids]
        provisorio = False
        img_urls: list[str] = []

        if by_id and not usadas:
            ordenados = sorted(c["preco"] for c in by_id.values())
            mediana = ordenados[len(ordenados) // 2]
            usadas = [c for c in by_id.values() if c["preco"] >= 0.3 * mediana]
            provisorio = True
            avisos.append(
                f"Item {i + 1} ({nome_item}): ofertas provisórias adotadas. "
                "Verifique os links manualmente."
            )

        # ---------- priorizar ofertas NÃO esgotadas ----------
        if usadas:
            usadas_ok = [c for c in usadas if not c.get("esgotado")]
            usadas_esgotadas = [c for c in usadas if c.get("esgotado")]
            if usadas_ok:
                usadas = usadas_ok + usadas_esgotadas  # esgotadas ficam por último
            else:
                if usadas_esgotadas:
                    avisos.append(
                        f"Item {i + 1} ({nome_item}): todas as ofertas válidas estão esgotadas. "
                        "Elas serão usadas como última alternativa."
                    )
                usadas = usadas_esgotadas

        if usadas:
            for c in usadas:
                c["usada"] = True
            melhor = by_id[rec] if rec in valid_ids else min(usadas, key=lambda c: c["preco"])
            for c in [melhor] + sorted(usadas, key=lambda c: c["preco"]):
                for u in c.get("imagens") or []:
                    if u not in img_urls:
                        img_urls.append(u)
                    if len(img_urls) >= MAX_IMAGENS_POR_ITEM:
                        break
                if len(img_urls) >= MAX_IMAGENS_POR_ITEM:
                    break
            precos = [c["preco"] for c in usadas]
            preco, faixa, ref = melhor["preco"], (min(precos), max(precos)), melhor["titulo"]
            confianca = "baixa (provisória)" if provisorio else ("alta" if len(usadas) >= 2 else "média")
        else:
            preco, faixa, confianca, ref = 0.0, (0.0, 0.0), "sem cotação", ""
            avisos.append(f"Item {i + 1} ({nome_item}): nenhuma oferta encontrada automaticamente.")

        qtd = to_float(it.get("quantidade"), 1.0) or 1.0
        linhas.append({
            "numero": i + 1,
            "nome": nome_item,
            "unidade": str(it.get("unidade") or "un"),
            "quantidade": qtd,
            "preco_unitario": round(preco, 2),
            "subtotal": round(qtd * preco, 2),
            "faixa": faixa,
            "especificacoes": as_list(it.get("especificacoes_obrigatorias")),
            "requisitos": as_list(it.get("requisitos_minimos")),
            "descricao": str(v.get("descricao_comercial") or ""),
            "produto_referencia": ref,
            "cotacoes": cs,
            "sem_preco": sem_preco.get(i, []),
            "faixa_plausivel": faixas.get(i),
            "imagem_candidatas": img_urls,
            "imagem_bytes": [],
            "imagem_thumb": [],
            "imagem_auto": None,
            "imagens_list": [],
            "imagem_legenda": "",
            "diagnostico": logs.get(i, []),
            "confianca": confianca,
            "justificativa": str(v.get("justificativa") or ""),
            "requisitos_nao_atendidos": v.get("requisitos_nao_atendidos", []),
        })

    tasks = [(l, u) for l in linhas for u in l["imagem_candidatas"]]
    if tasks:
        on_step("🖼️ Baixando galeria de imagens dos produtos...")
        results: dict[tuple, bytes] = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(fetch_image, u): (id(l), u) for l, u in tasks}
            for f in as_completed(futs):
                key = futs[f]
                try:
                    b = f.result()
                    if b:
                        results[key] = b
                except Exception:
                    pass
        for l in linhas:
            l["imagem_bytes"] = [
                results[(id(l), u)] for u in l["imagem_candidatas"]
                if (id(l), u) in results
            ]
            l["imagem_thumb"] = [
                normalize_image(b, max_side=200, quality=70)
                for b in l["imagem_bytes"]
            ]
            if l["imagem_bytes"]:
                l["imagem_auto"] = l["imagem_bytes"][0]

    on_step("✍️ Redigindo proposta comercial formal...")
    redator = Agent(
        "Redator de Propostas",
        "organizar termos técnicos e comerciais da proposta",
        "Especialista em redação técnica e comercial para editais.",
        llm,
    )
    resumo = [
        {"nome": l["nome"], "unidade": l["unidade"], "quantidade": l["quantidade"],
         "especificacoes": l["especificacoes"], "indice": l["numero"] - 1}
        for l in linhas
    ]
    red = redator.run(
        f"Proponente: {empresa.get('razao_social')}. Edital: {empresa.get('edital')}.\n"
        f"Formato JSON esperado: {SCHEMA_RED}\n\nITENS:\n{json.dumps(resumo, ensure_ascii=False)}"
    )
    red_map = {int(to_float(r.get("indice"), -1)): r for r in red.get("itens", [])}
    for i, l in enumerate(linhas):
        r = red_map.get(i, {})
        l["descricao_proposta"] = str(r.get("descricao_proposta") or l["descricao"] or l["nome"])
        l["especificacoes_finais"] = as_list(r.get("especificacoes")) or (l["especificacoes"] + l["requisitos"])

    cond = red.get("condicoes") or {}
    proposta = {
        "_v": PROPOSTA_V,
        "_gen_id": uuid.uuid4().hex[:8],
        "_run_seed": run_seed,
        "empresa": empresa,
        "data": now_br().strftime("%d/%m/%Y"),
        "pesquisa_em": pesquisa_em,
        "avisos": list(dict.fromkeys(avisos)),
        "apresentacao": str(red.get("apresentacao") or "Apresentamos nossa proposta comercial conforme especificações do edital."),
        "linhas": linhas,
        "condicoes": {
            "validade_dias": int(to_float(cond.get("validade_dias"), 60) or 60),
            "prazo_entrega": str(cond.get("prazo_entrega") or "Conforme edital."),
            "garantia": str(cond.get("garantia") or "Conforme edital."),
            "pagamento": str(cond.get("pagamento") or "Conforme edital."),
        },
    }
    recalc(proposta)
    return proposta


def recalc(p: dict) -> None:
    for l in p["linhas"]:
        l["subtotal"] = round(l["quantidade"] * l["preco_unitario"], 2)
    p["total"] = round(sum(l["subtotal"] for l in p["linhas"]), 2)


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
def _image_grid(imgs_bytes, max_size_cm=4.0, ncols=3):
    cells = []
    for b in imgs_bytes:
        try:
            w, h = PILImage.open(io.BytesIO(b)).size
            s = min(max_size_cm * cm / w, max_size_cm * cm / h)
            cells.append(RLImage(io.BytesIO(b), width=w * s, height=h * s))
        except Exception:
            cells.append("")
    rows, row = [], []
    for c in cells:
        row.append(c)
        if len(row) == ncols:
            rows.append(row); row = []
    if row:
        while len(row) < ncols:
            row.append("")
        rows.append(row)
    t = Table(rows, colWidths=[(17.0 / ncols) * cm] * ncols)
    t.setStyle(TableStyle([
        ("ALIGN",  (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def _pdf_signature(p: dict, anexo: bool) -> str:
    h = _hashlib.sha256()
    h.update(f"{PROPOSTA_V}|{anexo}".encode())
    h.update(p.get("apresentacao", "").encode("utf-8", "ignore"))
    for l in p["linhas"]:
        for k in ("numero", "nome", "unidade", "quantidade", "preco_unitario",
                  "subtotal", "descricao_proposta"):
            h.update(str(l.get(k, "")).encode("utf-8", "ignore"))
        for s in l.get("especificacoes_finais") or []:
            h.update(str(s).encode("utf-8", "ignore"))
        for img in l.get("imagens_list") or []:
            h.update(_hashlib.md5(img).digest())
        if l.get("imagem"):
            h.update(_hashlib.md5(l["imagem"]).digest())
    return h.hexdigest()


def build_pdf(p: dict, anexo: bool = False) -> bytes:
    emp = p["empresa"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Proposta Comercial", author=emp.get("razao_social") or "",
    )
    ss = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=ss["Normal"], fontSize=9.5, leading=13)
    title = ParagraphStyle("title", parent=base, fontSize=20, leading=24, alignment=TA_CENTER,
                           textColor=NAVY, fontName="Helvetica-Bold")
    sub = ParagraphStyle("sub", parent=base, alignment=TA_CENTER, textColor=colors.grey)
    h2 = ParagraphStyle("h2", parent=base, fontSize=12, leading=15, textColor=NAVY,
                        fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5)
    just = ParagraphStyle("just", parent=base, alignment=TA_JUSTIFY)
    th = ParagraphStyle("th", parent=base, textColor=colors.white, fontName="Helvetica-Bold",
                        alignment=TA_CENTER, fontSize=9)
    right = ParagraphStyle("right", parent=base, alignment=TA_RIGHT)
    bold_r = ParagraphStyle("bold_r", parent=right, fontName="Helvetica-Bold")
    center = ParagraphStyle("center", parent=base, alignment=TA_CENTER)
    bullet = ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=3)
    small = ParagraphStyle("small", parent=base, fontSize=8, leading=10)
    cap = ParagraphStyle("cap", parent=base, fontSize=7.5, leading=9,
                         alignment=TA_CENTER, textColor=colors.grey)

    def P(text, style=base):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)

    story = [P("PROPOSTA COMERCIAL", title), Spacer(1, 4)]
    ref = emp.get("edital") or ""
    story.append(P(f"{ref + ' — ' if ref else ''}Emitida em {p['data']}", sub))
    story.append(Spacer(1, 12))

    story.append(P("1. Dados do proponente", h2))
    dados = [
        ("Razão social", emp.get("razao_social")), ("CNPJ", emp.get("cnpj")),
        ("Endereço", emp.get("endereco")), ("Contato", emp.get("contato")),
    ]
    t = Table(
        [[P(k, ParagraphStyle("k", parent=base, fontName="Helvetica-Bold")), P(v or "—")] for k, v in dados],
        colWidths=[3.5 * cm, 13.5 * cm],
    )
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [t, Spacer(1, 6), P("2. Apresentação", h2), P(p["apresentacao"], just)]

    story.append(P("3. Itens e preços", h2))
    rows = [[P(h, th) for h in ("Item", "Descrição", "Un.", "Qtd.", "Preço unit.", "Total")]]
    for l in p["linhas"]:
        rows.append([
            P(l["numero"], center),
            P(f"{l['nome']}\n{l['descricao_proposta']}"),
            P(l["unidade"], center),
            P(fmt_qtd(l["quantidade"]), center),
            P(brl(l["preco_unitario"]), right),
            P(brl(l["subtotal"]), right),
        ])
    rows.append([P("VALOR TOTAL DA PROPOSTA", bold_r), "", "", "", "", P(brl(p["total"]), bold_r)])
    it = Table(rows, colWidths=[1.3 * cm, 6.7 * cm, 1.3 * cm, 1.7 * cm, 3.0 * cm, 3.0 * cm], repeatRows=1)
    it.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (0, -1), (4, -1)),
        ("BACKGROUND", (0, -1), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(it)

    story.append(P("4. Especificações técnicas", h2))
    for l in p["linhas"]:
        story.append(P(
            f"Item {l['numero']} — {l['nome']}",
            ParagraphStyle("b", parent=base, fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2),
        ))
        bullets = [Paragraph(escape(s), bullet, bulletText="•")
                   for s in (l["especificacoes_finais"] or ["Conforme edital."])]

        imgs = (l.get("imagens_list") or [])[:MAX_IMAGENS_POR_ITEM]
        if not imgs and l.get("imagem"):
            imgs = [l["imagem"]]

        if not imgs:
            story += bullets
        elif len(imgs) == 1:
            try:
                w, h = PILImage.open(io.BytesIO(imgs[0])).size
                s = min(4.2 * cm / w, 4.2 * cm / h)
                cell = [RLImage(io.BytesIO(imgs[0]), width=w * s, height=h * s)]
                if l.get("imagem_legenda"):
                    cell.append(P(l["imagem_legenda"], cap))
                tb = Table([[bullets, cell]], colWidths=[12.6 * cm, 4.4 * cm])
                tb.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]))
                story.append(tb)
            except Exception:
                story += bullets
        else:
            story += bullets
            story.append(Spacer(1, 4))
            story.append(_image_grid(imgs, max_size_cm=4.0, ncols=min(3, len(imgs))))

    c = p["condicoes"]
    story.append(P("5. Condições comerciais", h2))
    for k, v in (
        ("Validade da proposta", f"{c['validade_dias']} dias"),
        ("Prazo de entrega", c["prazo_entrega"]),
        ("Garantia", c["garantia"]),
        ("Pagamento", c["pagamento"]),
    ):
        story.append(Paragraph(f"<b>{k}:</b> {escape(str(v))}", base))
        story.append(Spacer(1, 2))

    linha = Table([[""]], colWidths=[8 * cm], rowHeights=[0.1 * cm])
    linha.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black)]))
    story.append(KeepTogether([
        Spacer(1, 18),
        P(f"{emp.get('cidade') or '____________'}, {p['data']}.", center),
        Spacer(1, 40), linha, Spacer(1, 3),
        P(emp.get("representante") or "Representante legal", center),
        P(emp.get("cargo") or "", center),
        P(emp.get("razao_social") or "", center),
    ]))

    if anexo:
        story += [PageBreak(), P("ANEXO — Pesquisa de preços", h2),
                  P(f"Consulta realizada em {p['pesquisa_em']}.", small), Spacer(1, 6)]
        for l in p["linhas"]:
            validas = [q for q in l["cotacoes"] if q.get("usada")]
            story.append(P(
                f"Item {l['numero']} — {l['nome']}",
                ParagraphStyle("b2", parent=base, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3),
            ))
            if not validas:
                story.append(P("Sem ofertas utilizadas.", small))
                continue
            rows = [[P(h, th) for h in ("Loja", "Preço", "Link")]]
            for q in validas:
                href = escape(q["url"], {'"': "&quot;"})
                short = escape(q["url"][:70] + ("…" if len(q["url"]) > 70 else ""))
                rows.append([
                    P(q["loja"], small),
                    P(brl(q["preco"]), small),
                    Paragraph(f'<link href="{href}" color="blue">{short}</link>', small),
                ])
            qt = Table(rows, colWidths=[4 * cm, 3 * cm, 10 * cm], repeatRows=1)
            qt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(qt)

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.1 * cm, f"{emp.get('razao_social') or ''} — Proposta Comercial")
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Página {d.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Limpeza de estado
# ----------------------------------------------------------------------------
def _reset_proposal_state() -> None:
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in ("up_", "sel_", "imgpick_", "editor_", "_pdf_")):
            del st.session_state[k]
    st.session_state.pop("proposta", None)
    st.session_state.pop("llm_used", None)


# ----------------------------------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Gerador de Propostas", page_icon="📄", layout="wide")

require_login()

st.session_state.setdefault("gen_id", 0)

# ---------- sidebar com persistência de cookies ----------
with st.sidebar:
    st.header("🏢 Dados do proponente")

    # chaves usadas para sincronizar com cookies
    cookie_keys = [
        "emp_razao_social", "emp_cnpj", "emp_endereco", "emp_contato",
        "emp_representante", "emp_cargo", "emp_cidade", "emp_edital",
    ]

    if HAS_ST_COOKIE:
        with st_cookie.sync(*cookie_keys):
            razao_social = st.text_input("Razão social", key="emp_razao_social")
            cnpj = st.text_input("CNPJ", key="emp_cnpj")
            endereco = st.text_input("Endereço", key="emp_endereco")
            contato = st.text_input("E-mail / telefone", key="emp_contato")
            representante = st.text_input("Representante legal", key="emp_representante")
            cargo = st.text_input("Cargo", key="emp_cargo")
            cidade = st.text_input("Cidade", key="emp_cidade")
            edital = st.text_input("Edital / Pregão nº", key="emp_edital")
    else:
        st.caption("⚠️ `st-cookie` não instalado — os dados não serão lembrados entre sessões.")
        razao_social = st.text_input("Razão social", key="emp_razao_social")
        cnpj = st.text_input("CNPJ", key="emp_cnpj")
        endereco = st.text_input("Endereço", key="emp_endereco")
        contato = st.text_input("E-mail / telefone", key="emp_contato")
        representante = st.text_input("Representante legal", key="emp_representante")
        cargo = st.text_input("Cargo", key="emp_cargo")
        cidade = st.text_input("Cidade", key="emp_cidade")
        edital = st.text_input("Edital / Pregão nº", key="emp_edital")

    empresa = {
        "razao_social": razao_social,
        "cnpj": cnpj,
        "endereco": endereco,
        "contato": contato,
        "representante": representante,
        "cargo": cargo,
        "cidade": cidade,
        "edital": edital,
    }

    with st.expander("🤖 Provedores de IA (fallback automático)", expanded=True):
        st.caption(
            "A chamada tenta primeiro o Provedor 1; se falhar, "
            "cai para o 2 e depois para o 3 — sem interromper a geração."
        )

        st.markdown("### 1️⃣ Google Gemini (Principal)")
        g_key = get_secret("GEMINI_API_KEY") or get_secret("LLM_API_KEY")
        if not g_key:
            g_key = st.text_input("Gemini API Key", type="password", key="in_g_key")
        else:
            st.success("Chave Gemini carregada dos Secrets.")
        g_models_str = st.text_input(
            "Modelos Gemini (vírgula = alternativas)",
            value=get_secret("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            key="in_g_models",
        )

        st.markdown("### 2️⃣ Groq (Backup 1)")
        gr_key = get_secret("GROQ_API_KEY")
        if not gr_key:
            gr_key = st.text_input("Groq API Key", type="password", key="in_gr_key")
        else:
            st.success("Chave Groq carregada.")
        gr_models_str = st.text_input(
            "Modelos Groq",
            value=get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL),
            key="in_gr_models",
        )

        st.markdown("### 3️⃣ OpenAI / DeepSeek (Backup 2)")
        oa_key = get_secret("OPENAI_API_KEY")
        if not oa_key:
            oa_key = st.text_input("OpenAI/DeepSeek API Key", type="password", key="in_oa_key")
        else:
            st.success("Chave OpenAI/DeepSeek carregada.")
        oa_base = st.text_input(
            "Base URL",
            value=get_secret("OPENAI_BASE_URL", OPENAI_BASE),
            key="in_oa_base",
        )
        oa_models_str = st.text_input(
            "Modelos OpenAI/DeepSeek",
            value=get_secret("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            key="in_oa_models",
        )

        st.divider()
        st.markdown("### 🌐 Scraping")
        jina_key = get_secret("JINA_API_KEY")
        if not jina_key:
            jina_key = st.text_input("Jina Reader API Key (opcional)", type="password", key="in_jina_key")
        else:
            st.success("Chave Jina carregada.")
        use_jina = st.checkbox("Usar Jina Reader (renderiza JS)", value=True)

    st.divider()
    if st.button("🔄 Limpar sessão / recomeçar",
                 help="Apaga resultado atual e widgets antigos (não desloga)"):
        _reset_proposal_state()
        st.rerun()

providers: list[ProviderConfig] = []
g_models = parse_models(g_models_str)
if g_key and g_models:
    providers.append(ProviderConfig("Gemini", g_key, GEMINI_BASE, g_models[0], g_models[1:]))
gr_models = parse_models(gr_models_str)
if gr_key and gr_models:
    providers.append(ProviderConfig("Groq", gr_key, GROQ_BASE, gr_models[0], gr_models[1:]))
oa_models = parse_models(oa_models_str)
if oa_key and oa_models:
    providers.append(ProviderConfig("OpenAI/DeepSeek", oa_key, oa_base, oa_models[0], oa_models[1:]))

st.title("📄 Gerador de Propostas para Licitações")
st.caption("Cole a especificação do edital: os agentes pesquisam preços reais na web e geram a proposta em PDF.")

edital_text = st.text_area(
    "Especificação do produto (texto bruto do edital)", height=260,
    placeholder=(
        'Ex.: Item 1 — Notebook, processador mínimo 8 núcleos, 16 GB RAM, '
        'SSD 512 GB, tela 15,6". Quantidade: 20 unidades.'
    ),
)

if st.button("Gerar Proposta Comercial", type="primary"):
    if not edital_text.strip():
        st.warning("Cole a especificação do edital antes de gerar.")
    elif not providers:
        st.warning("Informe ao menos uma chave de API na barra lateral (ou nos Secrets).")
    else:
        _reset_proposal_state()
        llm = MultiProviderLLM(providers)
        run_seed = uuid.uuid4().hex[:8]
        try:
            with st.spinner("Os agentes estão trabalhando (a pesquisa web pode levar alguns instantes)..."):
                status = st.empty()
                proposta_new = run_pipeline(
                    llm, edital_text.strip(), empresa,
                    jina_key=jina_key or "",
                    use_jina=use_jina,
                    on_step=status.info,
                    run_seed=run_seed,
                )
                st.session_state["proposta"] = proposta_new
                st.session_state["gen_id"] += 1
                st.session_state["llm_used"] = llm.last_used
                status.empty()
            if llm.last_used and providers and llm.last_used[0] != providers[0].name:
                st.warning(
                    f"⚠️ Fallback ativo: a resposta foi gerada por "
                    f"**{llm.last_used[0]}** ({llm.last_used[1]})."
                )
        except Exception as e:
            st.session_state.pop("proposta", None)
            st.error(f"Erro ao gerar a proposta: {e}")

proposta = st.session_state.get("proposta")
if proposta and proposta.get("_v") != PROPOSTA_V:
    _reset_proposal_state()
    proposta = None
    st.info("O app foi atualizado. Gere a proposta novamente.")

if proposta:
    st.divider()
    if st.session_state.get("llm_used"):
        name, model = st.session_state["llm_used"]
        st.caption(f"🤖 Provedor usado: **{name}** · `{model}`")
    st.caption(
        f"🆔 Proposta `{proposta.get('_gen_id', '?')}` · "
        f"pesquisa de **{proposta['pesquisa_em']}** · "
        f"seed `{proposta.get('_run_seed', '?')}`"
    )

    st.subheader("Revise os valores e itens")
    for aviso in proposta["avisos"]:
        st.warning(aviso)

    with st.expander("🔍 Consultas efetivamente executadas (verificação)"):
        for l in proposta["linhas"]:
            st.markdown(f"**Item {l['numero']} — {l['nome']}**")
            for d in l.get("diagnostico", []):
                q, n, err = d.get("consulta", ""), d.get("resultados", 0), d.get("erro", "")
                icon = "❌" if err else "✓"
                st.caption(f"{icon} `{q}` — {n} resultado(s){' · ' + err if err else ''}")

    df = pd.DataFrame([
        {"Item": l["numero"], "Produto": l["nome"], "Un.": l["unidade"], "Qtd": l["quantidade"],
         "Preço unitário": l["preco_unitario"],
         "Faixa encontrada": f"{brl(l['faixa'][0])} – {brl(l['faixa'][1])}" if l["faixa"][1] else "—",
         "Confiança": l["confianca"]}
        for l in proposta["linhas"]
    ])
    edited = st.data_editor(
        df, hide_index=True,
        disabled=["Item", "Produto", "Un.", "Faixa encontrada", "Confiança"],
        key=f"editor_{st.session_state['gen_id']}",
        column_config={
            "Qtd": st.column_config.NumberColumn(min_value=0.0),
            "Preço unitário": st.column_config.NumberColumn(format="R$ %.2f", min_value=0.0),
        },
    )

    final = copy.deepcopy(proposta)
    for idx, row in edited.iterrows():
        if idx < len(final["linhas"]):
            final["linhas"][idx]["quantidade"] = to_float(row["Qtd"], final["linhas"][idx]["quantidade"])
            final["linhas"][idx]["preco_unitario"] = to_float(row["Preço unitário"], final["linhas"][idx]["preco_unitario"])
    recalc(final)

    st.metric("Valor total da proposta", brl(final["total"]))
    incluir_imagens = st.checkbox("Incluir imagens dos produtos no PDF", value=True)

    for l in final["linhas"]:
        with st.expander(
            f"Item {l['numero']} — {l['nome']}  ·  ofertas: {len(l['cotacoes'])}  ·  confiança: {l['confianca']}",
            expanded=not l["cotacoes"],
        ):
            gid = f"{st.session_state['gen_id']}_{l['numero']}"

            up = st.file_uploader(
                "Enviar imagem própria (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"up_{gid}",
            )
            chosen_full: list[bytes] = []

            if up is not None:
                norm = normalize_image(up.getvalue())
                if norm:
                    chosen_full = [norm]
                    st.image(norm, width=180, caption="Upload próprio")
            else:
                thumbs = l.get("imagem_thumb") or []
                fulls = l.get("imagem_bytes") or []
                if thumbs:
                    st.markdown(
                        f"**{len(thumbs)} imagem(ns) encontrada(s) — marque as que irão para o PDF:**"
                    )
                    ncols = min(3, len(thumbs))
                    cols = st.columns(ncols)
                    for idx, thumb in enumerate(thumbs):
                        with cols[idx % ncols]:
                            st.image(thumb, width=150, caption=f"Foto {idx + 1}")
                            incluir = st.checkbox(
                                "Incluir no PDF",
                                value=(idx == 0),
                                key=f"imgpick_{gid}_{idx}",
                            )
                            if incluir and idx < len(fulls):
                                chosen_full.append(fulls[idx])
                    if chosen_full:
                        st.caption(f"✅ {len(chosen_full)} imagem(ns) selecionada(s) para este item.")
                else:
                    st.caption("Nenhuma imagem encontrada automaticamente. Faça upload se desejar.")

            l["imagens_list"] = chosen_full if incluir_imagens else []
            l["imagem_legenda"] = "Imagem ilustrativa" if l["imagens_list"] else ""
            l["imagem"] = None

            if l["cotacoes"]:
                st.markdown("**Ofertas encontradas**")
                # Adiciona coluna "Esgotado" para transparência
                st.dataframe(
                    pd.DataFrame([
                        {
                            "Usada": q.get("usada", False),
                            "Loja": q["loja"],
                            "Produto": q["titulo"],
                            "Preço": q["preco"],
                            "Esgotado": "Sim" if q.get("esgotado") else "Não",
                            "Link": q["url"],
                        }
                        for q in l["cotacoes"]
                    ]),
                    hide_index=True,
                    column_config={
                        "Preço": st.column_config.NumberColumn(format="R$ %.2f"),
                        "Link": st.column_config.LinkColumn(display_text="abrir"),
                    },
                )
            if l.get("sem_preco"):
                with st.popover("Sugestões sem preço extraído"):
                    for q in l["sem_preco"][:8]:
                        st.markdown(f"- [{q['titulo']}]({q['url']}) — *{q['motivo_sem_preco']}*")

            # Mostra requisitos não atendidos, se houver
            if l.get("requisitos_nao_atendidos"):
                st.caption(
                    "⚠️ Requisitos não atendidos: " + ", ".join(l["requisitos_nao_atendidos"])
                )

    anexo = st.checkbox("Incluir anexo com a pesquisa de preços e links no PDF")

    sig = _pdf_signature(final, anexo)
    if st.session_state.get("_pdf_sig") != sig:
        try:
            st.session_state["_pdf_bytes"] = build_pdf(final, anexo)
            st.session_state["_pdf_sig"] = sig
            st.session_state["_pdf_name"] = (
                f"proposta_{now_br().strftime('%Y%m%d_%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}.pdf"
            )
        except Exception as e:
            st.session_state["_pdf_bytes"] = None
            st.error(f"Erro ao gerar o PDF: {e}")

    if st.session_state.get("_pdf_bytes"):
        st.download_button(
            "⬇️ Baixar proposta em PDF",
            data=st.session_state["_pdf_bytes"],
            mime="application/pdf",
            file_name=st.session_state["_pdf_name"],
            type="primary",
        )