"""
Gerador de Propostas Comerciais para Licitações (v3 — Otimizado e sem Serper)
Streamlit + agentes modulares (CrewAI-style) + busca web inteligente (ddgs) + ReportLab

Secrets (Streamlit Cloud > Settings > Secrets):
    LLM_API_KEY        = "sua-chave-gemini"
    LLM_BASE_URL       = "https://generativelanguage.googleapis.com/v1beta/openai/"   # opcional
    LLM_MODEL          = "gemini-3.8-flash"                                           # opcional
    LLM_FALLBACK_MODEL = "gemini-3.5-flash-lite"                                 # opcional
"""
from __future__ import annotations

import copy
import io
import ipaddress
import json
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from reportlab.platypus import Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.5-flash-lite"
PROPOSTA_V = 5                
MAX_ITENS = 6                 
MAX_PAGINAS_POR_ITEM = 8      
NAVY = colors.HexColor("#1F3A5F")
LIGHT = colors.HexColor("#EEF2F7")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9",
}

BLOCKED_DOMAINS = ("youtube.com", "wikipedia.org", "facebook.com", "instagram.com", "reddit.com", "pinterest.com",
                   "tiktok.com", "twitter.com", "x.com", "linkedin.com", "mercadolivre.com.br/anuncios", "mercadolivre.com/p/ML")
SITE_TARGETS = ("mercadolivre.com.br", "kabum.com.br", "magazineluiza.com.br", "amazon.com.br")
KNOWN_STORES = (
    "mercadolivre", "amazon.com.br", "kabum", "magazineluiza", "magalu", "americanas", "casasbahia", "pontofrio",
    "carrefour", "fastshop", "terabyteshop", "pichau", "dell.com", "lenovo.com", "submarino", "shoptime",
    "leroymerlin", "kalunga", "extra.com", "havan", "samsung.com", "hp.com", "ferreiracosta", "telhanorte",
)


# ----------------------------------------------------------------------------
# Utilitários
# ----------------------------------------------------------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


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
    return json.loads(text[start : end + 1])


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


# ----------------------------------------------------------------------------
# Ferramentas de pesquisa web e extração
# ----------------------------------------------------------------------------
def search_web(query: str, n: int = 8) -> list[dict]:
    if DDGS is None:
        raise RuntimeError("Biblioteca de busca não instalada (adicione 'ddgs' ao requirements.txt).")
    last = None
    for attempt in range(3):
        try:
            return list(DDGS().text(query, region="br-pt", safesearch="off", max_results=n))
        except Exception as e:
            if "no results" in str(e).lower():
                return []
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Falha na busca web: {last}")


def _known_store(name: str) -> bool:
    norm = re.sub(r"[^a-z0-9.]", "", name.lower())
    return any(s in norm for s in KNOWN_STORES)


def is_public_url(url: str) -> bool:
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        for info in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                return False
        return True
    except Exception:
        return False


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


def normalize_image(data: bytes) -> bytes | None:
    try:
        im = PILImage.open(io.BytesIO(data))
        im.load()
        if im.mode != "RGB":
            im = im.convert("RGBA")
            bg = PILImage.new("RGB", im.size, "white")
            bg.paste(im, mask=im.split()[3])
            im = bg
        im.thumbnail((700, 700))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=85)
        return out.getvalue()
    except Exception:
        return None


def fetch_image(url: str, max_bytes: int = 3_000_000) -> bytes | None:
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


def first_image(urls: list[str]) -> tuple[bytes | None, str]:
    for u in urls:
        img = fetch_image(u)
        if img:
            return img, u
    return None, ""


_LD_RE = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.S | re.I)
_OG_IMG_RE = re.compile(r'<meta[^>]+(?:property|name)=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)["\']', re.I)
_PRICE_RE = re.compile(r"R\$\s?(\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\d+(?:,\d{2})?)")


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


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


_BAD_CONTEXT = re.compile(r"frete|cupom|desconto|\boff\b|economi|cashback|parcel|juros|entrada|acima de|taxa", re.I)


def price_from_text(text: str) -> float | None:
    for m in _PRICE_RE.finditer(text):
        before = text[max(0, m.start() - 30) : m.start()].lower()
        after = text[m.end() : m.end() + 14].lower()
        if re.search(r"\d\s*x\s*(de\s*)?$", before) or _BAD_CONTEXT.search(before[-14:] + " " + after):
            continue
        if re.match(r"\s*por\b", after):
            continue
        p = to_float(m.group(1))
        if p > 0:
            return p
    return None


def enrich_result(r: dict) -> dict:
    url, host = r.get("href", ""), host_of(r.get("href", ""))
    html, status = fetch_html(url)
    try:
        info = parse_product_page(html) if html else {}
    except Exception:
        info = {}
    preco = info.get("preco")
    origem = "página do produto (schema)" if preco else None
    if not preco:
        preco = price_from_text(f"{r.get('title', '')} {r.get('body', '')}")
        origem = "trecho da busca" if preco else None
    if preco and not (1 <= preco <= 5_000_000):
        preco = None
    motivo = "" if preco else (f"sem preço legível ({status})" if html else f"falha na leitura: {status}")
    og = _OG_IMG_RE.search(html) if html else None
    return {
        "imagem_url": urljoin(url, og.group(1)) if og else "",
        "titulo": (info.get("nome") or r.get("title") or "")[:160],
        "url": url,
        "loja": host,
        "loja_conhecida": _known_store(host),
        "preco": preco,
        "origem": origem,
        "nota": info.get("nota"),
        "n_avaliacoes": info.get("n"),
        "motivo_sem_preco": motivo,
    }


def _is_blocked(host: str) -> bool:
    return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)


def collect_quotes(queries_by_item: dict[int, list[str]], avisos: list[str], on_step=lambda m: None):
    cands = {i: [] for i in queries_by_item}
    sem_preco = {i: [] for i in queries_by_item}
    logs = {i: [] for i in queries_by_item}
    seen = {i: set() for i in queries_by_item}

    def run_phase(qs_by_item: dict[int, list[str]]) -> None:
        if not qs_by_item:
            return
        found: dict[int, list[dict]] = {}
        for i, qs in qs_by_item.items():
            found[i] = []
            for q in qs:
                on_step(f"🌐 Pesquisando na web: “{q}”")
                try:
                    results = search_web(q)
                    logs[i].append({"consulta": q, "resultados": len(results), "erro": ""})
                except Exception as e:
                    avisos.append(str(e))
                    logs[i].append({"consulta": q, "resultados": 0, "erro": str(e)})
                    continue
                for r in results:
                    url, host = r.get("href") or "", host_of(r.get("href") or "")
                    if url and url not in seen[i] and not _is_blocked(host):
                        seen[i].add(url)
                        found[i].append(r)
                time.sleep(0.6)
            found[i] = found[i][:MAX_PAGINAS_POR_ITEM]
        
        on_step("🛒 Lendo páginas das lojas e extraindo preços...")
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [(i, ex.submit(enrich_result, r)) for i, rs in found.items() for r in rs]
            for i, f in futures:
                try:
                    c = f.result()
                except Exception:
                    continue
                if c["preco"]:
                    c["id"] = f"{i}-{len(cands[i])}"
                    cands[i].append(c)
                else:
                    sem_preco[i].append(c)

    run_phase(queries_by_item)
    
    # Tentativa de recuperação em grandes sites se faltarem ofertas
    retry = {i: [f"{queries_by_item[i][0]} site:{d}" for d in SITE_TARGETS] for i in queries_by_item if not cands[i] and queries_by_item[i]}
    if retry:
        on_step("🔁 Buscando diretamente em grandes varejistas parceiros...")
        run_phase(retry)
        
    return cands, sem_preco, logs


NON_TEXT_MODEL_HINTS = ("tts", "live", "image", "embedding", "transcribe", "robotics", "computer-use", "audio",
                        "omni", "translate", "veo", "imagen", "lyria", "banana", "antigravity", "deep-research")


@st.cache_data(ttl=3600, show_spinner=False)
def list_gemini_models(api_key: str, base_url: str) -> list[str]:
    try:
        ids = [m.id.removeprefix("models/") for m in OpenAI(api_key=api_key, base_url=base_url, max_retries=1, timeout=20).models.list()]
    except Exception:
        return []
    return sorted({i for i in ids if i.startswith("gemini") and not any(h in i for h in NON_TEXT_MODEL_HINTS)})


def _flash_version(name: str):
    m = re.fullmatch(r"gemini-(\d+(?:\.\d+)*)-flash(-lite)?", name)
    return (tuple(int(x) for x in m.group(1).split(".")), bool(m.group(2))) if m else None


def pick_defaults(models: list[str], want: str, want_fallback: str) -> tuple[str, str]:
    flash = sorted((m for m in models if (v := _flash_version(m)) and not v[1]), key=lambda m: _flash_version(m)[0], reverse=True)
    lite = sorted((m for m in models if (v := _flash_version(m)) and v[1]), key=lambda m: _flash_version(m)[0], reverse=True)
    main = want if want in models else (flash[0] if flash else models[0])
    fb = want_fallback if want_fallback in models else (lite[0] if lite else next((m for m in flash if m != main), main))
    return main, fb


# ----------------------------------------------------------------------------
# Camada de LLM e Agentes
# ----------------------------------------------------------------------------
RETRY_WAITS = (3, 8, 16)


class LLM:
    def __init__(self, api_key: str, base_url: str, model: str, fallback_model: str = ""):
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=1, timeout=90)
        self.models = [model] + ([fallback_model] if fallback_model and fallback_model != model else [])

    def _create(self, model: str, system: str, user: str):
        return self.client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )

    def _complete(self, system: str, user: str) -> str:
        last: Exception | None = None
        for model in self.models:
            for attempt in range(len(RETRY_WAITS) + 1):
                try:
                    return self._create(model, system, user).choices[0].message.content or ""
                except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
                    last = e
                    if attempt < len(RETRY_WAITS):
                        time.sleep(RETRY_WAITS[attempt])
                except (openai.NotFoundError, openai.PermissionDeniedError) as e:
                    last = e
                    break
        raise RuntimeError(f"O serviço da IA está sobrecarregado ou indisponível no momento: {last}")

    def json_call(self, system: str, user: str, retries: int = 2) -> dict:
        last_error: Exception | None = None
        for _ in range(retries + 1):
            try:
                return extract_json(self._complete(system, user))
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
        raise RuntimeError(f"Resposta da IA em formato inválido: {last_error}")


@dataclass
class Agent:
    role: str
    goal: str
    backstory: str
    llm: LLM

    def run(self, task: str) -> dict:
        system = (
            f"Você é {self.role}. Objetivo: {self.goal}. Contexto: {self.backstory} "
            "Responda SOMENTE com um objeto JSON válido, sem markdown e sem texto extra."
        )
        return self.llm.json_call(system, task)


SCHEMA_REQ = '{"itens":[{"nome_produto":"","especificacoes_obrigatorias":[""],"quantidade":1,"unidade":"un","requisitos_minimos":[""]}]}'
SCHEMA_QRY = '{"consultas":[{"indice":0,"queries":["",""],"faixa_preco_unitario_brl":{"min":0,"max":0}}]}'
SCHEMA_VAL = '{"validacoes":[{"indice":0,"validos":["id","id"],"recomendado":"id ou null","descartados":[{"id":"","motivo":""}],"descricao_comercial":"","justificativa":""}]}'
SCHEMA_RED = '{"apresentacao":"","itens":[{"indice":0,"descricao_proposta":"","especificacoes":[""]}],"condicoes":{"validade_dias":60,"prazo_entrega":"","garantia":"","pagamento":""}}'


def run_pipeline(llm: LLM, edital: str, empresa: dict, on_step=lambda msg: None) -> dict:
    avisos: list[str] = []

    # Agente 1: Analista de Requisitos
    on_step("🔎 Analisando requisitos e itens do edital...")
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

    # Agente 2: Pesquisa e Validação
    on_step("💰 Planejando buscas de mercado...")
    pesquisador = Agent(
        "Pesquisador de Mercado",
        "gerar consultas eficientes e validar ofertas coletadas",
        "Especialista em compras e pesquisa de preços web.",
        llm,
    )
    qry = pesquisador.run(
        "Para cada item, crie 2 consultas de busca eficientes (4 a 8 palavras) para encontrar o produto em lojas online. "
        "Estime também uma faixa de preço unitário razoável em reais (min e max) apenas para filtro de sanidade.\n"
        f"Formato JSON esperado: {SCHEMA_QRY}\n\nITENS:\n{json.dumps(itens, ensure_ascii=False)}"
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

    cands, sem_preco, logs = collect_quotes(queries, avisos, on_step)
    pesquisa_em = now_br().strftime("%d/%m/%Y %H:%M")

    # Filtro de sanidade de preços
    for i, cs in cands.items():
        fx = faixas.get(i)
        for c in cs:
            c["implausivel"] = bool(fx) and not (0.3 * fx[0] <= c["preco"] <= 3 * fx[1])

    validacoes: dict[int, dict] = {}
    payload = [
        {
            "indice": i,
            "item": itens[i].get("nome_produto"),
            "ofertas": [{k: c[k] for k in ("id", "titulo", "loja", "loja_conhecida", "preco", "origem", "nota", "n_avaliacoes")} for c in cs if not c["implausivel"]]
        }
        for i, cs in cands.items() if any(not c["implausivel"] for c in cs)
    ]
    if payload:
        on_step("🧪 Validando ofertas e selecionando melhor custo-benefício...")
        val = pesquisador.run(
            "Selecione em 'validos' os ids das ofertas compatíveis com o produto. "
            "Priorize menor preço em lojas confiáveis (loja_conhecida=true). Indique a melhor em 'recomendado'.\n"
            f"Formato JSON esperado: {SCHEMA_VAL}\n\nDADOS:\n{json.dumps(payload, ensure_ascii=False)}"
        )
        validacoes = {int(to_float(v.get("indice"), -1)): v for v in val.get("validacoes", [])}

    linhas = []
    for i, it in enumerate(itens):
        nome_item = str(it.get("nome_produto") or f"Item {i + 1}")
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
            avisos.append(f"Item {i + 1} ({nome_item}): ofertas provisórias adotadas. Verifique os links manualmente.")

        if usadas:
            for c in usadas:
                c["usada"] = True
            melhor = by_id[rec] if rec in valid_ids else min(usadas, key=lambda c: c["preco"])
            img_urls = list(dict.fromkeys(c.get("imagem_url") for c in [melhor] + sorted(usadas, key=lambda c: c["preco"]) if c.get("imagem_url")))[:3]
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
            "imagem_auto": None,
            "diagnostico": logs.get(i, []),
            "confianca": confianca,
            "justificativa": str(v.get("justificativa") or ""),
        })

    if any(l["imagem_candidatas"] for l in linhas):
        on_step("🖼️ Baixando imagens dos produtos...")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [(l, ex.submit(first_image, l["imagem_candidatas"])) for l in linhas if l["imagem_candidatas"]]
            for l, f in futs:
                try:
                    l["imagem_auto"] = f.result()[0]
                except Exception:
                    pass

    # Agente 3: Redator de Propostas
    on_step("✍️ Redigindo proposta comercial formal...")
    redator = Agent(
        "Redator de Propostas",
        "organizar termos técnicos e comerciais da proposta",
        "Especialista em redação técnica e comercial para editais.",
        llm,
    )
    resumo = [{"nome": l["nome"], "unidade": l["unidade"], "quantidade": l["quantidade"], "especificacoes": l["especificacoes"], "indice": l["numero"] - 1} for l in linhas]
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
# Geração de PDF (ReportLab)
# ----------------------------------------------------------------------------
def build_pdf(p: dict, anexo: bool = False) -> bytes:
    emp = p["empresa"]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
        title="Proposta Comercial", author=emp.get("razao_social") or "",
    )
    ss = getSampleStyleSheet()
    base = ParagraphStyle("base", parent=ss["Normal"], fontSize=9.5, leading=13)
    title = ParagraphStyle("title", parent=base, fontSize=20, leading=24, alignment=TA_CENTER, textColor=NAVY, fontName="Helvetica-Bold")
    sub = ParagraphStyle("sub", parent=base, alignment=TA_CENTER, textColor=colors.grey)
    h2 = ParagraphStyle("h2", parent=base, fontSize=12, leading=15, textColor=NAVY, fontName="Helvetica-Bold", spaceBefore=12, spaceAfter=5)
    just = ParagraphStyle("just", parent=base, alignment=TA_JUSTIFY)
    th = ParagraphStyle("th", parent=base, textColor=colors.white, fontName="Helvetica-Bold", alignment=TA_CENTER, fontSize=9)
    right = ParagraphStyle("right", parent=base, alignment=TA_RIGHT)
    bold_r = ParagraphStyle("bold_r", parent=right, fontName="Helvetica-Bold")
    center = ParagraphStyle("center", parent=base, alignment=TA_CENTER)
    bullet = ParagraphStyle("bullet", parent=base, leftIndent=14, bulletIndent=3)
    small = ParagraphStyle("small", parent=base, fontSize=8, leading=10)

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
    t = Table([[P(k, ParagraphStyle("k", parent=base, fontName="Helvetica-Bold")), P(v or "—")] for k, v in dados], colWidths=[3.5 * cm, 13.5 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (0, -1), LIGHT), ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, Spacer(1, 6), P("2. Apresentação", h2), P(p["apresentacao"], just)]

    story.append(P("3. Itens e preços", h2))
    rows = [[P(h, th) for h in ("Item", "Descrição", "Un.", "Qtd.", "Preço unit.", "Total")]]
    for l in p["linhas"]:
        rows.append([
            P(l["numero"], center), P(f"{l['nome']}\n{l['descricao_proposta']}"), P(l["unidade"], center),
            P(fmt_qtd(l["quantidade"]), center), P(brl(l["preco_unitario"]), right), P(brl(l["subtotal"]), right),
        ])
    rows.append([P("VALOR TOTAL DA PROPOSTA", bold_r), "", "", "", "", P(brl(p["total"]), bold_r)])
    it = Table(rows, colWidths=[1.3 * cm, 6.7 * cm, 1.3 * cm, 1.7 * cm, 3.0 * cm, 3.0 * cm], repeatRows=1)
    it.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("SPAN", (0, -1), (4, -1)), ("BACKGROUND", (0, -1), (-1, -1), LIGHT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(it)

    story.append(P("4. Especificações técnicas", h2))
    cap = ParagraphStyle("cap", parent=base, fontSize=7.5, leading=9, alignment=TA_CENTER, textColor=colors.grey)
    for l in p["linhas"]:
        story.append(P(f"Item {l['numero']} — {l['nome']}", ParagraphStyle("b", parent=base, fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=2)))
        bullets = [Paragraph(escape(s), bullet, bulletText="•") for s in (l["especificacoes_finais"] or ["Conforme edital."])]
        placed = False
        if l.get("imagem") and len(bullets) <= 25:
            try:
                w, h = PILImage.open(io.BytesIO(l["imagem"])).size
                s = min(4.2 * cm / w, 4.2 * cm / h)
                cell = [RLImage(io.BytesIO(l["imagem"]), width=w * s, height=h * s)]
                if l.get("imagem_legenda"):
                    cell.append(P(l["imagem_legenda"], cap))
                tb = Table([[bullets, cell]], colWidths=[12.6 * cm, 4.4 * cm])
                tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
                story.append(tb)
                placed = True
            except Exception:
                placed = False
        if not placed:
            story += bullets

    c = p["condicoes"]
    story.append(P("5. Condições comerciais", h2))
    for k, v in (("Validade da proposta", f"{c['validade_dias']} dias"), ("Prazo de entrega", c["prazo_entrega"]), ("Garantia", c["garantia"]), ("Pagamento", c["pagamento"])):
        story.append(Paragraph(f"<b>{k}:</b> {escape(str(v))}", base))
        story.append(Spacer(1, 2))

    linha = Table([[""]], colWidths=[8 * cm], rowHeights=[0.1 * cm])
    linha.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black)]))
    story.append(KeepTogether([
        Spacer(1, 18), P(f"{emp.get('cidade') or '____________'}, {p['data']}.", center), Spacer(1, 40), linha, Spacer(1, 3),
        P(emp.get("representante") or "Representante legal", center),
        P(emp.get("cargo") or "", center), P(emp.get("razao_social") or "", center),
    ]))

    if anexo:
        story += [PageBreak(), P("ANEXO — Pesquisa de preços", h2), P(f"Consulta realizada em {p['pesquisa_em']}.", small), Spacer(1, 6)]
        for l in p["linhas"]:
            validas = [q for q in l["cotacoes"] if q.get("usada")]
            story.append(P(f"Item {l['numero']} — {l['nome']}", ParagraphStyle("b2", parent=base, fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)))
            if not validas:
                story.append(P("Sem ofertas utilizadas.", small))
                continue
            rows = [[P(h, th) for h in ("Loja", "Preço", "Link")]]
            for q in validas:
                href = escape(q["url"], {'"': "&quot;"})
                short = escape(q["url"][:70] + ("…" if len(q["url"]) > 70 else ""))
                rows.append([P(q["loja"], small), P(brl(q["preco"]), small), Paragraph(f'<link href="{href}" color="blue">{short}</link>', small)])
            qt = Table(rows, colWidths=[4 * cm, 3 * cm, 10 * cm], repeatRows=1)
            qt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
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
# Interface Streamlit
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Gerador de Propostas", page_icon="📄", layout="wide")
st.session_state.setdefault("gen_id", 0)

with st.sidebar:
    st.header("🏢 Dados do proponente")
    empresa = {
        "razao_social": st.text_input("Razão social"),
        "cnpj": st.text_input("CNPJ"),
        "endereco": st.text_input("Endereço"),
        "contato": st.text_input("E-mail / telefone"),
        "representante": st.text_input("Representante legal"),
        "cargo": st.text_input("Cargo"),
        "cidade": st.text_input("Cidade"),
        "edital": st.text_input("Edital / Pregão nº"),
    }
    with st.expander("⚙️ Configuração da IA"):
        api_key = get_secret("LLM_API_KEY")
        if api_key:
            st.success("Chave de API carregada dos Secrets.")
        else:
            api_key = st.text_input("Chave de API (Gemini)", type="password")
        base_url = st.text_input("Base URL", value=get_secret("LLM_BASE_URL", DEFAULT_BASE_URL))
        want, want_fb = get_secret("LLM_MODEL", DEFAULT_MODEL), get_secret("LLM_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL)
        disponiveis = list_gemini_models(api_key, base_url) if api_key else []
        if disponiveis:
            d_main, d_fb = pick_defaults(disponiveis, want, want_fb)
            model = st.selectbox("Modelo", disponiveis, index=disponiveis.index(d_main))
            fallback_model = st.selectbox("Modelo reserva", disponiveis, index=disponiveis.index(d_fb))
        else:
            model = st.text_input("Modelo", value=want)
            fallback_model = st.text_input("Modelo reserva", value=want_fb)

st.title("📄 Gerador de Propostas para Licitações")
st.caption("Cole a especificação do edital: os agentes pesquisam preços reais na web e geram a proposta em PDF.")

edital_text = st.text_area(
    "Especificação do produto (texto bruto do edital)", height=260,
    placeholder="Ex.: Item 1 — Notebook, processador mínimo 8 núcleos, 16 GB RAM, SSD 512 GB, tela 15,6\". Quantidade: 20 unidades.",
)

if st.button("Gerar Proposta Comercial", type="primary"):
    if not edital_text.strip():
        st.warning("Cole a especificação do edital antes de gerar.")
    elif not api_key:
        st.warning("Informe a chave de API na barra lateral (ou nos Secrets).")
    else:
        try:
            with st.spinner("Os agentes estão trabalhando (a pesquisa web pode levar alguns instantes)..."):
                status = st.empty()
                st.session_state["proposta"] = run_pipeline(
                    LLM(api_key, base_url, model, fallback_model), edital_text.strip(), empresa, on_step=status.info
                )
                st.session_state["gen_id"] += 1
                status.empty()
        except Exception as e:
            st.session_state.pop("proposta", None)
            st.error(f"Erro ao gerar a proposta: {e}")

proposta = st.session_state.get("proposta")
if proposta and proposta.get("_v") != PROPOSTA_V:
    st.session_state.pop("proposta", None)
    proposta = None
    st.info("O app foi atualizado. Gere a proposta novamente.")

if proposta:
    st.divider()
    st.subheader("Revise os valores e itens")
    for aviso in proposta["avisos"]:
        st.warning(aviso)
        
    df = pd.DataFrame([
        {"Item": l["numero"], "Produto": l["nome"], "Un.": l["unidade"], "Qtd": l["quantidade"],
         "Preço unitário": l["preco_unitario"],
         "Faixa encontrada": f"{brl(l['faixa'][0])} – {brl(l['faixa'][1])}" if l["faixa"][1] else "—",
         "Confiança": l["confianca"]}
        for l in proposta["linhas"]
    ])
    edited = st.data_editor(
        df, hide_index=True, disabled=["Item", "Produto", "Un.", "Faixa encontrada", "Confiança"],
        key=f"editor_{st.session_state['gen_id']}",
        column_config={
            "Qtd": st.column_config.NumberColumn(min_value=0.0),
            "Preço unitário": st.column_config.NumberColumn(format="R$ %.2f", min_value=0.0),
        },
    )

    final = copy.deepcopy(proposta)
    for l, (_, row) in zip(final["linhas"], edited.iterrows()):
        l["quantidade"] = to_float(row["Qtd"], l["quantidade"])
        l["preco_unitario"] = to_float(row["Preço unitário"], l["preco_unitario"])
    recalc(final)

    st.metric("Valor total da proposta", brl(final["total"]))
    incluir_imagens = st.checkbox("Incluir imagens dos produtos no PDF", value=True)
    
    for l in final["linhas"]:
        with st.expander(f"Item {l['numero']} — {l['nome']}  ·  ofertas: {len(l['cotacoes'])}  ·  confiança: {l['confianca']}", expanded=not l["cotacoes"]):
            gid = f"{st.session_state['gen_id']}_{l['numero']}"
            img = None
            up = st.file_uploader("Enviar imagem própria (PNG, JPG, WEBP)", type=["png", "jpg", "jpeg", "webp"], key=f"up_{gid}")
            if up is not None:
                img = normalize_image(up.getvalue())
            elif l.get("imagem_auto"):
                if st.checkbox("Usar imagem ilustrativa encontrada", value=True, key=f"auto_{gid}"):
                    img = l["imagem_auto"]
            if img:
                st.image(img, width=180)
            l["imagem"], l["imagem_legenda"] = (img, "Imagem ilustrativa") if incluir_imagens else (None, "")
            
            if l["cotacoes"]:
                st.markdown("**Ofertas encontradas**")
                st.dataframe(
                    pd.DataFrame([
                        {"Usada": q.get("usada", False), "Loja": q["loja"], "Produto": q["titulo"], "Preço": q["preco"], "Link": q["url"]}
                        for q in l["cotacoes"]
                    ]),
                    hide_index=True,
                    column_config={"Preço": st.column_config.NumberColumn(format="R$ %.2f"), "Link": st.column_config.LinkColumn(display_text="abrir")},
                )

    anexo = st.checkbox("Incluir anexo com a pesquisa de preços e links no PDF")
    try:
        st.download_button(
            "⬇️ Baixar proposta em PDF", data=build_pdf(final, anexo), mime="application/pdf",
            file_name=f"proposta_comercial_{now_br().date().isoformat()}.pdf", type="primary",
        )
    except Exception as e:
        st.error(f"Erro ao gerar o PDF: {e}")