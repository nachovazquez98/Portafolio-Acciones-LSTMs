"""Noticias por acción (Yahoo Finance + Google News RSS) con puntaje de sentimiento.

Las APIs gratuitas sólo devuelven noticias recientes, así que cada ejecución
agrega lo descargado a data/news.parquet. Con el tiempo ese histórico propio
permite usar el sentimiento como feature del modelo.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import yfinance as yf

from stockcast import config

log = logging.getLogger(__name__)
GOOGLE_RSS = "https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def _query_for(ticker: str) -> str:
    base = ticker.replace(".MX", "").replace("-USD", "")
    return f'"{base}" acciones' if ticker.endswith(".MX") else f"{base} stock"


def _yahoo(ticker: str) -> list[dict]:
    items = []
    try:
        for n in yf.Ticker(ticker).news or []:
            c = n.get("content", n)
            title = c.get("title")
            if not title:
                continue
            url = (c.get("canonicalUrl") or {}).get("url") or c.get("link")
            when = c.get("pubDate") or c.get("providerPublishTime")
            published = (pd.to_datetime(when, unit="s", utc=True) if isinstance(when, (int, float))
                         else pd.to_datetime(when, utc=True))
            items.append({"title": title, "summary": c.get("summary", ""), "url": url,
                          "source": (c.get("provider") or {}).get("displayName", "Yahoo Finance"),
                          "published": published})
    except Exception as e:  # yfinance cambia su esquema con frecuencia
        log.debug("Yahoo news falló para %s: %s", ticker, e)
    return items


def _google(ticker: str, lang: dict[str, str]) -> list[dict]:
    feed = feedparser.parse(GOOGLE_RSS.format(q=quote_plus(_query_for(ticker)), **lang))
    items = []
    for e in feed.entries:
        published = pd.to_datetime(datetime(*e.published_parsed[:6], tzinfo=timezone.utc)) \
            if getattr(e, "published_parsed", None) else pd.NaT
        items.append({"title": e.title, "summary": "", "url": e.link,
                      "source": getattr(e, "source", {}).get("title", "Google News"),
                      "published": published})
    return items


@lru_cache(maxsize=1)
def _scorer():
    method = config.load()["news"]["sentiment"]
    if method == "finbert":
        try:
            from transformers import pipeline
            clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)

            def score(texts: list[str]) -> list[float]:
                out = []
                for res in clf(texts, truncation=True):
                    d = {r["label"]: r["score"] for r in res}
                    out.append(d.get("positive", 0) - d.get("negative", 0))
                return out
            return score
        except ImportError:
            log.warning("transformers no instalado; usando VADER")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    return lambda texts: [vader.polarity_scores(t)["compound"] for t in texts]


def fetch_news(ticker: str, market_key: str) -> pd.DataFrame:
    """Noticias recientes de un ticker con columna `sentiment` en [-1, 1]."""
    lang = config.market(market_key).news_lang
    # Yahoo Finance limita las peticiones de noticias (rate limit) y es redundante con Google News;
    # se usa sólo si se activa en config (news.yahoo: true).
    items = (_yahoo(ticker) if config.load()["news"].get("yahoo", False) else []) + _google(ticker, lang)
    if not items:
        return pd.DataFrame(columns=["ticker", "title", "summary", "url", "source", "published", "sentiment"])
    df = pd.DataFrame(items).drop_duplicates("title")
    df["published"] = pd.to_datetime(df["published"], utc=True)
    df = df.sort_values("published", ascending=False).head(config.load()["news"]["max_items_per_ticker"])
    df["sentiment"] = _scorer()((df["title"] + ". " + df["summary"].fillna("")).tolist())
    df.insert(0, "ticker", ticker)
    return df.reset_index(drop=True)


def update_news_history(tickers: list[str], market_key: str) -> pd.DataFrame:
    """Descarga noticias de todos los tickers y las agrega al histórico local."""
    frames = [fetch_news(t, market_key) for t in tickers]
    new = pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()
    cache = config.path("data") / "news.parquet"
    if cache.exists():
        new = pd.concat([pd.read_parquet(cache), new], ignore_index=True)
    if not new.empty:
        new = new.drop_duplicates(["ticker", "title"], keep="last")
        new.to_parquet(cache, index=False)
    return new


def daily_sentiment(news: pd.DataFrame) -> pd.DataFrame:
    """Sentimiento promedio diario por ticker (para usar como feature cuando haya histórico)."""
    if news.empty:
        return pd.DataFrame(columns=["date", "ticker", "news_sent", "news_count"])
    d = news.assign(date=news["published"].dt.tz_convert(None).dt.normalize())
    return (d.groupby(["date", "ticker"])["sentiment"].agg(news_sent="mean", news_count="size")
            .reset_index())
