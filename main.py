import os
import sys
import re
import json
import logging
import mimetypes
import time
import socket
from dataclasses import dataclass, asdict
from urllib.parse import urlparse
from urllib.error import URLError
from typing import List, Dict, Any, Tuple
import numpy as np
import feedparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from google import genai
from google.genai import types
from feedgen.feed import FeedGenerator
import pytz
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 外部リソース取得時のハングアップを防ぐための全体タイムアウト
socket.setdefaulttimeout(15)

# 調整用パラメータを一箇所に集約し保守性を向上
CONFIG = {
    "gemini_model":           'gemini-3.1-flash-lite',
    "max_grounded":           2,
    "grounding_wait_sec":     60,
    "grounding_max_attempts": 3,
    "grounding_retry_wait":   90,
    "filter_k_sigma":         0.5,
    "filter_n_topics":        8,
    "filter_alpha":           0.5,
    "filter_beta":            0.5,
    "filter_sigmoid_scale":   8.0,
    "max_process":            15,
    "cache_file":             "processed_urls.json",
    "max_cache_size":         500,
}

SOURCE_RSS_URLS = [
    "https://rss.itmedia.co.jp/rss/2.0/business.xml",
    "https://prtimes.jp/index.rdf",
    "https://assets.wor.jp/rss/rdf/nikkei/news.rdf",
    "https://feeds.japan.cnet.com/rss/cnet/all.rdf",
    "https://news.yahoo.co.jp/rss/topics/domestic.xml",
    "https://news.yahoo.co.jp/rss/topics/world.xml",
    "https://news.yahoo.co.jp/rss/topics/business.xml",
    "https://feeds.bbci.co.uk/japanese/rss.xml"
]

EXCLUDE_KEYWORDS = ["有料会員", "会員限定", "ログイン", "この記事は有料", "続きは有料", "プレミアム", "🔒"]

INTEREST_TEXTS = [
    "IT技術、プログラミング、人工知能、機械学習、データ分析、アルゴリズム、クラウド",
    "ガジェット、スマートフォン、iPhone、Android、デジタル家電、ウェアラブル",
    "音楽、芸術、ゲーム、推し活、アニメ、同人、クリエイター",
    "旅、自然、登山、キャンプ、アウトドア"
]

DISINTEREST_TEXTS = [
    "政治、選挙、国会、法律、政党、議員、外交、安全保障、防衛",
    "スポーツ、野球、サッカー、バスケ、競馬、格闘技、オリンピック、Jリーグ",
    "芸能、タレント、アイドル、ドラマ、ゴシップ、不倫、芸人",
    "料理、レシピ、グルメ、食事、飲食店、レストラン、食材",
    "不動産、住宅、マンション、土地、建築、引越し、住まい"
]


# =============================================================================
# データクラス
# =============================================================================

@dataclass
class Article:
    """RSSエントリから整形された記事データ。"""
    url: str
    original_title: str
    summary: str          # AI生成に使う要約（短すぎる場合は本文冒頭で補完済み）
    original_html: str    # RSSフィード表示用の元HTML
    image_url: str
    feed_title: str
    pub_date_iso: str
    entry: Any            # feedparserエントリ（ベクトル化処理で使用）

@dataclass
class ScoredArticle:
    """フィルタリングスコアを付与した記事。"""
    article: Article
    sim_interest: float   # 興味クラスタとのコサイン類似度（低いほど非興味）
    sim_diff: float       # disinterest類似度 - interest類似度（高いほど非興味）
    combined_score: float = 0.0  # 最終アンサンブルスコア

@dataclass
class ProcessedArticle:
    """LLM生成済みのフィード出力記事。"""
    id: str
    title: str
    link: str
    description: str
    content: str
    pubDate: str
    enclosure: str


# =============================================================================
# ユーティリティ
# =============================================================================

def get_mime_type(url: str) -> str:
    if not url:
        return 'image/jpeg'
    mime_type, _ = mimetypes.guess_type(urlparse(url).path)
    return mime_type or 'image/jpeg'

def strip_html(text: str) -> str:
    """HTMLタグ・URL・記号などを除去してプレーンテキスト化する。NMF/embedding前処理用。"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    text = re.sub(r'(?:https?|ftp)://[^\s]+', ' ', text)
    text = re.sub(r'[?#][^\s]*', ' ', text)
    text = re.sub(r'[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}', ' ', text)
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', ' ', text)
    text = re.sub(r'(?:/|\\)[^\s]*', ' ', text)
    text = re.sub(r'\b[a-zA-Z]{1,2}\b', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'[^\w\s\u3002\u3001\uff01\uff1f\u300c\u300d\u300e\u300f\uff08\uff09\u30fb\u30fc]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# =============================================================================
# キャッシュ
# =============================================================================

def load_cache() -> Tuple[List[str], List[Dict]]:
    if not os.path.exists(CONFIG["cache_file"]):
        logger.info("初回実行、またはキャッシュが存在しません。")
        return [], []
    try:
        with open(CONFIG["cache_file"], 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 旧バージョンのキャッシュ形式との互換性維持
            if isinstance(data, list):
                logger.info("旧形式のキャッシュを検出。フォーマットを移行します。")
                return data, []
            seen_urls = data.get("seen_urls", [])
            last_run  = data.get("last_run_articles", [])
            logger.info(f"キャッシュ読込成功（既読URL: {len(seen_urls)}件, 前回記事: {len(last_run)}件）")
            return seen_urls, last_run
    except Exception as e:
        logger.warning(f"キャッシュの読み込みに失敗: {e}")
        return [], []

def save_cache(seen_urls: List[str], articles: List[Dict]) -> None:
    try:
        data = {
            "seen_urls":          seen_urls[-CONFIG["max_cache_size"]:],
            "last_run_articles":  articles
        }
        with open(CONFIG["cache_file"], 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"キャッシュ保存成功（URL保持: {len(data['seen_urls'])}件, 保存記事: {len(articles)}件）")
    except Exception as e:
        logger.error(f"キャッシュ保存失敗: {e}")


# =============================================================================
# 記事パース
# =============================================================================

def extract_image_url(entry: Any, original_html: str) -> str:
    """RSSの構造に応じて複数パスでフォールバック検索を行う。"""
    for link in getattr(entry, 'links', []):
        if 'image' in link.get('type', ''):
            return link.get('href', '')
    for attr in ['media_content', 'media_thumbnail']:
        media = getattr(entry, attr, [])
        if media:
            return media[0].get('url', '')
    match = re.search(r'<img[^>]+src=["\']([^\'\"]+)["\']', original_html)
    return match.group(1) if match else ''

def is_excluded(entry: Any) -> bool:
    """有料記事・除外キーワードを含む場合はTrue。"""
    text = f"{getattr(entry, 'title', '')} {getattr(entry, 'summary', getattr(entry, 'description', ''))}"
    return any(kw in text for kw in EXCLUDE_KEYWORDS)

def parse_entry_to_article(entry: Any, feed_title: str) -> Article:
    """feedparserエントリをArticleデータクラスに整形する。"""
    url     = getattr(entry, 'link', '').strip().split('#')[0]
    title   = getattr(entry, 'title', '')
    summary = getattr(entry, 'summary', getattr(entry, 'description', ''))

    # 概要が短すぎる場合は本文冒頭で補完（最大300文字）
    if not summary or len(summary.strip()) < 10:
        content_obj = getattr(entry, 'content', [])
        if content_obj and content_obj[0].get('value'):
            summary = content_obj[0].get('value', '')[:300]

    content_obj   = getattr(entry, 'content', [])
    original_html = content_obj[0].get('value', summary) if content_obj else summary

    pub_parsed = getattr(entry, 'published_parsed', None)
    dt = (
        pytz.utc.localize(datetime(*pub_parsed[:6])).astimezone(pytz.timezone('Asia/Tokyo'))
        if pub_parsed else datetime.now(pytz.timezone('Asia/Tokyo'))
    )

    return Article(
        url=url,
        original_title=title,
        summary=summary,
        original_html=original_html,
        image_url=extract_image_url(entry, original_html),
        feed_title=feed_title,
        pub_date_iso=dt.isoformat(),
        entry=entry
    )


# =============================================================================
# RSS取得
# =============================================================================

def fetch_free_articles(urls: List[str], seen_links: List[str]) -> List[Article]:
    articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            # bozo例外でタイムアウトとそれ以外を区別して適切にハンドリング
            if getattr(feed, 'bozo', 0) == 1:
                exc = getattr(feed, 'bozo_exception', 'Unknown error')
                if isinstance(exc, URLError) and isinstance(exc.reason, socket.timeout):
                    logger.error(f"URL取得タイムアウト ({url})")
                    continue
                logger.warning(f"フィード解析警告 ({url}): {exc}")

            feed_title = getattr(feed.feed, 'title', url)
            collected = excluded = skipped = 0

            for entry in feed.entries:
                if collected >= 10:
                    break
                raw_link = getattr(entry, 'link', '').strip().split('#')[0]
                if raw_link in seen_links:
                    skipped += 1
                    continue
                if is_excluded(entry):
                    excluded += 1
                    continue
                articles.append(parse_entry_to_article(entry, feed_title))
                collected += 1

            if not (excluded == 0 and skipped == 0 and collected == 0):
                logger.info(f"取得完了: {url} (新規: {collected}件 / 除外: {excluded}件 / 既読スキップ: {skipped}件)")
        except Exception as e:
            logger.error(f"URL取得予期せぬエラー ({url}): {e}")
    return articles


# =============================================================================
# フィルタリング（NMF + スコアリング）
# =============================================================================

def _sigmoid(arr: np.ndarray) -> np.ndarray:
    """ゼロ基準を保持したままsigmoid正規化。diff > 0 → 0.5以上。"""
    return 1.0 / (1.0 + np.exp(-CONFIG["filter_sigmoid_scale"] * arr))

def compute_nmf_noninterest_scores(articles: List[Article], embedder: SentenceTransformer) -> np.ndarray:
    """
    NMFで潜在トピックを抽出し、各記事の「非興味スコア」を返す。
    文字n-gram TF-IDFで日本語をトークナイザなしに処理。
    """
    texts    = [strip_html(f"{a.original_title} {a.summary}") for a in articles]
    n_topics = min(CONFIG["filter_n_topics"], max(2, len(texts) // 2))
    try:
        vectorizer    = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), max_features=3000, min_df=1)
        tfidf_matrix  = vectorizer.fit_transform(texts)
        nmf           = NMF(n_components=n_topics, random_state=42, max_iter=400)
        doc_topic     = nmf.fit_transform(tfidf_matrix)
        feature_names = vectorizer.get_feature_names_out()

        topic_repr_texts = []
        for i in range(n_topics):
            top_words = [feature_names[j] for j in nmf.components_[i].argsort()[-15:][::-1]]
            topic_repr_texts.append(" ".join(top_words))
            logger.info(f"[NMF] トピック{i}: {' '.join(top_words[:6])}")

        topic_vecs    = embedder.encode([f"query: {t}" for t in topic_repr_texts])
        interest_vecs = embedder.encode([f"query: {t}" for t in INTEREST_TEXTS])
        # トピックと興味クラスタの類似度を反転して「非興味度」を算出
        topic_ni       = 1.0 - cosine_similarity(topic_vecs, interest_vecs).max(axis=1)
        doc_topic_norm = doc_topic / (doc_topic.sum(axis=1, keepdims=True) + 1e-8)
        scores         = doc_topic_norm @ topic_ni
        return scores - scores.mean()  # 平均中央化してsigmoidと尺度を揃える
    except Exception as e:
        logger.error(f"NMF処理エラー: {e}。スコアを0で代替します。")
        return np.zeros(len(articles))

def filter_articles(articles: List[Article], embedder: SentenceTransformer) -> Tuple[List[ScoredArticle], bool]:
    """
    3段階フィルタリング:
      ① 負例差分スコア（disinterest類似度 - interest類似度）
      ② NMFスコア（非興味トピック割合）
      ③ 統計的動的閾値（mean + K_SIGMA * std）
    """
    if not articles:
        return [], False

    interest_vecs    = embedder.encode([f"query: {t}" for t in INTEREST_TEXTS])
    disinterest_vecs = embedder.encode([f"query: {t}" for t in DISINTEREST_TEXTS])
    passage_texts    = [f"passage: {strip_html(a.original_title)} {strip_html(a.summary)}" for a in articles]

    logger.info(f"ベクトル化を一括実行中... ({len(articles)}件)")
    article_vecs = embedder.encode(passage_texts)

    scored_list: List[ScoredArticle] = []
    diff_scores:  List[float]        = []
    for idx, article in enumerate(articles):
        sim_i  = float(cosine_similarity([article_vecs[idx]], interest_vecs)[0].max())
        sim_di = float(cosine_similarity([article_vecs[idx]], disinterest_vecs)[0].max())
        diff   = sim_di - sim_i
        diff_scores.append(diff)
        scored_list.append(ScoredArticle(article=article, sim_interest=sim_i, sim_diff=diff))

    logger.info("NMFによる潜在トピックスコア算出中...")
    nmf_scores = compute_nmf_noninterest_scores(articles, embedder)

    # 差分スコアとNMFスコアをsigmoidで正規化してアンサンブル
    combined      = CONFIG["filter_alpha"] * _sigmoid(np.array(diff_scores)) + CONFIG["filter_beta"] * _sigmoid(nmf_scores)
    mean_c, std_c = float(combined.mean()), float(combined.std())
    threshold     = mean_c + CONFIG["filter_k_sigma"] * std_c
    logger.info(f"アンサンブルスコア: mean={mean_c:.3f} std={std_c:.3f} threshold={threshold:.3f}")

    for i, scored in enumerate(scored_list):
        scored.combined_score = float(combined[i])

    target      = [s for s in scored_list if s.combined_score >= threshold]
    is_fallback = not target

    # 閾値を超える記事がない場合、フィード枯渇を防ぐため上位3件を強制抽出
    if is_fallback:
        logger.warning("閾値超え記事が0件。スコア上位3件を強制抽出。")
        scored_list.sort(key=lambda x: x.combined_score, reverse=True)
        target = scored_list[:3]

    target.sort(key=lambda x: x.combined_score, reverse=True)
    logger.info(f"抽出記事数: {len(target)}件 (fallback={is_fallback})")
    return target, is_fallback


# =============================================================================
# AI生成
# =============================================================================

def _build_prompt(title: str, summary: str) -> str:
    return f"""
あなたは知的好奇心を持つ読者向けに、専門外のニュースを「わかりやすく・正確に」伝えるライターです。

以下の[対象テキスト]をもとに、「タイトル」と「解説」を作成してください。

=== タイトルのルール ===
- [対象テキスト]の事実を忠実に反映し、無理のない言い換えにとどめること。
- 「何が起きたか：何が変わるか」の2部構成を基本とする。
- 35文字以内。
- ドメイン変換（政治ニュースをIT用語で表現するなど）は絶対に行わないこと。

=== 解説のルール（4セクション構成）===

【何があったか】
- [対象テキスト]の事実のみを使い、3〜4文で要約する。
- 推測・補完・誇張は一切しない。

【知っておくべき背景】
- このニュースを理解するために必要な前提知識・業界構造・制度・歴史的経緯を説明する。
- 不確かな情報は「〜とされている」など断定を避けた表現を使う。

【何に影響するか】
- 社会・経済・生活への波及効果を中立的な視点で説明する。
- 特定の立場への誘導や感情的な表現は避ける。

【あなたの興味との接点】
- IT技術・ガジェット・音楽・アート・旅・自然などとの関連があれば具体的に示す。
- 直接的な接点が薄い場合は「このニュースと直接的な接点は薄いですが、〜という観点では〜」のように正直に書く。
- 無理な関連付けや比喩は絶対に使わないこと。

=== 共通ルール ===
- 全体で400〜600文字程度。
- 強調箇所は <strong> タグを使用。Markdownの ** は使わない。
- 見出しは【】形式のみ（HTMLの h タグ不要）。

=== 出力フォーマット ===
【タイトル】
(タイトル本文)

【解説】
【何があったか】
(本文)

【知っておくべき背景】
(本文)

【何に影響するか】
(本文)

【あなたの興味との接点】
(本文)

=== 対象テキスト ===
【{title}】
{summary}
"""

def _parse_response(response: Any, title: str, summary: str) -> dict:
    """APIレスポンスからtitleとexplanationを抽出する。"""
    result = {"title": title, "explanation": ""}
    if response.text:
        text = response.text.strip()
        tm   = re.search(r'【タイトル】\s*(.*?)\s*【解説】', text, re.DOTALL)
        em   = re.search(r'【解説】\s*(.*)', text, re.DOTALL)
        if tm and em:
            result["title"]       = tm.group(1).strip()
            result["explanation"] = em.group(1).strip()
        else:
            result["explanation"] = text
    else:
        result["explanation"] = (
            f"【何があったか】\n{summary}\n\n【知っておくべき背景】\n（生成エラー）\n\n"
            f"【何に影響するか】\n（生成エラー）\n\n【あなたの興味との接点】\n（生成エラー）"
        )
    return result

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=10, max=120))
def generate_without_grounding(client: Any, title: str, summary: str) -> dict:
    """第1パス: Grounding無しで生成（安定・高速）。"""
    response = client.models.generate_content(
        model=CONFIG["gemini_model"],
        contents=_build_prompt(title, summary),
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return _parse_response(response, title, summary)

def generate_with_grounding_fallback(client: Any, title: str, summary: str) -> Tuple[dict, bool]:
    """
    第2パス: Groundingで再生成を試みる。
    429が続く場合は grounding_retry_wait 秒待機してリトライ。
    全試行失敗時は Grounding無しで生成してフォールバック。
    戻り値: (result_dict, grounding_used: bool)
    """
    grounding_config = types.GenerateContentConfig(
        temperature=0.2,
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    for attempt in range(1, CONFIG["grounding_max_attempts"] + 1):
        try:
            logger.info(f"  Grounding試行 {attempt}/{CONFIG['grounding_max_attempts']}...")
            response = client.models.generate_content(
                model=CONFIG["gemini_model"],
                contents=_build_prompt(title, summary),
                config=grounding_config
            )
            logger.info(f"  Grounding成功 (attempt {attempt})")
            return _parse_response(response, title, summary), True
        except Exception as e:
            if "429" in str(e):
                if attempt < CONFIG["grounding_max_attempts"]:
                    logger.warning(f"  Grounding 429 (attempt {attempt})。{CONFIG['grounding_retry_wait']}秒待機後リトライ...")
                    time.sleep(CONFIG["grounding_retry_wait"])
                else:
                    logger.warning(f"  Grounding全試行失敗（{CONFIG['grounding_max_attempts']}回）。Grounding無しで生成します。")
            else:
                logger.error(f"  Grounding予期せぬエラー: {e}。Grounding無しで生成します。")
                break

    # フォールバック: Grounding無しで生成
    try:
        return generate_without_grounding(client, title, summary), False
    except Exception as e:
        logger.error(f"フォールバック生成も失敗: {e}")
        return {
            "title": title,
            "explanation": (
                f"【何があったか】\n{summary}\n\n【知っておくべき背景】\nLLM生成エラー。元の記事を参照してください。\n\n"
                f"【何に影響するか】\n（生成エラー）\n\n【あなたの興味との接点】\n（生成エラー）"
            )
        }, False


# =============================================================================
# HTML組み立て
# =============================================================================

def build_description_html(
    scored: ScoredArticle,
    original_title: str,
    ai_explanation: str,
    original_html: str,
    image_url: str,
    is_fallback: bool,
    grounding_used: bool
) -> str:
    fallback_notice = "<p>※出力確保のための抽出記事</p>" if is_fallback else ""
    grounding_badge = (
        '<span style="background:#e6f4ea;color:#1e7e34;font-size:small;padding:2px 6px;border-radius:4px;">'
        'Grounding有効</span> '
    ) if grounding_used else ""
    ai_exp_html = re.sub(r'\n{2,}', '\n', ai_explanation).strip().replace('\n', '<br>')
    img_html    = f'<p><img src="{image_url}" style="max-width:100%; height:auto;" /></p>' if image_url else ""
    return (
        f'<p style="color:gray; font-size: small;">'
        f'{grounding_badge}元タイトル「{original_title}」<br>'
        f'興味類似度: {scored.sim_interest:.3f} / 差分スコア: {scored.sim_diff:.3f} / 総合スコア: {scored.combined_score:.3f}'
        f'</p>{fallback_notice}<h3>LLM解説</h3>{img_html}<p>{ai_exp_html}</p><hr><h3>元の記事</h3>{original_html}'
    )


# =============================================================================
# 2パス処理
# =============================================================================

def process_articles(
    client: Any,
    scored_articles: List[ScoredArticle],
    is_fallback: bool
) -> List[ProcessedArticle]:
    """
    第1パス: 全記事をGrounding無しで生成（安定・高速）。
    第2パス: combined_score最上位N件のみGroundingで再生成し結果を上書き。
    """
    target = scored_articles[:CONFIG["max_process"]]

    # article.url をキーに、AI生成結果（title / explanation / grounding_used）を保持
    generation_map: Dict[str, Dict] = {}

    # --- 第1パス ---
    logger.info("3a. 第1パス: 全記事をGrounding無しで生成")
    for scored in target:
        article = scored.article
        try:
            result = generate_without_grounding(client, article.original_title, article.summary)
        except Exception as e:
            logger.error(f"第1パス生成失敗: {e}")
            result = {
                "title": article.original_title,
                "explanation": (
                    f"【何があったか】\n{article.summary}\n\n【知っておくべき背景】\nLLM生成エラー。元の記事を参照してください。\n\n"
                    f"【何に影響するか】\n（生成エラー）\n\n【あなたの興味との接点】\n（生成エラー）"
                )
            }
        generation_map[article.url] = {**result, "grounding_used": False}
        logger.info(f"[第1パス] 完了: {result['title'][:30]}...")
        time.sleep(8)

    # --- 第2パス: combined_score最上位N件のみGroundingで再生成し上書き ---
    grounding_targets = sorted(target, key=lambda x: x.combined_score, reverse=True)[:CONFIG["max_grounded"]]
    logger.info(f"3b. 第2パス: Grounding再生成 ({len(grounding_targets)}件) 開始")

    for scored in grounding_targets:
        article = scored.article
        logger.info(f"[第2パス] {CONFIG['grounding_wait_sec']}秒待機後にGrounding試行: {article.original_title[:30]}...")
        time.sleep(CONFIG["grounding_wait_sec"])

        result, grounding_used = generate_with_grounding_fallback(client, article.original_title, article.summary)
        generation_map[article.url] = {**result, "grounding_used": grounding_used}

        status = "Grounding有効" if grounding_used else "Grounding失敗→fallback"
        logger.info(f"[第2パス] 完了({status}): {result['title'][:30]}...")

    # ProcessedArticle に変換して返す（targetの並び順を保持）
    return [
        ProcessedArticle(
            id          = scored.article.url,
            title       = generation_map[scored.article.url]["title"],
            link        = scored.article.url,
            description = f"総合スコア: {scored.combined_score:.3f} | ソース: {scored.article.feed_title}",
            content     = build_description_html(
                              scored,
                              scored.article.original_title,
                              generation_map[scored.article.url]["explanation"],
                              scored.article.original_html,
                              scored.article.image_url,
                              is_fallback,
                              generation_map[scored.article.url]["grounding_used"]
                          ),
            pubDate     = scored.article.pub_date_iso,
            enclosure   = scored.article.image_url,
        )
        for scored in target
    ]


# =============================================================================
# フィード生成
# =============================================================================

def build_feed(processed: List[ProcessedArticle], last_run_articles: List[Dict]) -> List[Dict]:
    """現在実行分と前回分をIDでマージし、rss.xmlとindex.htmlを出力する。"""
    # 前回分をベースに現在分で上書きマージ（ID重複を排除）
    unique: Dict[str, Dict] = {art["id"]: art for art in last_run_articles}
    for article in processed:
        unique[article.id] = asdict(article)

    all_articles = sorted(unique.values(), key=lambda x: x["pubDate"], reverse=True)[:30]

    fg = FeedGenerator()
    fg.title('LLM再構築フィード v2')
    fg.link(href='https://github.com/', rel='alternate')
    fg.description('負例クラスタ差分・NMF・動的閾値による3段階フィルタで、興味外ニュースを抽出・LLM解説するカスタムフィード')
    fg.language('ja')

    for art in all_articles:
        fe = fg.add_entry()
        fe.id(art["id"])
        fe.title(art["title"])
        fe.link(href=art["link"])
        fe.description(art["description"])
        fe.content(content=art["content"], type='html')
        fe.pubDate(datetime.fromisoformat(art["pubDate"]))
        if art.get("enclosure"):
            fe.enclosure(art["enclosure"], 0, get_mime_type(art["enclosure"]))

    fg.rss_file('rss.xml')
    logger.info(f"rss.xml 生成完了 (出力件数: {len(all_articles)}件)")

    html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM再構築フィード v2</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2rem; color: #333; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
        .rss-link {{ display: inline-block; background: #ee802f; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 1rem; }}
        .rss-link:hover {{ background: #c66a26; }}
        .badge {{ display: inline-block; background: #e8f4fd; color: #1a6ca8; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px; }}
    </style>
</head>
<body>
    <h1>LLM再構築フィード v2</h1>
    <p>フィルターバブルを打破するため、ユーザーの関心領域外のニュースをLLM（Gemini）が構造化して配信するカスタムフィードです。</p>
    <p>
        <span class="badge">負例クラスタ差分スコア</span>
        <span class="badge">NMFトピックモデル</span>
        <span class="badge">統計的動的閾値</span>
        による3段階フィルタリングを実装。上位{CONFIG['max_grounded']}件はGoogle Search Groundingで最新情報を付加。
    </p>
    <p>最終更新: {datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} (JST)</p>
    <h2>利用方法</h2>
    <p>お使いのRSSリーダー（Feedly, Inoreaderなど）に以下のリンクを登録してください。</p>
    <a href="rss.xml" class="rss-link">RSSフィード (rss.xml) を取得</a>
    <p style="margin-top: 3rem; font-size: 0.8rem; color: #666;">Powered by GitHub Actions & Gemini API</p>
</body>
</html>
"""
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    logger.info("index.html 生成完了")

    return all_articles


# =============================================================================
# エントリポイント
# =============================================================================

def main():
    try:
        gemini_key = os.environ.get("API_KEY1")
        hf_token   = os.environ.get("HF_TOKEN1")
        if not gemini_key:
            logger.critical("API_KEY1 が未設定")
            sys.exit(1)
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        logger.info(
            f"設定: MAX_PROCESS={CONFIG['max_process']} / "
            f"MAX_GROUNDED={CONFIG['max_grounded']} / "
            f"GROUNDING_WAIT={CONFIG['grounding_wait_sec']}s"
        )

        seen_links, last_run_articles = load_cache()

        articles = fetch_free_articles(SOURCE_RSS_URLS, seen_links)
        logger.info(f"合計新規抽出数: {len(articles)}件")

        processed: List[ProcessedArticle] = []
        if articles:
            embedder = SentenceTransformer('intfloat/multilingual-e5-small')
            scored_articles, is_fallback = filter_articles(articles, embedder)
            logger.info(f"処理対象: {min(len(scored_articles), CONFIG['max_process'])}件")

            client    = genai.Client(api_key=gemini_key)
            processed = process_articles(client, scored_articles, is_fallback)

            for p in processed:
                if p.id not in seen_links:
                    seen_links.append(p.id)

        all_articles = build_feed(processed, last_run_articles)
        save_cache(seen_links, all_articles)

    except Exception as e:
        logger.critical(f"予期せぬ致命的なエラーにより処理が中断されました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
