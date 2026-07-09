import os
import sys
import re
import json
import logging
import mimetypes
import time
import socket
from urllib.parse import urlparse
from urllib.error import URLError
from typing import List, Dict, Any
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

socket.setdefaulttimeout(15)

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

GEMINI_MODEL_NAME = 'gemini-3.1-flash-lite'

USE_GROUNDING = os.environ.get("USE_GROUNDING", "false").lower() == "true"

K_SIGMA        = 0.5
N_TOPICS       = 8
ALPHA          = 0.5
BETA           = 0.5
SIGMOID_SCALE  = 8.0

# [修正①] Grounding有効時は処理記事数を上限8件に制限
# Grounding=True時: 1記事あたり最大10コール消費のため多すぎると429が連発する
MAX_PROCESS_PER_RUN = 8 if USE_GROUNDING else 15

CACHE_FILE = "processed_urls.json"
MAX_CACHE_SIZE = 500


def get_mime_type(url: str) -> str:
    if not url:
        return 'image/jpeg'
    parsed_url = urlparse(url)
    mime_type, _ = mimetypes.guess_type(parsed_url.path)
    return mime_type or 'image/jpeg'


def strip_html(text: str) -> str:
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
    text = re.sub(r'[^\w\s。、！？「」『』（）・ー]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_cache() -> tuple[List[str], List[Dict]]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    logger.info("旧形式のキャッシュを検出。フォーマットを移行します。")
                    return data, []
                seen_urls = data.get("seen_urls", [])
                last_run_articles = data.get("last_run_articles", [])
                logger.info(f"キャッシュ読込成功（既読URL: {len(seen_urls)}件, 前回記事: {len(last_run_articles)}件）")
                return seen_urls, last_run_articles
        except Exception as e:
            logger.warning(f"キャッシュの読み込みに失敗: {e}")
    else:
        logger.info("初回実行、またはキャッシュが存在しません。")
    return [], []


def save_cache(seen_urls: List[str], current_run_articles: List[Dict]):
    try:
        limited_urls = seen_urls[-MAX_CACHE_SIZE:]
        data = {"seen_urls": limited_urls, "last_run_articles": current_run_articles}
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"キャッシュ保存成功（URL保持: {len(limited_urls)}件, 保存記事: {len(current_run_articles)}件）")
    except Exception as e:
        logger.error(f"キャッシュの保存失敗: {e}")


def extract_image_url(entry: Any, original_html: str) -> str:
    links = getattr(entry, 'links', [])
    image_url = next((link.get('href') for link in links if 'image' in link.get('type', '')), '')
    if not image_url:
        for attr in ['media_content', 'media_thumbnail']:
            media = getattr(entry, attr, [])
            if media:
                image_url = media[0].get('url', '')
                break
    if not image_url and 'src=' in original_html:
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', original_html)
        if img_match:
            image_url = img_match.group(1)
    return image_url


def fetch_free_articles(urls: List[str], seen_links: List[str], max_per_feed: int = 10) -> List[Any]:
    free_articles = []
    for url in urls:
        try:
            feed = feedparser.parse(url)
            if getattr(feed, 'bozo', 0) == 1:
                exception = getattr(feed, 'bozo_exception', 'Unknown error')
                if isinstance(exception, URLError) and isinstance(exception.reason, socket.timeout):
                    logger.error(f"URL取得タイムアウト ({url})")
                    continue
                else:
                    logger.warning(f"フィード解析警告 ({url}): {exception}")

            feed_title = getattr(feed.feed, 'title', url)
            collected = 0
            excluded_count = 0
            skipped_count = 0

            for entry in feed.entries:
                if collected >= max_per_feed:
                    break
                raw_link = getattr(entry, 'link', '').strip().split('#')[0]
                if raw_link in seen_links:
                    skipped_count += 1
                    continue
                summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
                title = getattr(entry, 'title', '')
                text_to_check = f"{title} {summary}"
                excluded_kw = next((kw for kw in EXCLUDE_KEYWORDS if kw in text_to_check), None)
                if excluded_kw:
                    excluded_count += 1
                    continue
                entry['feed_title'] = feed_title
                free_articles.append(entry)
                collected += 1

            if not (excluded_count == 0 and skipped_count == 0 and collected == 0):
                logger.info(f"取得完了: {url} (新規: {collected}件 / 除外: {excluded_count}件 / 既読スキップ: {skipped_count}件)")
        except Exception as e:
            logger.error(f"URL取得予期せぬエラー ({url}): {e}")
    return free_articles


def _sigmoid(arr: np.ndarray, scale: float = SIGMOID_SCALE) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scale * arr))


def compute_nmf_noninterest_scores(
    articles: List[Any],
    embedder: SentenceTransformer,
    interest_texts: List[str],
    n_topics: int = N_TOPICS
) -> List[float]:
    texts = [
        strip_html(f"{getattr(e, 'title', '')} {getattr(e, 'summary', getattr(e, 'description', ''))}")
        for e in articles
    ]
    n_topics = min(n_topics, max(2, len(texts) // 2))

    try:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), max_features=3000, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(texts)
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
        doc_topic = nmf.fit_transform(tfidf_matrix)
        topic_word = nmf.components_

        feature_names = vectorizer.get_feature_names_out()
        topic_repr_texts = []
        for topic_idx in range(n_topics):
            top_words = [feature_names[i] for i in topic_word[topic_idx].argsort()[-15:][::-1]]
            topic_repr_texts.append(" ".join(top_words))
            logger.info(f"[NMF] トピック{topic_idx}: {' '.join(top_words[:6])}")

        topic_vectors    = embedder.encode([f"query: {t}" for t in topic_repr_texts])
        interest_vectors = embedder.encode([f"query: {t}" for t in interest_texts])
        topic_interest_sim = cosine_similarity(topic_vectors, interest_vectors).max(axis=1)
        topic_noninterest  = 1.0 - topic_interest_sim

        row_sums       = doc_topic.sum(axis=1, keepdims=True)
        doc_topic_norm = doc_topic / (row_sums + 1e-8)
        nmf_scores     = doc_topic_norm @ topic_noninterest
        return (nmf_scores - nmf_scores.mean()).tolist()

    except Exception as e:
        logger.error(f"NMF処理エラー: {e}。NMFスコアを0で代替します。")
        return [0.0] * len(articles)


def filter_articles_by_similarity(
    articles: List[Any],
    embedder: SentenceTransformer,
    interest_texts: List[str],
    disinterest_texts: List[str]
) -> tuple[List[Dict], bool]:
    if not articles:
        return [], False

    try:
        interest_vecs    = embedder.encode([f"query: {t}" for t in interest_texts])
        disinterest_vecs = embedder.encode([f"query: {t}" for t in disinterest_texts])

        passage_texts = [
            f"passage: {strip_html(getattr(e, 'title', ''))} {strip_html(getattr(e, 'summary', getattr(e, 'description', '')))}"
            for e in articles
        ]
        logger.info(f"ベクトル化を一括実行中... ({len(articles)}件)")
        article_vecs = embedder.encode(passage_texts)

        diff_scores = []
        scored_articles = []
        for idx, entry in enumerate(articles):
            max_interest    = float(cosine_similarity([article_vecs[idx]], interest_vecs)[0].max())
            max_disinterest = float(cosine_similarity([article_vecs[idx]], disinterest_vecs)[0].max())
            diff = max_disinterest - max_interest
            diff_scores.append(diff)

            summary = getattr(entry, 'summary', getattr(entry, 'description', ''))
            if not summary or len(summary.strip()) < 10:
                content_obj = getattr(entry, 'content', [])
                if content_obj and content_obj[0].get('value'):
                    summary = content_obj[0].get('value', '')[:300]

            scored_articles.append({
                'entry': entry, 'sim': max_interest,
                'diff': diff, 'summary': summary
            })

        logger.info("NMFによる潜在トピックスコア算出中...")
        nmf_scores = compute_nmf_noninterest_scores(articles, embedder, interest_texts)

        diff_sig = _sigmoid(np.array(diff_scores))
        nmf_sig  = _sigmoid(np.array(nmf_scores))
        combined = ALPHA * diff_sig + BETA * nmf_sig

        mean_c    = float(combined.mean())
        std_c     = float(combined.std())
        threshold = mean_c + K_SIGMA * std_c
        logger.info(f"アンサンブルスコア: mean={mean_c:.3f} std={std_c:.3f} threshold={threshold:.3f} (K={K_SIGMA})")

        for i, item in enumerate(scored_articles):
            item['combined_score'] = float(combined[i])

        target_articles = [item for item in scored_articles if item['combined_score'] >= threshold]
        is_fallback = False

        if not target_articles:
            logger.warning("閾値超え記事が0件。スコア上位3件を強制抽出。")
            scored_articles.sort(key=lambda x: x['combined_score'], reverse=True)
            target_articles = scored_articles[:3]
            is_fallback = True

        target_articles.sort(key=lambda x: x['combined_score'], reverse=True)
        logger.info(f"抽出記事数: {len(target_articles)}件 (fallback={is_fallback})")
        return target_articles, is_fallback

    except Exception as e:
        logger.error(f"フィルタリング処理中に致命的なエラー: {e}", exc_info=True)
        return [], False


# [修正②] Grounding有効時は専用のリトライ設定（待機時間を大幅に延長）
def _make_retry_decorator():
    if USE_GROUNDING:
        # Grounding=True: AFCが10コール消費するため待機を長めに設定
        return retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=30, max=180))
    else:
        return retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=10, max=120))


def generate_ai_explanation(client: Any, original_title: str, summary: str) -> dict:
    prompt = f"""
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
<br>
【解説】
【何があったか】
(本文)
<br>
【知っておくべき背景】
(本文)
<br>
【何に影響するか】
(本文)
<br>
【あなたの興味との接点】
(本文)
<br>
=== 対象テキスト ===
【{original_title}】
{summary}
"""

    # [修正③] Groundingで429が続いた場合はGrounding無効でフォールバック
    @_make_retry_decorator()
    def _call(use_grounding: bool) -> Any:
        cfg = types.GenerateContentConfig(
            temperature=0.2,
            tools=[types.Tool(google_search=types.GoogleSearch())] if use_grounding else []
        )
        if use_grounding:
            logger.info("Grounding有効でリクエスト送信")
        return client.models.generate_content(
            model=GEMINI_MODEL_NAME, contents=prompt, config=cfg
        )

    try:
        response = _call(USE_GROUNDING)
    except Exception as e:
        if USE_GROUNDING and "429" in str(e):
            logger.warning("GroundingでRetryError(429)。Grounding無効でフォールバック再試行します。")
            try:
                response = _call(False)
            except Exception as e2:
                raise e2
        else:
            raise e

    result = {"title": original_title, "explanation": ""}
    if response.text:
        text = response.text.strip()
        title_match       = re.search(r'【タイトル】\s*(.*?)\s*【解説】', text, re.DOTALL)
        explanation_match = re.search(r'【解説】\s*(.*)', text, re.DOTALL)
        if title_match and explanation_match:
            result["title"]       = title_match.group(1).strip()
            result["explanation"] = explanation_match.group(1).strip()
        else:
            result["explanation"] = text
    else:
        result["explanation"] = (
            f"【何があったか】\n{summary}\n\n"
            f"【知っておくべき背景】\n（生成エラー）\n\n"
            f"【何に影響するか】\n（生成エラー）\n\n"
            f"【あなたの興味との接点】\n（生成エラー）"
        )
    return result


def main():
    try:
        gemini_key = os.environ.get("API_KEY1")
        hf_token   = os.environ.get("HF_TOKEN1")

        if not gemini_key:
            logger.critical("API_KEY1 が未設定")
            sys.exit(1)
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        logger.info(f"Grounding設定: USE_GROUNDING={USE_GROUNDING} / MAX_PROCESS_PER_RUN={MAX_PROCESS_PER_RUN}")

        logger.info("0. 処理済みURLキャッシュの読み込み")
        seen_links, last_run_articles = load_cache()

        logger.info("1. RSSフィードから記事取得")
        free_articles = fetch_free_articles(SOURCE_RSS_URLS, seen_links)
        logger.info(f"合計新規抽出数: {len(free_articles)}件")

        if not free_articles:
            logger.info("新規処理対象の記事がないため、キャッシュの更新・RSS再構築のみ行います。")
            target_articles = []
            is_fallback = False
        else:
            logger.info("2. モデルロードと3段階フィルタリング")
            try:
                embedder = SentenceTransformer('intfloat/multilingual-e5-small')
                target_articles, is_fallback = filter_articles_by_similarity(
                    free_articles, embedder, INTEREST_TEXTS, DISINTEREST_TEXTS
                )
            except Exception as e:
                logger.error(f"モデルロードに失敗: {e}")
                target_articles, is_fallback = [], False

        # [修正①] 処理記事数を上限で切り詰める
        if len(target_articles) > MAX_PROCESS_PER_RUN:
            logger.info(f"処理記事数を上限 {MAX_PROCESS_PER_RUN} 件に切り詰めます（抽出数: {len(target_articles)}件）")
            target_articles = target_articles[:MAX_PROCESS_PER_RUN]

        logger.info(f"処理対象: {len(target_articles)}件")

        logger.info("3. LLM再構築とフィード生成")
        client = genai.Client(api_key=gemini_key)
        current_run_articles = []

        for item in target_articles:
            entry          = item['entry']
            summary        = item['summary']
            original_title = getattr(entry, 'title', '')

            content_obj   = getattr(entry, 'content', [])
            original_html = content_obj[0].get('value', summary) if content_obj else summary

            try:
                ai_result      = generate_ai_explanation(client, original_title, summary)
                ai_title       = ai_result["title"]
                ai_explanation = ai_result["explanation"]
            except Exception as e:
                if "429" in str(e):
                    logger.error(f"APIレートリミット到達 (429): {e}")
                else:
                    logger.error(f"LLM解説生成に失敗: {e}")
                ai_title       = original_title
                ai_explanation = (
                    f"【何があったか】\n{summary}\n\n"
                    f"【知っておくべき背景】\nLLM生成エラーのため元の記事を参照してください。\n\n"
                    f"【何に影響するか】\n（生成エラー）\n\n"
                    f"【あなたの興味との接点】\n（生成エラー）"
                )

            raw_link  = getattr(entry, 'link', '').strip().split('#')[0]
            image_url = extract_image_url(entry, original_html)

            fallback_notice = "<p>※出力確保のための抽出記事</p>" if is_fallback else ""
            ai_exp_html     = re.sub(r'\n{2,}', '\n', ai_explanation).strip().replace('\n', '<br>')
            img_html        = f'<p><img src="{image_url}" style="max-width:100%; height:auto;" /></p>' if image_url else ""

            description_html = f"""
            <p style="color:gray; font-size: small;">
            ・元タイトル「{original_title}」<br>
            ・興味類似度: {item['sim']:.3f} / 差分スコア: {item['diff']:.3f} / 総合スコア: {item['combined_score']:.3f}
            </p>
            {fallback_notice}
            <h3>LLM解説</h3>
            {img_html}
            <p>{ai_exp_html}</p>
            <hr>
            <h3>元の記事</h3>
            {original_html}
            """

            feed_title     = entry.get('feed_title', '不明なソース')
            custom_summary = f"総合スコア: {item['combined_score']:.3f} | ソース: {feed_title}"

            pub_parsed = getattr(entry, 'published_parsed', None)
            if pub_parsed:
                dt           = datetime(*pub_parsed[:6])
                pub_date_iso = pytz.utc.localize(dt).astimezone(pytz.timezone('Asia/Tokyo')).isoformat()
            else:
                pub_date_iso = datetime.now(pytz.timezone('Asia/Tokyo')).isoformat()

            article_data = {
                "id":          raw_link,
                "title":       ai_title,
                "link":        raw_link,
                "description": custom_summary,
                "content":     description_html,
                "pubDate":     pub_date_iso,
                "enclosure":   image_url
            }
            current_run_articles.append(article_data)

            if raw_link not in seen_links:
                seen_links.append(raw_link)

            logger.info(f"処理完了: {ai_title[:30]}...")
            # [修正②] Grounding時は45秒待機（AFCコール分のレート消費を考慮）
            sleep_sec = 45 if USE_GROUNDING else 8
            time.sleep(sleep_sec)

        unique_articles   = {art["id"]: art for art in (current_run_articles + last_run_articles)}
        all_feed_articles = list(unique_articles.values())
        all_feed_articles.sort(key=lambda x: x["pubDate"], reverse=True)

        MAX_FEED_ITEMS    = 30
        all_feed_articles = all_feed_articles[:MAX_FEED_ITEMS]

        fg = FeedGenerator()
        fg.title('LLM再構築フィード')
        fg.link(href='https://github.com/', rel='alternate')
        fg.description('負例クラスタ差分・NMF・動的閾値による3段階フィルタで、興味外ニュースを抽出・LLM解説するカスタムフィード')
        fg.language('ja')

        for art in all_feed_articles:
            fe = fg.add_entry()
            fe.id(art["id"])
            fe.title(art["title"])
            fe.link(href=art["link"])
            fe.description(art["description"])
            fe.content(content=art["content"], type='html')
            fe.pubDate(datetime.fromisoformat(art["pubDate"]))
            if art.get("enclosure"):
                fe.enclosure(art["enclosure"], 0, get_mime_type(art["enclosure"]))

        save_cache(seen_links, all_feed_articles)

        fg.rss_file('rss.xml')
        logger.info(f"rss.xml 生成完了 (出力件数: {len(all_feed_articles)}件)")

        html_content = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM再構築フィード</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2rem; color: #333; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
        .rss-link {{ display: inline-block; background: #ee802f; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin-top: 1rem; }}
        .rss-link:hover {{ background: #c66a26; }}
        .badge {{ display: inline-block; background: #e8f4fd; color: #1a6ca8; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin: 2px; }}
    </style>
</head>
<body>
    <h1>LLM再構築フィード</h1>
    <p>フィルターバブルを打破するため、ユーザーの関心領域外のニュースをLLM（Gemini）が構造化して配信するカスタムフィードです。</p>
    <p>
        <span class="badge">① 負例クラスタ差分スコア</span>
        <span class="badge">② NMFトピックモデル</span>
        <span class="badge">③ 統計的動的閾値</span>
        による3段階フィルタリングを実装しています。
    </p>
    <p>最終更新: {datetime.now(pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')} (JST)</p>
    <h2>利用方法</h2>
    <p>お使いのRSSリーダー（Feedly, Inoreaderなど）に以下のリンクを登録してください。</p>
    <a href="rss.xml" class="rss-link">RSSフィード (rss.xml) を取得</a>
    <p style="margin-top: 3rem; font-size: 0.8rem; color: #666;">
        Powered by GitHub Actions & Gemini API
    </p>
</body>
</html>
"""
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info("index.html 生成完了")

    except Exception as e:
        logger.critical(f"予期せぬ致命的なエラーにより処理が中断されました: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
