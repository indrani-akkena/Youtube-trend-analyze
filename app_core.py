import os
from functools import lru_cache
import pandas as pd
import numpy as np
from prophet import Prophet
from googleapiclient.discovery import build
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from textblob import TextBlob
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import requests
from io import BytesIO


# --- CONSTANTS ---
SUSPICIOUS_KEYWORDS = [
    "fake", "hoax", "false", "misleading", "scam", "untrue", "debunked", "conspiracy",
    "fraud", "clickbait", "bogus", "fabricated", "manipulated", "falsified", "counterfeit",
    "phony", "deceptive", "disinformation", "propaganda", "myth", "misinformation",
    "lies", "exposed", "disproved", "rumor", "rumour", "deceit", "scandal", "alleged",
    "fake news", "false claim", "deepfake", "fabrication", "sensationalism"
]


@lru_cache(maxsize=1)
def load_detector(model_url: str = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"):
    """Load object detector from TF Hub. Cached to avoid reloading repeatedly."""
    return hub.load(model_url)


def get_coco_labels():
    """Download and parse COCO label map. Returns dict id->name.

    This function is intentionally free of Streamlit calls so it can be
    used in non-UI contexts (and tested easily).
    """
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        labels = {}
        current_id = None
        for line in response.text.splitlines():
            if 'id:' in line:
                current_id = int(line.split(':')[1].strip())
            elif 'display_name:' in line and current_id is not None:
                display_name = line.split(':')[1].strip().replace('"', '')
                labels[current_id] = display_name
                current_id = None
        return labels
    except requests.exceptions.RequestException:
        return {}


def contains_fake_news_text(text: str) -> bool:
    text_lower = (text or "").lower()
    return any(word in text_lower for word in SUSPICIOUS_KEYWORDS)


def detect_objects(image_url: str, detector=None, labels=None, threshold: float = 0.5):
    """Download image from URL and run object detection. Returns list of detected names.

    If `detector` or `labels` are not provided this function will load them.
    """
    try:
        if detector is None:
            detector = load_detector()
        if labels is None:
            labels = get_coco_labels()

        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detector_output = detector(input_tensor)
        detection_scores = detector_output['detection_scores'][0].numpy()
        detection_classes = detector_output['detection_classes'][0].numpy().astype(np.int64)

        detected_objects = []
        for i in range(len(detection_scores)):
            if detection_scores[i] > threshold:
                class_id = int(detection_classes[i])
                class_name = labels.get(class_id, "Unknown")
                detected_objects.append(f"{class_name.capitalize()} ({detection_scores[i]:.0%})")
        return detected_objects
    except Exception as e:
        return [f"Error detecting objects: {e}"]


def get_video_data(keyword: str, max_results: int = 100):
    """Fetch videos for a keyword using YouTube Data API and return a DataFrame."""
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        raise RuntimeError("API_KEY not set in environment. Set it in .env or env vars.")

    youtube = build("youtube", "v3", developerKey=API_KEY)
    videos = []
    next_token = None
    while len(videos) < max_results:
        search = youtube.search().list(q=keyword, part="id", type="video",
                                       maxResults=min(50, max_results - len(videos)),
                                       pageToken=next_token).execute()
        ids = [item["id"]["videoId"] for item in search.get("items", [])]
        if not ids:
            break
        video_details = youtube.videos().list(part="statistics,snippet", id=",".join(ids)).execute()
        for item in video_details.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            description = snippet.get("description", "")[:150]
            fake_news_flag = contains_fake_news_text(title) or contains_fake_news_text(description)
            videos.append({
                "video_id": item.get("id"),
                "title": title,
                "description": description,
                "published_at": pd.to_datetime(snippet.get("publishedAt")).date() if snippet.get("publishedAt") else None,
                "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                "views": int(stats.get("viewCount", 0)),
                "fake_news": fake_news_flag
            })
        next_token = search.get("nextPageToken")
        if not next_token:
            break
    return pd.DataFrame(videos)


def classify_trend(series: pd.Series) -> str:
    if len(series) < 7:
        return "Not enough data"
    y = series[-7:].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    slope = LinearRegression().fit(X, y).coef_[0][0]
    return "📈 Rising" if slope > 1000 else "📉 Declining" if slope < -1000 else "➖ Stable"


def forecast_views(df: pd.DataFrame):
    df = df.rename(columns={"published_at": "ds", "views": "y"})
    df = df[df["y"] > 0]
    if len(df) < 14:
        return None
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return m, forecast


def format_views(n: int) -> str:
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1_000:.1f}K" if n >= 1_000 else str(n)


def get_comments(video_id: str, max_comments: int = 50):
    from dotenv import load_dotenv
    load_dotenv()
    API_KEY = os.getenv('API_KEY')
    if not API_KEY:
        return []
    youtube = build("youtube", "v3", developerKey=API_KEY)
    comments = []
    try:
        req = youtube.commentThreads().list(part="snippet", videoId=video_id,
                                            maxResults=max_comments, textFormat="plainText").execute()
        for item in req.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    except Exception:
        pass
    return comments


def analyze_sentiment(comments):
    results = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for c in comments:
        polarity = TextBlob(c).sentiment.polarity
        if polarity > 0.1:
            results["Positive"] += 1
        elif polarity < -0.1:
            results["Negative"] += 1
        else:
            results["Neutral"] += 1
    return results


def recommend_similar_topics(df: pd.DataFrame, current_keywords, top_n: int = 5):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["text"] = df["title"] + " " + df["description"]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    input_vec = vectorizer.transform([" ".join(current_keywords)])
    similarities = cosine_similarity(input_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1]
    recommended = []
    for idx in top_indices:
        title = df.iloc[idx]["title"]
        if not any(kw.lower() in title.lower() for kw in current_keywords):
            recommended.append({
                "title": title,
                "description": df.iloc[idx]["description"],
                "video_id": df.iloc[idx]["video_id"],
                "views": df.iloc[idx]["views"],
                "thumbnail": df.iloc[idx]["thumbnail"],
                "similarity": similarities[idx]
            })
        if len(recommended) >= top_n:
            break
    return pd.DataFrame(recommended)
