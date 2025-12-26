import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from dotenv import load_dotenv
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import requests
from io import BytesIO
import json


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    body {
        background-color: #ee6c4d;
        color: #FAFAFA;
    }
    .stApp {
        background: url("https://www.transparenttextures.com/patterns/dark-matter.png");
    }
    .stTextInput > div > div > input {
        background-color: #1C2128;
        color: #FAFAFA;
        border-radius: 10px;
    }
    .stButton > button {
        background-color: #007BFF;
        color: white;
        border-radius: 10px;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        transform: scale(1.05);
    }
    .card {
        background-color: #161B22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363D;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    button[data-baseweb="tab"]:hover {
        background-color: #007BFF;
        color: white;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #58A6FF;
    }
    p {
        color: black;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .reportview-container .main .block-container {
        animation: fadeIn 1s;
    }
</style>
""", unsafe_allow_html=True)


# --- HEADER ---
st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png", width=100)
st.title("YouTube Trends Analyzer")
st.markdown("### Forecasts, Sentiment Analysis, Fake News Detection & Recommendations")

# --- LOAD AND VALIDATE API KEY ---
load_dotenv()
API_KEY = os.getenv('API_KEY')

if not API_KEY:
    st.error("❌ **API_KEY not found in environment variables!**")
    st.markdown("""
    ### How to fix:
    1. Create a `.env` file in your project root
    2. Add this line: `API_KEY=your_youtube_api_key_here`
    3. Get your API key from: [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
    4. Enable **YouTube Data API v3** in your Google Cloud project
    
    ### For Streamlit Cloud:
    1. Go to your app settings
    2. Click "Secrets" in the sidebar
    3. Add: `API_KEY = "your_key_here"`
    """)
    st.stop()

# Initialize YouTube API client
youtube = None
api_working = False

try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
    # Test the API with minimal quota usage
    test_request = youtube.search().list(q="test", part="id", maxResults=1).execute()
    api_working = True
    st.success("✅ YouTube API connected successfully!")
except HttpError as e:
    error_content = e.content.decode('utf-8') if hasattr(e, 'content') else str(e)
    st.error("❌ **YouTube API Error**")
    
    # Parse error for specific issues
    if "quotaExceeded" in error_content or "quota" in error_content.lower():
        st.warning("""
        ### ⚠️ API Quota Exceeded
        
        YouTube API has daily limits:
        - **Free tier**: 10,000 units/day
        - **Each search**: ~100 units
        - **Each video details request**: ~1 unit
        
        **Solutions:**
        1. Wait until tomorrow (quota resets at midnight Pacific Time)
        2. Request a quota increase in Google Cloud Console
        3. Reduce `max_results` parameter in the code
        """)
    elif "API_NOT_ENABLED" in error_content or "accessNotConfigured" in error_content:
        st.warning("""
        ### ⚠️ YouTube Data API v3 Not Enabled
        
        **Fix this by:**
        1. Go to: [Google Cloud Console - API Library](https://console.cloud.google.com/apis/library/youtube.googleapis.com)
        2. Select your project
        3. Click **"Enable API"**
        4. Wait 2-3 minutes for changes to propagate
        5. Restart this app
        """)
    elif "invalid" in error_content.lower() or "API key not valid" in error_content:
        st.warning("""
        ### ⚠️ Invalid API Key
        
        **Check:**
        1. API key is copied correctly (no extra spaces)
        2. API key restrictions don't block YouTube Data API
        3. Generate a new key if needed: [Credentials Page](https://console.cloud.google.com/apis/credentials)
        """)
    else:
        st.error(f"**Error details:** {error_content[:500]}")
    
    st.info("💡 **Quick fix:** The API key in your `.env` file may be invalid or restricted.")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error: {str(e)}")
    st.stop()


# --- CONSTANTS ---
SUSPICIOUS_KEYWORDS = [
    "fake", "hoax", "false", "misleading", "scam", "untrue", "debunked", "conspiracy",
    "fraud", "clickbait", "bogus", "fabricated", "manipulated", "falsified", "counterfeit",
    "phony", "deceptive", "disinformation", "propaganda", "myth", "misinformation",
    "lies", "exposed", "disproved", "rumor", "rumour", "deceit", "scandal", "alleged",
    "fake news", "false claim", "deepfake", "fabrication", "sensationalism"
]


# --- LOAD MODELS ---
@st.cache_resource
def load_detector():
    try:
        model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        return hub.load(model_url)
    except Exception as e:
        st.warning(f"Could not load object detection model: {e}")
        return None

@st.cache_data
def get_coco_labels():
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
    try:
        response = requests.get(url, timeout=10)
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
    except Exception as e:
        st.warning(f"Could not load COCO labels: {e}")
        return {}

detector = load_detector()
COCO_LABELS = get_coco_labels()


# --- HELPER FUNCTIONS ---
def contains_fake_news_text(text):
    text_lower = (text or "").lower()
    return any(word in text_lower for word in SUSPICIOUS_KEYWORDS)


def detect_objects(image_url):
    if not detector or not COCO_LABELS:
        return []
    
    try:
        response = requests.get(image_url, timeout=5)
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
            if detection_scores[i] > 0.5:
                class_id = int(detection_classes[i])
                class_name = COCO_LABELS.get(class_id, "Unknown")
                detected_objects.append(f"{class_name.capitalize()} ({detection_scores[i]:.0%})")
        
        return detected_objects
    except Exception:
        return []


def get_video_data(keyword, max_results=50):
    """Fetch video data with comprehensive error handling."""
    videos = []
    next_token = None
    attempts = 0
    max_attempts = 3
    
    try:
        while len(videos) < max_results and attempts < max_attempts:
            attempts += 1
            
            try:
                search_response = youtube.search().list(
                    q=keyword,
                    part="id",
                    type="video",
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_token,
                    order="viewCount",
                    relevanceLanguage="en"
                ).execute()
            except HttpError as e:
                error_content = e.content.decode('utf-8') if hasattr(e, 'content') else str(e)
                
                if "quotaExceeded" in error_content:
                    st.error("⚠️ **Quota exceeded** - Try again tomorrow or reduce search results")
                    return pd.DataFrame()
                elif "invalidParameter" in error_content:
                    st.error(f"⚠️ Invalid search parameter for keyword: '{keyword}'")
                    return pd.DataFrame()
                else:
                    st.error(f"⚠️ API Error: {error_content[:200]}")
                    return pd.DataFrame()
            
            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
            
            if not video_ids:
                break
            
            try:
                video_response = youtube.videos().list(
                    part="statistics,snippet",
                    id=",".join(video_ids)
                ).execute()
            except HttpError as e:
                st.warning(f"Could not fetch details for some videos: {str(e)[:100]}")
                break
            
            for item in video_response.get("items", []):
                stats = item.get("statistics", {})
                snippet = item.get("snippet", {})
                
                title = snippet.get("title", "")
                description = snippet.get("description", "")[:150]
                
                videos.append({
                    "video_id": item.get("id", ""),
                    "title": title,
                    "description": description,
                    "published_at": pd.to_datetime(snippet.get("publishedAt")).date() if snippet.get("publishedAt") else pd.Timestamp.now().date(),
                    "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    "views": int(stats.get("viewCount", 0)),
                    "fake_news": contains_fake_news_text(title) or contains_fake_news_text(description)
                })
            
            next_token = search_response.get("nextPageToken")
            if not next_token:
                break
        
        return pd.DataFrame(videos)
    
    except Exception as e:
        st.error(f"⚠️ Unexpected error fetching videos: {str(e)}")
        return pd.DataFrame()


def classify_trend(series):
    if len(series) < 7:
        return "Not enough data"
    y = series[-7:].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    slope = LinearRegression().fit(X, y).coef_[0][0]
    return "📈 Rising" if slope > 1000 else "📉 Declining" if slope < -1000 else "➖ Stable"


def forecast_views(df):
    df = df.rename(columns={"published_at": "ds", "views": "y"})
    df = df[df["y"] > 0]
    if len(df) < 14:
        return None
    try:
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        return m, forecast
    except Exception:
        return None


def format_views(n):
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1_000:.1f}K" if n >= 1_000 else str(n)


def get_comments(video_id, max_comments=50):
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText"
        ).execute()
        
        comments = []
        for item in req.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)
        
        return comments
    except:
        return []


def analyze_sentiment(comments):
    results = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for c in comments:
        try:
            polarity = TextBlob(c).sentiment.polarity
            if polarity > 0.1:
                results["Positive"] += 1
            elif polarity < -0.1:
                results["Negative"] += 1
            else:
                results["Neutral"] += 1
        except:
            results["Neutral"] += 1
    return results


def display_top_videos(df):
    st.subheader("🎬 Top 3 Recent Videos")
    top = df.sort_values("views", ascending=False).head(3)
    
    if top.empty:
        st.info("No videos to display")
        return
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            if row["thumbnail"]:
                st.image(row["thumbnail"])
                
                # Object detection (optional)
                if detector and COCO_LABELS:
                    detected = detect_objects(row["thumbnail"])
                    if detected:
                        st.caption("🖼️ " + ", ".join(detected[:3]))
            
            # Title with link
            title_html = f"**<a href='https://www.youtube.com/watch?v={row['video_id']}' target='_blank' style='color:white;text-decoration:none;'>{row['title']}</a>**"
            if row["fake_news"]:
                title_html += " <span style='color:red;'>⚠️</span>"
            st.markdown(title_html, unsafe_allow_html=True)
            
            st.caption(row["description"])
            st.write(f"🗓️ {row['published_at']} | 👁️ {format_views(row['views'])}")
            
            # Sentiment analysis
            comments = get_comments(row["video_id"])
            if comments:
                sentiments = analyze_sentiment(comments)
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.pie(
                    list(sentiments.values()),
                    labels=list(sentiments.keys()),
                    colors=['#28a745', '#6c757d', '#dc3545'],
                    autopct='%1.1f%%',
                    startangle=140,
                    textprops={'color': "w"}
                )
                ax.axis('equal')
                fig.patch.set_facecolor('#161B22')
                st.pyplot(fig)
            
            st.markdown('</div>', unsafe_allow_html=True)


def recommend_similar_topics(df, current_keywords, top_n=5):
    if df.empty:
        return pd.DataFrame()
    
    try:
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
    except Exception:
        return pd.DataFrame()


# --- USER INPUT ---
keywords_input = st.text_input(
    "🔑 Enter keyword(s) separated by commas (e.g., AI, Python)",
    "AI",
    help="Enter multiple keywords to analyze them side-by-side"
)
keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]


# --- MAIN APPLICATION ---
if keywords and api_working:
    for kw in keywords:
        st.markdown(f"---\n### 🔍 Results for: `{kw}`")
        
        with st.spinner(f"Fetching videos for '{kw}'..."):
            video_df = get_video_data(kw, max_results=50)  # Reduced to save quota
        
        if video_df.empty:
            st.warning(f"No videos found for '{kw}'. This might be due to API limits or search issues.")
            continue
        
        st.success(f"Found {len(video_df)} videos!")
        
        tab1, tab2, tab3 = st.tabs(["🏆 Top Videos", "📈 Trends & Forecast", "🤝 Recommendations"])
        
        with tab1:
            display_top_videos(video_df)
        
        with tab2:
            daily_views = video_df.groupby("published_at")["views"].sum().reset_index()
            daily_views = daily_views.sort_values("published_at", ascending=True)
            
            trend = classify_trend(daily_views["views"])
            st.markdown(f"""
            <div style='background-color:#161B22; padding: 15px; border-radius: 10px; text-align: center; border: 1px solid #30363D;'>
                <h3 style='color:#58A6FF;'>Trend Analysis</h3>
                <p style='font-size: 24px; font-weight: bold;'>{trend}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 Daily View Trends & Forecast")
            col1, col2 = st.columns(2)
            
            with col1:
                if len(daily_views) >= 7:
                    last_7 = daily_views.tail(7)
                    fig, ax = plt.subplots()
                    ax.plot(last_7["published_at"], last_7["views"], marker='o', color='#007BFF')
                    ax.set_title("Daily Views (Last 7 Days)", color='w')
                    ax.set_xlabel("Date", color='w')
                    ax.set_ylabel("Views", color='w')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    fig.patch.set_facecolor('#161B22')
                    ax.set_facecolor('#0E1117')
                    plt.xticks(rotation=45, color='w')
                    plt.yticks(color='w')
                    st.pyplot(fig)
                else:
                    st.info("Not enough data for trend visualization")
            
            with col2:
                result = forecast_views(daily_views)
                if result:
                    model, forecast = result
                    fig2 = model.plot(forecast)
                    fig2.gca().set_title("7-Day Forecast", color='w')
                    fig2.gca().set_xlabel("Date", color='w')
                    fig2.gca().set_ylabel("Views", color='w')
                    fig2.patch.set_facecolor('#161B22')
                    fig2.gca().set_facecolor('#0E1117')
                    plt.xticks(color='w')
                    plt.yticks(color='w')
                    st.pyplot(fig2)
                else:
                    st.warning("Not enough data for forecast (need 14+ data points)")
        
        with tab3:
            st.subheader("🤝 Recommended Similar Videos")
            recs_df = recommend_similar_topics(video_df, [kw], top_n=5)
            
            if recs_df.empty:
                st.info("No recommendations available")
            else:
                for _, rec in recs_df.iterrows():
                    st.markdown('<div class="card" style="margin-bottom: 10px;">', unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 4])
                    
                    with c1:
                        if rec["thumbnail"]:
                            st.image(rec["thumbnail"])
                    
                    with c2:
                        st.markdown(
                            f"**<a href='https://www.youtube.com/watch?v={rec['video_id']}' target='_blank' style='color:white;text-decoration:none;'>{rec['title']}</a>**",
                            unsafe_allow_html=True
                        )
                        st.caption(rec["description"])
                        st.write(f"👁️ {format_views(rec['views'])}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

elif not api_working:
    st.error("❌ Cannot proceed - API not working. Please fix the API key issue above.")
else:
    st.info("👋 Welcome! Enter a keyword above to begin your analysis.")
