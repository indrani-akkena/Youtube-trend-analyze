import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Trends Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎥"
)

# --- CUSTOM CSS FOR IMPROVED STYLING ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* General Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 12px;
        padding: 12px;
        font-size: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #58A6FF;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Card Styles */
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        margin-right: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    button[data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 700;
    }
    
    /* Text */
    p, span, div {
        color: #FFFFFF;
    }
    
    .card p, .card span, .card div {
        color: #333333;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #667eea;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Warning/Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main .block-container {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png", width=80)
with col2:
    st.title("🎥 YouTube Trends Analyzer")
    st.markdown("### 📊 Forecasts • 💬 Sentiment Analysis • 🔍 Fake News Detection • 🎯 Recommendations")

# --- API KEY INPUT (SECURE) ---
st.sidebar.title("🔐 Configuration")
API_KEY = st.sidebar.text_input(
    "YouTube API Key",
    type="password",
    help="Enter your YouTube Data API v3 key. Get one at https://console.cloud.google.com/"
)

if not API_KEY:
    st.warning("⚠️ Please enter your YouTube API Key in the sidebar to begin.")
    st.info("📌 **How to get an API Key:**\n1. Go to [Google Cloud Console](https://console.cloud.google.com/)\n2. Create a new project\n3. Enable YouTube Data API v3\n4. Create credentials (API Key)\n5. Copy and paste it in the sidebar")
    st.stop()

try:
    youtube = build("youtube", "v3", developerKey=API_KEY)
except Exception as e:
    st.error(f"❌ Invalid API Key or connection error: {str(e)}")
    st.stop()

# --- KEYWORD INPUT ---
keywords_input = st.text_input(
    "🔑 Enter keyword(s) separated by commas",
    "AI, Machine Learning",
    help="Enter multiple keywords to analyze them side-by-side"
)
keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]

# --- CONSTANTS ---
SUSPICIOUS_KEYWORDS = [
    "fake", "hoax", "false", "misleading", "scam", "untrue", "debunked", "conspiracy",
    "fraud", "clickbait", "bogus", "fabricated", "manipulated", "falsified", "counterfeit",
    "phony", "deceptive", "disinformation", "propaganda", "myth", "misinformation",
    "lies", "exposed", "disproved", "rumor", "rumour", "deceit", "scandal", "alleged",
    "fake news", "false claim", "deepfake", "fabrication", "sensationalism"
]

# --- OBJECT DETECTION MODEL & LABELS ---
@st.cache_resource
def load_model():
    try:
        model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
        model = hub.load(model_url)
        return model
    except Exception as e:
        st.warning(f"Object detection model loading failed: {e}")
        return None

@st.cache_data
def get_coco_labels():
    """Downloads and parses the official COCO label map."""
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
        st.warning(f"Could not download COCO labels: {e}")
        return {}

detector = load_model()
COCO_LABELS = get_coco_labels()

# --- FUNCTIONS ---
def contains_fake_news_text(text):
    """Check if text contains suspicious keywords."""
    text_lower = text.lower()
    return any(word in text_lower for word in SUSPICIOUS_KEYWORDS)

def detect_objects(image_url):
    """Detect objects in thumbnail image."""
    if not detector or not COCO_LABELS:
        return []
    
    try:
        response = requests.get(image_url, timeout=5)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detector_output = detector(input_tensor)
        
        detection_scores = detector_output['detection_scores'][0].numpy()
        detection_classes = detector_output['detection_classes'][0].numpy().astype(np.int64)
        
        detected_objects = []
        for i in range(min(5, len(detection_scores))):
            if detection_scores[i] > 0.5:
                class_id = detection_classes[i]
                class_name = COCO_LABELS.get(class_id, "Unknown")
                detected_objects.append(f"{class_name.capitalize()} ({detection_scores[i]:.0%})")
        return detected_objects
    except Exception as e:
        return []

def get_video_data(keyword, max_results=100):
    """Fetch video data from YouTube API."""
    videos = []
    next_token = None
    
    try:
        while len(videos) < max_results:
            search = youtube.search().list(
                q=keyword,
                part="id",
                type="video",
                maxResults=min(50, max_results - len(videos)),
                pageToken=next_token,
                order="relevance"
            ).execute()
            
            ids = [item["id"]["videoId"] for item in search.get("items", [])]
            if not ids:
                break
            
            video_details = youtube.videos().list(
                part="statistics,snippet",
                id=",".join(ids)
            ).execute()
            
            for item in video_details["items"]:
                stats = item["statistics"]
                snippet = item["snippet"]
                title = snippet["title"]
                description = snippet.get("description", "")[:150]
                fake_news_flag = contains_fake_news_text(title) or contains_fake_news_text(description)
                
                videos.append({
                    "video_id": item["id"],
                    "title": title,
                    "description": description,
                    "published_at": pd.to_datetime(snippet["publishedAt"]).date(),
                    "thumbnail": snippet["thumbnails"]["high"]["url"],
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "comments": int(stats.get("commentCount", 0)),
                    "fake_news": fake_news_flag
                })
            
            next_token = search.get("nextPageToken")
            if not next_token:
                break
                
    except Exception as e:
        st.error(f"Error fetching videos: {str(e)}")
    
    return pd.DataFrame(videos)

def classify_trend(series):
    """Classify trend as Rising, Declining, or Stable."""
    if len(series) < 7:
        return "Not enough data"
    
    y = series[-7:].values.reshape(-1, 1)
    X = np.arange(len(y)).reshape(-1, 1)
    
    try:
        slope = LinearRegression().fit(X, y).coef_[0][0]
        if slope > 1000:
            return "📈 Rising"
        elif slope < -1000:
            return "📉 Declining"
        else:
            return "➖ Stable"
    except:
        return "➖ Stable"

def forecast_views(df):
    """Forecast future views using Prophet."""
    df = df.rename(columns={"published_at": "ds", "views": "y"})
    df = df[df["y"] > 0]
    
    if len(df) < 14:
        return None
    
    try:
        m = Prophet(daily_seasonality=True, yearly_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        return m, forecast
    except Exception as e:
        st.warning(f"Forecasting failed: {str(e)}")
        return None

def format_views(n):
    """Format view count in readable format."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    else:
        return str(n)

def get_comments(video_id, max_comments=50):
    """Fetch comments for a video."""
    comments = []
    try:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText"
        ).execute()
        
        for item in req.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    except:
        pass
    
    return comments

def analyze_sentiment(comments):
    """Analyze sentiment of comments."""
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
    """Display top 3 videos with enhanced styling."""
    st.subheader("🏆 Top 3 Most Viewed Videos")
    
    top = df.sort_values("views", ascending=False).head(3)
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(row["thumbnail"], use_container_width=True)
            
            # Object Detection
            detected_objects = detect_objects(row["thumbnail"])
            if detected_objects:
                st.caption("🖼️ **Detected:** " + ", ".join(detected_objects[:3]))
            
            # Title with fake news warning
            title_display = f"**[{row['title'][:60]}...]({f'https://www.youtube.com/watch?v={row["video_id"]}'})**"
            if row["fake_news"]:
                st.markdown(f"⚠️ {title_display}", unsafe_allow_html=True)
                st.caption("🚨 Potential misinformation detected")
            else:
                st.markdown(title_display, unsafe_allow_html=True)
            
            st.caption(row["description"][:100] + "...")
            
            # Metrics
            col_a, col_b = st.columns(2)
            col_a.metric("👁️ Views", format_views(row['views']))
            col_b.metric("👍 Likes", format_views(row['likes']))
            
            # Sentiment Analysis
            with st.expander("💬 Sentiment Analysis"):
                comments = get_comments(row["video_id"])
                if comments:
                    sentiments = analyze_sentiment(comments)
                    
                    fig, ax = plt.subplots(figsize=(4, 4))
                    colors = ['#28a745', '#6c757d', '#dc3545']
                    ax.pie(
                        list(sentiments.values()),
                        labels=list(sentiments.keys()),
                        colors=colors,
                        autopct='%1.1f%%',
                        startangle=140,
                        textprops={'fontsize': 10, 'color': 'black'}
                    )
                    ax.axis('equal')
                    fig.patch.set_facecolor('white')
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("No comments available")
            
            st.markdown('</div>', unsafe_allow_html=True)

def recommend_similar_topics(df, current_keywords, top_n=5):
    """Recommend similar videos based on content."""
    if df.empty:
        return pd.DataFrame()
    
    df["text"] = df["title"] + " " + df["description"]
    
    try:
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
    except Exception as e:
        st.warning(f"Recommendation generation failed: {str(e)}")
        return pd.DataFrame()

# --- MAIN APP LOGIC ---
if keywords:
    for kw in keywords:
        st.markdown(f"---")
        st.markdown(f"## 🔍 Analysis for: **{kw}**")
        
        with st.spinner(f"🔎 Fetching videos for '{kw}'..."):
            video_df = get_video_data(kw, max_results=50)
        
        if video_df.empty:
            st.warning(f"No videos found for '{kw}'")
            continue
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📹 Total Videos", len(video_df))
        col2.metric("👁️ Total Views", format_views(video_df['views'].sum()))
        col3.metric("⚠️ Flagged Content", video_df['fake_news'].sum())
        col4.metric("📅 Date Range", f"{(video_df['published_at'].max() - video_df['published_at'].min()).days}d")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["🏆 Top Videos", "📈 Trends & Forecast", "🎯 Recommendations"])
        
        with tab1:
            display_top_videos(video_df)
        
        with tab2:
            daily_views = video_df.groupby("published_at")["views"].sum().reset_index()
            daily_views = daily_views.sort_values("published_at", ascending=True)
            
            trend = classify_trend(daily_views["views"])
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 16px; text-align: center; 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-bottom: 30px;'>
                <h3 style='color: white; margin: 0;'>📊 Trend Analysis</h3>
                <p style='font-size: 28px; font-weight: bold; color: white; margin: 10px 0 0 0;'>{trend}</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Daily View Trends (Last 7 Days)")
                last_7_days = daily_views[
                    daily_views["published_at"] >= (daily_views["published_at"].max() - pd.Timedelta(days=6))
                ]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(
                    last_7_days["published_at"],
                    last_7_days["views"],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    color='#667eea'
                )
                ax.fill_between(
                    last_7_days["published_at"],
                    last_7_days["views"],
                    alpha=0.3,
                    color='#667eea'
                )
                ax.set_title("Total Daily Views", fontsize=14, fontweight='bold')
                ax.set_xlabel("Date", fontsize=12)
                ax.set_ylabel("Views", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.3)
                fig.patch.set_facecolor('white')
                ax.set_facecolor('#f8f9fa')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col2:
                st.subheader("🔮 7-Day Forecast")
                result = forecast_views(daily_views)
                
                if result:
                    model, forecast = result
                    fig2 = model.plot(forecast)
                    fig2.gca().set_title("Prophet Forecast", fontsize=14, fontweight='bold')
                    fig2.gca().set_xlabel("Date", fontsize=12)
                    fig2.gca().set_ylabel("Views", fontsize=12)
                    fig2.patch.set_facecolor('white')
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                else:
                    st.warning("⚠️ Insufficient data for reliable forecast (need 14+ days)")
        
        with tab3:
            st.subheader("🎯 Recommended Similar Videos")
            recs_df = recommend_similar_topics(video_df, [kw], top_n=5)
            
            if recs_df.empty:
                st.info("No recommendations available at this time.")
            else:
                for _, rec in recs_df.iterrows():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    c1, c2 = st.columns([1, 3])
                    
                    with c1:
                        st.image(rec["thumbnail"], use_container_width=True)
                        detected_objects = detect_objects(rec["thumbnail"])
                        if detected_objects:
                            st.caption("🖼️ " + ", ".join(detected_objects[:2]))
                    
                    with c2:
                        st.markdown(f"**[{rec['title'][:80]}...]({f'https://www.youtube.com/watch?v={rec["video_id"]}'})**")
                        st.caption(rec["description"][:120] + "...")
                        st.metric("👁️ Views", format_views(rec['views']))
                    
                    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👋 Welcome! Please enter keywords above to begin your analysis.")
    st.markdown("""
    ### 🚀 Features:
    - 📊 **Trend Analysis**: Identify rising, declining, or stable trends
    - 🔮 **AI Forecasting**: Predict future views with Prophet
    - 💬 **Sentiment Analysis**: Understand audience reactions
    - 🚨 **Fake News Detection**: Flag potentially misleading content
    - 🖼️ **Object Detection**: Analyze video thumbnails
    - 🎯 **Smart Recommendations**: Discover similar content
    """)