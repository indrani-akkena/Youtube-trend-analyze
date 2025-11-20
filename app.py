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
from dotenv import load_dotenv
import os
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import requests
from io import BytesIO


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR STYLING ---
# (Your CSS remains the same)
st.markdown("""
<style>
    /* General Styles */
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
    /* Card Styles */
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
            /* On hover, change background to blue and text to white */
button[data-baseweb="tab"]:hover {
    background-color: #007BFF; /* Blue color on hover */
    color: white;
}
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #58A6FF;
    }
            p{
            color:black}
    /* Animation */
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

# --- API AND KEYWORD INPUT ---
load_dotenv()
API_KEY = os.getenv('API_KEY')
youtube = build("youtube", "v3", developerKey=API_KEY)

keywords_input = st.text_input(
    "🔑 Enter keyword(s) separated by commas (e.g. AI, Python)",
    "AI",
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
@st.cache(allow_output_mutation=True)
def load_model():
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    model = hub.load(model_url)
    return model

@st.cache
def get_coco_labels():
    """Downloads and parses the official COCO label map in a robust way."""
    url = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/mscoco_label_map.pbtxt"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        
        labels = {}
        current_id = None
        
        # Iterate through the lines to find ID and display_name pairs
        for line in response.text.splitlines():
            if 'id:' in line:
                current_id = int(line.split(':')[1].strip())
            elif 'display_name:' in line and current_id is not None:
                display_name = line.split(':')[1].strip().replace('"', '')
                labels[current_id] = display_name
                current_id = None # Reset for the next item block
        
        if not labels:
             st.error("Failed to parse COCO labels. Object detection may not work.")
        return labels
    except requests.exceptions.RequestException as e:
        st.error(f"Could not download COCO labels: {e}")
        return {} # Return empty dict on failure
detector = load_model()
COCO_LABELS = get_coco_labels()


# --- FUNCTIONS ---
def contains_fake_news_text(text):
    text_lower = text.lower()
    return any(word in text_lower for word in SUSPICIOUS_KEYWORDS)

def detect_objects(image_url):
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        image_np = np.array(image)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        detector_output = detector(input_tensor)
        
        detection_scores = detector_output['detection_scores'][0].numpy()
        detection_classes = detector_output['detection_classes'][0].numpy().astype(np.int64)
        
        detected_objects = []
        for i in range(len(detection_scores)):
            if detection_scores[i] > 0.5: # Confidence threshold
                class_id = detection_classes[i]
                class_name = COCO_LABELS.get(class_id, "Unknown")
                detected_objects.append(f"{class_name.capitalize()} ({detection_scores[i]:.0%})")
        return detected_objects
    except Exception as e:
        return [f"Error detecting objects: {e}"]

# (The rest of your functions: get_video_data, classify_trend, etc. remain the same)
def get_video_data(keyword, max_results=100):
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
                "fake_news": fake_news_flag
            })
        next_token = search.get("nextPageToken")
        if not next_token:
            break
    return pd.DataFrame(videos)

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
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=7)
    forecast = m.predict(future)
    return m, forecast

def format_views(n):
    return f"{n / 1_000_000:.1f}M" if n >= 1_000_000 else f"{n / 1_000:.1f}K" if n >= 1_000 else str(n)

def get_comments(video_id, max_comments=50):
    comments = []
    try:
        req = youtube.commentThreads().list(part="snippet", videoId=video_id,
                                            maxResults=max_comments, textFormat="plainText").execute()
        for item in req.get("items", []):
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
    except:
        pass
    return comments

def analyze_sentiment(comments):
    results = {"Positive": 0, "Neutral": 0, "Negative": 0}
    for c in comments:
        polarity = TextBlob(c).sentiment.polarity
        if polarity > 0.1: results["Positive"] += 1
        elif polarity < -0.1: results["Negative"] += 1
        else: results["Neutral"] += 1
    return results

def display_top_videos(df):
    st.subheader("🎬 Top 3 Recent Videos")
    top = df.sort_values("views", ascending=False).head(3)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(row["thumbnail"])
                
                # Object Detection
                with st.spinner("Detecting objects..."):
                    detected_objects = detect_objects(row["thumbnail"])
                if detected_objects:
                    st.write("🖼️ **Thumbnail Objects:** ", ", ".join(detected_objects))
                else:
                    st.info("No objects detected in thumbnail.")

                title_display = f"**<a href='https://www.youtube.com/watch?v={row['video_id']}' style='color:white;text-decoration:none;'>{row['title']}</a>**"
                if row["fake_news"]:
                    title_display += " <span style='color:red;'>⚠️ Potential Fake News</span>"
                st.markdown(title_display, unsafe_allow_html=True)
                st.caption(row["description"])
                st.write(f"🗓️ {row['published_at']} | 👁️ {format_views(row['views'])}")

                comments = get_comments(row["video_id"])
                if comments:
                    sentiments = analyze_sentiment(comments)
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pie(list(sentiments.values()), labels=list(sentiments.keys()),
                           colors=['#28a745', '#6c757d', '#dc3545'], autopct='%1.1f%%',
                           startangle=140, textprops={'color':"w"})
                    ax.axis('equal')
                    fig.patch.set_facecolor('#161B22')
                    st.pyplot(fig)
                else:
                    st.info("No comments to analyze.")
                st.markdown('</div>', unsafe_allow_html=True)

def recommend_similar_topics(df, current_keywords, top_n=5):
    if df.empty: return pd.DataFrame()
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
        if len(recommended) >= top_n: break
    return pd.DataFrame(recommended)

# --- APP LOGIC ---
# (Your main App Logic remains the same)
if keywords:
    for kw in keywords:
        st.markdown(f"--- \n ### 🔍 Results for: `{kw}`")
        with st.spinner(f"Fetching videos for '{kw}'..."):
            video_df = get_video_data(kw, max_results=100)

        if video_df.empty:
            st.warning("No videos found.")
            continue

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
                last_7_days = daily_views[daily_views["published_at"] >= (daily_views["published_at"].max() - pd.Timedelta(days=6))]
                fig, ax = plt.subplots()
                ax.plot(last_7_days["published_at"], last_7_days["views"], marker='o', color='#007BFF')
                ax.set_title("Total Daily Views (Past 7 Days)", color='w')
                ax.set_xlabel("Date", color='w'); ax.set_ylabel("Views", color='w')
                ax.grid(True, linestyle='--', alpha=0.3)
                fig.patch.set_facecolor('#161B22'); ax.set_facecolor('#0E1117')
                plt.xticks(rotation=45, color='w'); plt.yticks(color='w')
                st.pyplot(fig)

            with col2:
                result = forecast_views(daily_views)
                if result:
                    model, forecast = result
                    fig2 = model.plot(forecast)
                    fig2.gca().set_title("Prophet Forecast", color='w')
                    fig2.gca().set_xlabel("Date", color='w'); fig2.gca().set_ylabel("Views", color='w')
                    fig2.patch.set_facecolor('#161B22'); ax.set_facecolor('#0E1117')
                    plt.xticks(color='w'); plt.yticks(color='w')
                    st.pyplot(fig2)
                else:
                    st.warning("Not enough data to build a reliable forecast.")
        with tab3:
            st.subheader("🤝 Recommended Similar Videos")
            recs_df = recommend_similar_topics(video_df, [kw], top_n=5)
            if recs_df.empty:
                st.info("No recommendations available.")
            else:
                for _, rec in recs_df.iterrows():
                    st.markdown('<div class="card" style="margin-bottom: 10px;">', unsafe_allow_html=True)
                    c1, c2 = st.columns([1, 4])
                    with c1: 
                        st.image(rec["thumbnail"])
                        # Object Detection
                        with st.spinner("Detecting objects..."):
                           detected_objects = detect_objects(rec["thumbnail"])
                        if detected_objects:
                           st.write("🖼️ **Objects:** ", ", ".join(detected_objects))
                        else:
                           st.info("No objects detected.")
                    with c2:
                        st.markdown(f"**<a href='https://www.youtube.com/watch?v={rec['video_id']}' style='color:white;text-decoration:none;'>{rec['title']}</a>**", unsafe_allow_html=True)
                        st.caption(rec["description"])
                        st.write(f"👁️ Views: {format_views(rec['views'])}")
                    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👋 Welcome! Please enter a keyword above to begin your analysis.")