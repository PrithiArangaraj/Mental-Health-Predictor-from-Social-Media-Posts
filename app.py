
import streamlit as st
import pandas as pd
import re
import requests
import praw
import random
import os
import torch
import snscrape.modules.twitter as sntwittercu
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import pipeline
from weasyprint import HTML
import base64
from io import BytesIO
import tweepy

# ------------------------------
# 🔍 Models
# ------------------------------
from transformers import pipeline
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
mbti_model = pipeline("text-classification", model="Shunian/mbti-classification-xlnet-base-cased-augment")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model to empty state with the appropriate device
mbti_model.model.to_empty(device=device)




# Map model's labels to MBTI types
label_to_mbti = {
    "LABEL_0": "INTJ",
    "LABEL_1": "ENTP",
    "LABEL_2": "INFJ",
    "LABEL_3": "ISFP",
    "LABEL_4": "ESTP",
    "LABEL_5": "ISTJ",
    "LABEL_6": "ENFP",
    "LABEL_7": "ESFJ"
}

mbti_descriptions = {
    "INTJ": "Imaginative and strategic thinkers, with a plan for everything. They are independent, determined, and value intelligence and competence.",
    "ENTP": "Innovative, curious, and clever. ENTPs love to challenge ideas, debate, and explore new possibilities. They are quick-witted and enjoy intellectual discussions.",
    "INFJ": "Insightful, idealistic, and compassionate. INFJs are driven by their values and a desire to help others. They are creative and have a deep sense of empathy.",
    "ISFP": "Artistic, sensitive, and spontaneous. ISFPs are gentle and enjoy living in the moment, often expressing themselves through art, music, or nature.",
    "ESTP": "Energetic, perceptive, and bold. ESTPs love excitement and are quick to take action. They are practical problem-solvers who thrive in fast-paced environments.",
    "ISTJ": "Responsible, organized, and reliable. ISTJs value tradition and order, and are known for their strong sense of duty and attention to detail.",
    "ENFP": "Enthusiastic, creative, and sociable. ENFPs are passionate about ideas and people, and are driven by curiosity and a desire to make meaningful connections.",
    "ESFJ": "Caring, outgoing, and loyal. ESFJs are highly attuned to the needs of others and excel at creating harmony in groups. They are supportive and value community.",
    "Unknown": "No description available for this type."
}

def classify_mbti(posts):
    """
    Classify MBTI types based on user's social media posts.
    :param posts: List of text posts (e.g., tweets or Reddit submissions)
    :return: Predicted MBTI type and confidence score
    """
    # Combine up to the first 10 posts into a single string for classification
    combined_text = " ".join(posts[:10])

    # Use the model to classify the combined text
    result = mbti_model(combined_text[:512])  # Truncate to 512 characters if necessary
    label = result[0]["label"]  # Get the label from the model output
    confidence = result[0]["score"]  # Get the confidence score

    # Map the label to an MBTI type
    mbti_type = label_to_mbti.get(label, "Unknown")  # Default to "Unknown" if label not found

    return mbti_type, confidence

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text.lower()[:512]

def analyze(text):
    if not text.strip():
        return "Neutral", 0.5, "Neutral", 0.0
    sentiment_result = sentiment_model(text[:512])[0]
    emotion_result = emotion_model(text[:512])[0]

    # Convert sentiment confidence score to 1 for positive and 0 for negative
    sentiment_score = 1.0 if sentiment_result['label'] == "POSITIVE" else 0.0

    # Determine risk level
    high_risk_emotions = ['anger', 'fear', 'sadness']
    risk_level = "High" if emotion_result['label'] in high_risk_emotions and sentiment_result['label'] == "NEGATIVE" else "Low"
    return sentiment_result['label'], sentiment_score, emotion_result['label'], emotion_result['score'], risk_level

def calculate_mental_state(df, mbti_type):
    # Aggregate sentiment and emotion results to calculate mental state
    negative_count = df[df['Sentiment'] == "NEGATIVE"].shape[0]
    high_risk_emotions = ['anger', 'fear', 'sadness']
    high_risk_count = df[df['Emotion'].isin(high_risk_emotions)].shape[0]

    if negative_count > len(df) * 0.5 and high_risk_count > len(df) * 0.5:
        overall_state = "Critical"
    elif negative_count > len(df) * 0.3 or high_risk_count > len(df) * 0.3:
        overall_state = "At Risk"
    else:
        overall_state = "Stable"

    # Include MBTI insights (placeholder logic, can be expanded)
    if mbti_type in ["INTJ", "INFJ", "ISTJ"]:
        overall_state += " (Analytical Personality)"
    elif mbti_type in ["ENFP", "ESFJ", "ENTP"]:
        overall_state += " (Expressive Personality)"

    return overall_state

def fetch_twitter_posts(username, limit=100):
    try:
        # Twitter API v2 credentials
        client = tweepy.Client(
            bearer_token="AAAAAAAAAAAAAAAAAAAAAAoN1QEAAAAA20HPtrL1eYqFk86sYW5gfZyXFp4%3DcrMbZLM1MvcgBXtA1RUtXz6WPFz2YAx4QGc4fmWEKxWiVuPo7n",  
            consumer_key="jXmK3owxloyC6s4O88RDnxzQS",        
            consumer_secret="AG33ykjOATY33LBtN060Aok9slx8EkHRruGMmyyoHiRPl24fMp",  
            access_token="1922207421444874240-lhTpyMTGIuM7E1JhkjDRkNbKG5dhh9",   
            access_token_secret="jtQaCOxsHcZUV7UmoVB9YX01pQ9mragjBX34aGobWOXE1"  
        )
        
        # Get user ID from username
        user = client.get_user(username=username)
        if not user.data:
            st.error(f"Twitter user '{username}' not found.")
            return []
            
        user_id = user.data.id
        
        # Fetch tweets using Twitter API v2
        tweets = client.get_users_tweets(
            id=user_id,
            max_results=limit,
            tweet_fields=['created_at', 'text'],
            exclude=['retweets', 'replies']
        )
        
        if not tweets.data:
            st.warning(f"No tweets found for user '{username}'.")
            return []
            
        return [tweet.text for tweet in tweets.data]
        
    except tweepy.errors.TweepyException as e:
        if "401" in str(e):
            st.error("Twitter API authentication failed. Please check your API credentials.")
        elif "404" in str(e):
            st.error(f"Twitter user '{username}' not found.")
        elif "429" in str(e):
            st.error("Twitter API rate limit exceeded. Please try again later.")
        else:
            st.error(f"Twitter API error: {str(e)}")
        return []
    except Exception as e:
        st.error("An unexpected error occurred while fetching Twitter posts.")
        st.error(f"Error Details: {str(e)}")
        return []

def fetch_reddit_posts(username, limit=100):
    try:
        # Initialize Reddit API client
        reddit = praw.Reddit(
            client_id="BZNBfoQviM8oJ14_Ie8Oag",
            client_secret="1GBs2FcrMMF7BPUXTV-twOsR9GRB2g",
            user_agent="MentalHealthAnalyzer"
        )
        user = reddit.redditor(username)
        # Fetch Reddit submissions
        return [submission.title + " " + submission.selftext for submission in user.submissions.new(limit=limit)]
    except praw.exceptions.RedditAPIException as e:
        st.error("Reddit API error. Please check the username or try again later.")
        return []
    except Exception as e:
        st.error("An unexpected error occurred while fetching Reddit posts.")
        st.error(f"Error Details: {e}")
        return []

def generate_report(df):
    df.to_csv("report.csv", index=False)

    # Function to convert matplotlib figure to base64
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str

    # Generate sentiment chart
    plt.figure(figsize=(7, 4))
    sns.countplot(x=df['Sentiment'], palette="coolwarm")
    plt.title("Sentiment Distribution", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    sentiment_img = fig_to_base64(plt.gcf())

    # Generate emotion chart
    plt.figure(figsize=(7, 4))
    sns.countplot(x=df['Emotion'], palette="Set3")
    plt.title("Emotion Distribution", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    emotion_img = fig_to_base64(plt.gcf())

    # Generate wordcloud
    wordcloud = WordCloud(width=800, height=400, max_words=100).generate(' '.join(df['Post'].tolist()))
    plt.figure(figsize=(7, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    wordcloud_img = fig_to_base64(plt.gcf())

    # Build HTML with embedded base64 images
    table_rows = "".join(
        f"<tr><td>{row.Post[:100]}...</td><td>{row.Sentiment}</td><td>{row.Confidence:.2f}</td>"
        f"<td>{row.Emotion}</td><td>{row.Emotion_Confidence:.2f}</td></tr>"
        for _, row in df.iterrows()
    )

    html_content = f'''
    <html>
    <head>
        <title>Mental Health Analysis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1, h2 {{
                color: #2c3e50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 12px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f5f5f5;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
                page-break-inside: avoid;
            }}
            .chart-container img {{
                max-width: 100%;
                height: auto;
                display: block;
                margin: 10px auto;
            }}
        </style>
    </head>
    <body>
        <h1>Mental Health & Emotion Analysis Report</h1>
        <p>Total Posts Analyzed: {len(df)}</p>
        
        <div class="chart-container">
            <h2>Sentiment Analysis</h2>
            <img src="data:image/png;base64,{sentiment_img}" alt="Sentiment Distribution">
        </div>
        
        <div class="chart-container">
            <h2>Emotion Analysis</h2>
            <img src="data:image/png;base64,{emotion_img}" alt="Emotion Distribution">
        </div>
        
        <div class="chart-container">
            <h2>Word Cloud Analysis</h2>
            <img src="data:image/png;base64,{wordcloud_img}" alt="Word Cloud">
        </div>

        <h2>Detailed Analysis Table</h2>
        <table>
            <tr>
                <th>Post</th>
                <th>Sentiment</th>
                <th>Confidence</th>
                <th>Emotion</th>
                <th>Emotion Confidence</th>
            </tr>
            {table_rows}
        </table>
    </body>
    </html>
    '''

    # Save HTML report
    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    # Generate PDF with A4 page size
    HTML(string=html_content).write_pdf(
        "mental_health_report.pdf",
        stylesheets=[],
        presentational_hints=True,
        zoom=1.0,
        page_size='A4'
    )

# ------------------------------
# 🌐 Streamlit Frontend
# ------------------------------
st.set_page_config(page_title="Mental Health Analyzer", layout="wide")
st.title("🧠 Social Media Mental Health Analyzer")

platform = st.radio("Select Platform", ["Twitter", "Reddit"])
username = st.text_input("Enter Username")

if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        posts = fetch_twitter_posts(username) if platform == "Twitter" else fetch_reddit_posts(username)
        results = []
        for post in posts:
            clean = preprocess_text(post)
            s, sc, e, ec, risk = analyze(clean)
            results.append((post, s, sc, e, ec, risk))

        df = pd.DataFrame(results, columns=["Post", "Sentiment", "Confidence", "Emotion", "Emotion_Confidence", "Risk Level"])
        # Calculate MBTI personality
        mbti_type, mbti_confidence = classify_mbti(posts)

        # Calculate overall mental state
        mental_state = calculate_mental_state(df, mbti_type)

        st.success("Analysis complete!")
        st.dataframe(df)

        st.subheader("📊 Sentiment Distribution")
        sent_counts = df['Sentiment'].value_counts()
        st.bar_chart(sent_counts)

        st.subheader("📊 Emotion Distribution")
        emo_counts = df['Emotion'].value_counts()
        st.bar_chart(emo_counts)

        st.subheader("🧠 MBTI Personality Type")
        st.write(f"**Type:** {mbti_type} (Confidence: {mbti_confidence:.2f})")
        st.markdown(f"**Description:** {mbti_descriptions.get(mbti_type, mbti_descriptions['Unknown'])}")
        st.markdown(f"Learn more about your type [here](https://www.16personalities.com/{mbti_type.lower()}-personality)")

        st.subheader("💡 Overall Mental State")
        st.write(f"**State:** {mental_state}")

        generate_report(df)
        import os
        if os.path.exists("mental_health_report.pdf"):
          with open("mental_health_report.pdf", "rb") as f:
            st.download_button("📥 Download PDF Report", f, "mental_health_report.pdf", mime="application/pdf")
        else:
            st.error("❌ PDF report could not be generated. Please check for missing fonts or WeasyPrint issues.") 
    #    with open("mental_health_report.pdf", "rb") as f:
    #       st.download_button("📥 Download PDF Report", f, "mental_health_report.pdf", mime="application/pdf")

    
        st.subheader("🆘 Get Help")
        st.markdown("""
        - [MentalHealth.gov](https://nhm.gov.in/index1.php?lang=1&level=2&sublinkid=1043&lid=359)
        - [BetterHelp](https://www.betterhelp.com/)
        - [7 Cups](https://www.7cups.com/)
        - [National Suicide Prevention Lifeline](https://988lifeline.org/)
        """)

        st.subheader("💬 Chatbot Assistant (Coming Soon)")
        st.markdown("An AI-based assistant will be available here to discuss results and suggest coping strategies.")

        # Launch instructions
        st.sidebar.title("About This Tool")
        st.sidebar.info("""
        This website analyzes social media content to provide insights about sentiment, emotions, and potential mental health indicators.
         """) 

# ===============================
# 🚀 LAUNCH STREAMLIT
# ===============================
import subprocess

# Launch the Streamlit app directly
try:
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])
    print("Streamlit app running on http://localhost:8501")
except Exception as e:
    print(f"Error launching Streamlit: {str(e)}")
