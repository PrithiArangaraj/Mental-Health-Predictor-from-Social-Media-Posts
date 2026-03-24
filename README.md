# Mental Health Analyzer

An end-to-end AI-powered system that analyzes social media data to detect **sentiment, emotions, personality traits (MBTI), and mental health risk levels** using transformer-based NLP models.

---

##  Features

*  **Social Media Analysis**

  * Fetches data from **Twitter** and **Reddit**

*  **AI Models**

  * Sentiment Analysis (Positive / Negative)
  * Emotion Detection (Joy, Sadness, Anger, Fear, etc.)
  * MBTI Personality Classification

* **Mental Health Risk Detection**

  * Identifies high-risk emotional patterns
  * Classifies users into:

    * Stable
    * At Risk
    * Critical

* **Interactive Dashboard (Streamlit)**

  * Visualizes:

    * Sentiment distribution
    * Emotion distribution
  * Displays:

    * MBTI type + description
    * Overall mental state

*  **Automated Report Generation**

  * CSV, HTML, and PDF reports
  * Includes charts + word cloud + detailed analysis

---

##  Project Architecture

```
User Input (Username)
        ↓
Fetch Data (Twitter / Reddit)
        ↓
Preprocessing
        ↓
AI Models (Sentiment + Emotion + MBTI)
        ↓
Risk Detection + Mental State Analysis
        ↓
Streamlit Dashboard
        ↓
Report Generation (PDF / HTML / CSV)
```

---

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **AI/NLP Models:** Hugging Face Transformers
* **Libraries:**

  * pandas, numpy
  * matplotlib, seaborn
  * WordCloud
  * tweepy (Twitter API)
  * praw (Reddit API)
  * WeasyPrint (PDF generation)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mental-health-analyzer.git

# Navigate to project folder
cd mental-health-analyzer

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

##  Environment Variables (IMPORTANT)

 Do NOT hardcode API keys in your code.

Create a `.env` file:

```env
TWITTER_BEARER_TOKEN=your_token
TWITTER_API_KEY=your_key
TWITTER_API_SECRET=your_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_SECRET=your_access_secret

REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_secret
REDDIT_USER_AGENT=your_user_agent
```

---

##  Output Example

* Sentiment: Positive / Negative
* Emotion: Joy / Sadness / Anger / Fear
* MBTI: ENFP, INTJ, etc.
* Mental State: Stable / At Risk / Critical

---

##  Limitations

* Depends on availability of social media data
* API rate limits may affect performance
* Mental health predictions are **not medical diagnoses**

---

## Future Improvements

*  AI Chatbot for mental health support
*  Time-based trend analysis
*  Improved risk prediction using deep learning
*  Multi-language support

---

## Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

##  License

This project is licensed under the MIT License.

---

## Acknowledgements

* Hugging Face Transformers
* Streamlit
* Twitter & Reddit APIs

---

##  Author

**Prithiga**
AI & Data Science Enthusiast 

---
