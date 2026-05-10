from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fakenews import predict_news

app = FastAPI()

API_KEY = "27a10a2a7a974c3293d4fcaf83cee6ee"

news_results = []

def fetch_news():
    global news_results
    url = f"https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={API_KEY}"
    data = requests.get(url).json()

    articles = data.get("articles", [])
    results = []

    for a in articles:
        title = a["title"]
        label, score = predict_news(title)

        results.append({
            "title": title,
            "label": label,
            "score": score
        })

    news_results = results


scheduler = BackgroundScheduler()
scheduler.add_job(fetch_news, 'interval', seconds=60)
scheduler.start()


@app.get("/", response_class=HTMLResponse)
def home():
    html = f"""
    <html>
    <head>
        <title>Fake News Detector</title>
        <meta http-equiv="refresh" content="30">

        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
                transition: 0.3s;
                background: #f4f6f9;
                color: #111;
            }}

            .dark {{
                background: #0f172a;
                color: white;
            }}

            header {{
                display: flex;
                justify-content: space-between;
                padding: 20px 40px;
                font-weight: bold;
            }}

            .toggle {{
                cursor: pointer;
                padding: 8px 15px;
                border-radius: 20px;
                background: #ddd;
            }}

            .container {{
                text-align: center;
                margin-top: 20px;
            }}

            textarea {{
                width: 50%;
                height: 120px;
                padding: 15px;
                border-radius: 12px;
                border: none;
                outline: none;
                font-size: 16px;
                background: #e5e7eb;
            }}

            .dark textarea {{
                background: #1e293b;
                color: white;
            }}

            button {{
                margin-top: 20px;
                padding: 12px 30px;
                border-radius: 25px;
                border: none;
                background: #3b82f6;
                color: white;
                font-size: 16px;
                cursor: pointer;
                transition: 0.2s;
            }}

            button:hover {{
                background: #2563eb;
            }}

            .news-section {{
                margin-top: 60px;
                padding: 20px;
            }}

            .card {{
                margin: 10px auto;
                width: 60%;
                padding: 15px;
                border-radius: 12px;
                background: white;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                text-align: left;
                transition: 0.3s;
            }}

            .dark .card {{
                background: #1e293b;
            }}

            .real {{
                color: #22c55e;
                font-weight: bold;
            }}

            .fake {{
                color: #ef4444;
                font-weight: bold;
            }}
        </style>

        <script>
            function toggleDark() {{
                document.body.classList.toggle("dark");
            }}
        </script>

    </head>

    <body>

    <header>
        <div>Fake News Prediction</div>
        <div class="toggle" onclick="toggleDark()">🌙 Dark Mode</div>
    </header>

    <div class="container">
        <h1>Check if news is real or fake</h1>

        <form action="/predict">
            <textarea name="news" placeholder="Enter the news here..." required></textarea><br>
            <button type="submit">Predict</button>
        </form>
    </div>

    <div class="news-section">
        <h2>📰 Live News</h2>
    """

    for n in news_results:
        cls = "real" if n["label"] == "REAL" else "fake"

        html += f"""
        <div class="card">
            <p>{n['title']}</p>
            <p class="{cls}">{n['label']} ({n['score']}%)</p>
        </div>
        """

    html += """
    </div>

    </body>
    </html>
    """
    return html


@app.get("/predict", response_class=HTMLResponse)
def predict(news: str):
    label, score = predict_news(news)
    color = "#22c55e" if label == "REAL" else "#ef4444"

    return f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI';
                text-align: center;
                background: #f4f6f9;
            }}

            .bar {{
                width: 250px;
                height: 12px;
                background: #ddd;
                margin: 15px auto;
                border-radius: 10px;
                overflow: hidden;
            }}

            .fill {{
                width: {score}%;
                height: 100%;
                background: #3b82f6;
            }}

            button {{
                padding: 10px 25px;
                border-radius: 20px;
                border: none;
                background: #3b82f6;
                color: white;
                cursor: pointer;
            }}
        </style>
    </head>

    <body>

    <h2>{news}</h2>

    <h1 style="color:{color}">{label}</h1>

    <div class="bar">
        <div class="fill"></div>
    </div>

    <p>{score}% confidence</p>

    <br>
    <a href="/"><button>⬅ Back</button></a>

    </body>
    </html>
    """