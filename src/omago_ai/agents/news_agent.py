import os
import requests
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from omago_ai.rag.retriever import retrieve_news_docs

load_dotenv()

# 🔑 API keys
GEMINI_API_KEY = os.getenv("MY_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------- TICKER MAP ----------------

TICKER_TO_COMPANY = {
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HDFCBANK.NS": "HDFC Bank",
    "RELIANCE.NS": "Reliance Industries",
    "ITC.NS": "ITC Limited",
}

# ---------------- LIVE NEWS FETCHER ----------------

def fetch_live_news(company_name: str, ticker: str, limit: int = 3):
    if not NEWS_API_KEY:
        return []

    url = "https://newsdata.io/api/1/latest"
    params = {
        "apikey": NEWS_API_KEY,
        "q": company_name,
        "country": "in",
        "category": "business",
    }

    try:
        resp = requests.get(url, params=params, timeout=5).json()
        #print("Live news raw response:", resp)
    except Exception:
        return []

    articles = resp.get("results", [])
    docs = []

    company_lower = company_name.lower()
    short_name = ticker.replace(".NS", "").lower()

    for a in articles:
        title = a.get("title")
        description = a.get("description")

        if not title:
            continue

        # ✅ company must appear in title
        if company_lower not in title.lower() and short_name not in title.lower():
            continue

        # ✅ inject description if usable
        content_parts = [title]
        if description and "ONLY AVAILABLE" not in description.upper():
            content_parts.append(description)

        page_content = ". ".join(content_parts)

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "type": "live_news",
                    "ticker": ticker,
                    "source": a.get("source_id", "unknown"),
                },
            )
        )

        if len(docs) >= limit:
            break
    print(docs)
    return docs



# ---------------- NEWS AGENT ----------------

class NewsAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.3,
            google_api_key=GEMINI_API_KEY,
        )

    def analyze_company_news(self, *, ticker: str, headline: str):

        # 1️⃣ Resolve company name
        company_name = TICKER_TO_COMPANY.get(ticker, ticker.replace(".NS", ""))

        # 2️⃣ Headline-aware query
        headline_l = headline.lower()

        if "earnings" in headline_l or "results" in headline_l:
            query = f"{company_name} earnings"
        elif "policy" in headline_l or "government" in headline_l:
            query = f"{company_name} policy update"
        else:
            query = f"{company_name} business update"

        # 3️⃣ Fetch LIVE news
        live_docs = fetch_live_news(company_name, ticker)
        print("Live docs fetched:", [d.page_content for d in live_docs])

        # 4️⃣ Fetch STATIC RAG docs
        static_docs = retrieve_news_docs(query)

        # 5️⃣ Hybrid ordering
        latest_docs = [
            d for d in static_docs
            if d.metadata.get("type") == "latest_news"
            and d.metadata.get("ticker") == ticker
        ]

        other_docs = [d for d in static_docs if d not in latest_docs]

        ordered_docs = live_docs + latest_docs + other_docs

        # 6️⃣ Context (safe)
        context = "\n".join(d.page_content for d in ordered_docs)

        # ---------------- NEWS SIGNAL (CRITICAL FIX) ----------------

        news_signal_parts = [f"Headline: {headline}"]

        for d in live_docs:
            desc = d.metadata.get("description")
            if desc:
                news_signal_parts.append(
                    f"Title: {d.page_content}. Description: {desc}"
                )
            else:
                # 🔒 Force title reuse if description is missing
                news_signal_parts.append(
                    f"Title only available: {d.page_content}"
                )

        news_signal = " | ".join(news_signal_parts)

        # ---------------- PROMPT ----------------

        prompt = f"""
You are a calm news explanation assistant for an Indian stock market demo.


STRICT RULES:
- The NEWS HEADLINE and NEWS SIGNAL are the ONLY sources of meaning.
- You MUST reuse words from the titles in the News Signal.
- If only a title is available, build meaning ONLY from that title.
- Do NOT replace the title with generic phrases like "company update".
- Do NOT invent facts, numbers, or outcomes.
- Do NOT say details are missing.
- Don't use word signal or this news is about much in the ecxplanation
- Don't have to explain about definition of good/bad earnings
Context (secondary):
{context}

Company:
{company_name}

News headline:
{headline}

News signal:
{news_signal}

Tasks:
1. Restate the headline in simple words.
2. Identify the type of news.
3. Explain what the headline clearly signals.
4. Explain how this kind of news usually affects confidence.
5. Assign sentiment.
6. Assign impact score between 0 and 1.
7. If the signal mentions earnings, results, dividends, or records,
   you MUST explicitly use those words.
8. If the words crash, fall, drop are there assign negative sentiment
9. If the words rise, gain, increase, record are there assign positive sentiment

Style rules:
- Short sentences.
- One idea per sentence.
- No jargon.
- Calm tone.
- Suitable for cognitively disabled users.

Return JSON only.

Return format:
{{
  "explanation": "",
  "sentiment": "positive | negative | neutral",
  "impact_score": number
}}
"""

        response = self.llm.invoke(prompt)

        print("RAG docs used:", [d.metadata for d in ordered_docs])

        return response.content
