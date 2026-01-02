# rag/loader.py

def fetch_latest_news():
    """
    Hackathon-safe mock news.
    Replace with API later if needed.
    """
    return [
        {
            "title": "Tech stocks fall after global slowdown fears",
            "content": (
                "Technology companies saw a decline as investors worried "
                "about slowing global economic growth."
            ),
        },
        {
            "title": "IT companies see stable demand outlook",
            "content": (
                "Major IT firms reported stable demand and cautious optimism "
                "for the coming quarters."
            ),
        },
    ]
