# rag/docs.py

RAG_DOCUMENTS = [

    # ---------------- COMPANY PROFILES ----------------

    {
        "id": "company_tcs",
        "text": (
            "Tata Consultancy Services (TCS) is one of India's largest IT services companies. "
            "It provides software development, consulting, and digital services to global businesses. "
            "TCS earnings and business updates often affect investor confidence."
        ),
        "metadata": {"type": "company", "ticker": "TCS.NS", "name": "Tata Consultancy Services"},
    },

    {
        "id": "company_infosys",
        "text": (
            "Infosys is a global IT consulting and software services company based in India. "
            "It works with large enterprises on digital transformation and outsourcing. "
            "Infosys news and earnings can influence market sentiment."
        ),
        "metadata": {"type": "company", "ticker": "INFY.NS", "name": "Infosys"},
    },

    {
        "id": "company_icici",
        "text": (
            "ICICI Bank is a major private sector bank in India. "
            "It provides banking, loans, and financial services to individuals and businesses. "
            "ICICI Bank earnings and policy updates are closely watched by investors."
        ),
        "metadata": {"type": "company", "ticker": "ICICIBANK.NS", "name": "ICICI Bank"},
    },

    {
        "id": "company_hdfc",
        "text": (
            "HDFC Bank is one of India's largest private banks. "
            "It offers retail and corporate banking services. "
            "News about HDFC Bank often affects confidence in the banking sector."
        ),
        "metadata": {"type": "company", "ticker": "HDFCBANK.NS", "name": "HDFC Bank"},
    },

    {
        "id": "company_reliance",
        "text": (
            "Reliance Industries is a large Indian conglomerate with businesses in energy, telecom, retail, and digital services. "
            "Major announcements by Reliance often have a strong market impact."
        ),
        "metadata": {"type": "company", "ticker": "RELIANCE.NS", "name": "Reliance Industries"},
    },

    {
        "id": "company_itc",
        "text": (
            "ITC Limited is a diversified Indian company involved in consumer goods, hotels, and agriculture. "
            "ITC news often relates to FMCG performance or policy changes."
        ),
        "metadata": {"type": "company", "ticker": "ITC.NS", "name": "ITC Limited"},
    },

    # ---------------- EVENT TYPES ----------------

    {
        "id": "event_news",
        "text": (
            "General company news may include announcements, partnerships, policy changes, or market reactions. "
            "Such news can change how people feel about a company."
        ),
        "metadata": {"type": "event", "event_type": "news"},
    },

    {
        "id": "event_earnings",
        "text": (
            "Earnings news talks about how much money a company made in a recent period. "
            "Better-than-expected earnings are usually positive. "
            "Worse-than-expected earnings are usually negative."
        ),
        "metadata": {"type": "event", "event_type": "earnings"},
    },

    {
        "id": "event_policy",
        "text": (
            "Policy-related news includes government rules, regulations, or compliance updates. "
            "Such news can affect how a company operates."
        ),
        "metadata": {"type": "event", "event_type": "policy"},
    },

    # ---------------- IMPACT & SENTIMENT ----------------

    {
        "id": "impact_guide",
        "text": (
            "Impact score shows how strong the news effect is. "
            "Low impact means small or short-term effect. "
            "High impact means strong or long-lasting effect."
        ),
        "metadata": {"type": "impact"},
    },

    {
        "id": "sentiment_guide",
        "text": (
            "Positive sentiment means confidence or optimism. "
            "Negative sentiment means concern or worry. "
            "Neutral sentiment means mixed or unclear feelings."
        ),
        "metadata": {"type": "sentiment"},
    },

    # ---------------- ACCESSIBILITY ----------------

    {
        "id": "accessibility",
        "text": (
            "Always explain in simple English. "
            "Use short sentences. "
            "Avoid financial jargon. "
            "Keep a calm and reassuring tone. "
            "Make explanations suitable for cognitively disabled users."
        ),
        "metadata": {"type": "accessibility"},
    },

    # ---------------- LATEST NEWS CONTEXT (RECENT THEMES) ----------------

{
    "id": "latest_tcs_news",
    "text": (
        "Recent news about Tata Consultancy Services often includes business updates, "
        "client deals, technology partnerships, hiring trends, and operational announcements. "
        "These updates help people understand the company’s current performance and direction."
    ),
    "metadata": {
        "type": "latest_news",
        "ticker": "TCS.NS",
        "recency": "current"
    },
},

{
    "id": "latest_infosys_news",
    "text": (
        "Recent news about Infosys usually relates to business outlook updates, "
        "large contract wins, leadership or organizational changes, "
        "and cost or efficiency measures. "
        "Such news shapes short-term confidence in the company."
    ),
    "metadata": {
        "type": "latest_news",
        "ticker": "INFY.NS",
        "recency": "current"
    },
},

{
    "id": "latest_icici_news",
    "text": (
        "Recent news about ICICI Bank often includes business updates, "
        "digital banking initiatives, lending activity, asset quality trends, "
        "and operational announcements. "
        "These updates inform people about the bank’s current direction."
    ),
    "metadata": {
        "type": "latest_news",
        "ticker": "ICICIBANK.NS",
        "recency": "current"
    },
},

{
    "id": "latest_hdfc_news",
    "text": (
        "Recent news about HDFC Bank commonly involves operational updates, "
        "loan growth, customer activity, technology improvements, "
        "or regulatory-related developments. "
        "Such news affects confidence in the banking sector."
    ),
    "metadata": {
        "type": "latest_news",
        "ticker": "HDFCBANK.NS",
        "recency": "current"
    },
},

{
    "id": "latest_reliance_news",
    "text": (
        "Recent news about Reliance Industries often includes updates on its "
        "energy, retail, telecom, or digital businesses. "
        "Major announcements usually signal strategic direction or expansion plans."
    ),
    "metadata": {
        "type": "latest_news",
        "ticker": "RELIANCE.NS",
        "recency": "current"
    },
},

{
    "id": "latest_itc_news",
    "text": (
        "Recent news about ITC Limited typically relates to consumer goods performance, "
        "business restructuring, sustainability efforts, or policy-related updates. "
        "These announcements influence how people view the company’s near-term outlook."
    ),
    "metadata": {
        "type": "latest_news",
        "ticker": "ITC.NS",
        "recency": "current"
    },
},

]
