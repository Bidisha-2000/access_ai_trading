from omago_ai.agents.news_agent import NewsAgent


def main():
    agent = NewsAgent()

    company = "TCS.NS"
    result = agent.analyze_company_news(company)

    print("\n--- NEWS ANALYSIS OUTPUT ---\n")
    print(result)


if __name__ == "__main__":
    main()
