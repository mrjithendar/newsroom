"""AI News Desk (docs/news/ARCHITECTURE.md): ingestion, ML, the News Desk Crew, quality gates and cover images.

Everything here runs inside CrewAI Flows (app/flows/news.py). ML steps are plain Python. Nothing in this package can
publish: drafts go to the web app's review queue through the existing callback API, and only the existing
approval → publish_flow path publishes (tests/test_news_flows.py asserts there is no code path from here to it).
"""
