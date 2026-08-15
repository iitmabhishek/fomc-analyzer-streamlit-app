
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from typing import Literal
from pydantic import BaseModel, Field
from fredapi import Fred
from google import genai
from google.colab import userdata

# Streamlit specific imports
import streamlit as st

# ---------------------------------------------------------
# 1. API Configuration & Authentication
# ---------------------------------------------------------

gemini_api_key_from_userdata = None
try:
    gemini_api_key_from_userdata = userdata.get('GEMINI_API_KEY') # Corrected key name
except userdata.SecretNotFoundError:
    pass # Secret not found in userdata, will fallback

GEMINI_API_KEY = gemini_api_key_from_userdata or os.environ.get('GEMINI_API_KEY') or "YOUR_GEMINI_API_KEY"

fred_api_key_from_userdata = None
try:
    fred_api_key_from_userdata = userdata.get('FRED_API_KEY')
except userdata.SecretNotFoundError:
    pass # Secret not found in userdata, will fallback

FRED_API_KEY = fred_api_key_from_userdata or os.environ.get('FRED_API_KEY') or "YOUR_FRED_API_KEY"

fred = Fred(api_key=FRED_API_KEY)
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# 2. Automated FOMC Statement Scraper
# ---------------------------------------------------------
def scrape_fomc_statement(date_str: str) -> str:
    """
    Fetches official FOMC statement text given a date (YYYYMMDD).
    Example dates: '20230503', '20231213', '20240131', '20240501', '20240918'
    """
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date_str}a.htm"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"⚠️ Could not fetch from URL: {url} (Status: {response.status_code})")
        return ""

    soup = BeautifulSoup(response.content, "html.parser")
    # Federal reserve statements are enclosed in the article div
    article = soup.find("div", {"id": "article"}) or soup.find("div", {"class": "col-xs-12 col-sm-8 col-md-8"})
    if not article:
        return ""

    paragraphs = article.find_all("p")
    statement_text = "\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    return statement_text

# ---------------------------------------------------------
# 3. Macro Data Fetcher (Hard Time-Series)
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_macro_series(lookback_months: int = 36) -> pd.DataFrame:
    """
    Pulls Core CPI, Fed Funds Rate, Unemployment Rate, and 2-Year Treasury Yield.
    """
    st.info("📈 Fetching macroeconomic indicators from FRED...")
    series_map = {
        'FedFundsRate': 'FEDFUNDS',      # Monthly Effective Fed Funds Rate
        'CoreCPI': 'CPILFESL',           # Core CPI Index
        'Unemployment': 'UNRATE',        # Civilian Unemployment Rate (%)
        'Treasury2Y': 'DGS2'             # 2-Year Treasury Yield (%)
    }

    data = {}
    for name, series_id in series_map.items():
        s = fred.get_series(series_id)
        data[name] = s

    df = pd.DataFrame(data).dropna()
    # Compute Year-over-Year Core CPI
    df['CoreCPI_YoY'] = df['CoreCPI'].pct_change(12) * 100
    df = df.dropna().tail(lookback_months)
    return df

# ---------------------------------------------------------
# 4. Qualitative NLP Scorer (Pydantic Schema)
# ---------------------------------------------------------
class FOMCSentiment(BaseModel):
    meeting_date: str = Field(description="Date of the statement in YYYY-MM-DD format")
    hawkish_score: float = Field(
        description="Sentiment score: -1.0 (Extremely Dovish / Rate Cuts) to +1.0 (Extremely Hawkish / Rate Hikes)"
    )
    inflation_assessment: str = Field(description="One-sentence summary of the Fed's stance on inflation trajectory.")
    labor_assessment: str = Field(description="One-sentence summary of the Fed's stance on labor market conditions.")
    action_taken: Literal["Hike", "Pause / Hold", "Cut"] = Field(description="Rate decision enacted in this statement.")
    key_forward_guidance_quote: str = Field(description="Verbatim key quote signaling the future policy path.")

@st.cache_data(ttl=3600) # Cache results for 1 hour
def analyze_historical_statements(statements_dict: dict[str, str]) -> list[FOMCSentiment]:
    """
    Scores each historical FOMC statement using Gemini structured outputs.
    """
    parsed_results = []
    st.info(f"🤖 Processing and scoring {len(statements_dict)} historical FOMC statements...")

    for date_str, text in statements_dict.items():
        if not text:
            continue
        prompt = f"""
        You are a Federal Reserve monetary analyst. Analyze this official FOMC statement from {date_str}.
        Quantify its tone, summarize inflation/labor assessments, and extract key forward guidance.

        STATEMENT TEXT:
        {text[:4000]}
        """
        response = client.models.generate_content(
            model="gemini-2.5-flash", # Using a fast model for sentiment analysis
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FOMCSentiment,
                "temperature": 0.0 # Keep temperature low for deterministic output
            }
        )
        parsed_results.append(response.parsed)
        st.write(f"  ✓ Processed {date_str}: Score = {response.parsed.hawkish_score:+.2f} ({response.parsed.action_taken})")

    return parsed_results

# ---------------------------------------------------------
# 5. Dual-Stream Visualization
# ---------------------------------------------------------
def plot_macro_and_commentary_timeline(macro_df: pd.DataFrame, sentiments: list[FOMCSentiment]):
    """
    Plots macro indicators alongside historical FOMC sentiment markers.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # 1. Macro Indicators
    ax1.plot(macro_df.index, macro_df['FedFundsRate'], label='Fed Funds Rate (%)', color='#1f77b4', linewidth=2.5)
    ax1.plot(macro_df.index, macro_df['CoreCPI_YoY'], label='Core CPI (YoY %)', color='#d62728', linestyle='--', linewidth=2)
    ax1.plot(macro_df.index, macro_df['Unemployment'], label='Unemployment Rate (%)', color='#2ca02c', alpha=0.8)
    ax1.set_ylabel('Rate / Percentage (%)', fontsize=11)
    ax1.set_title('Macroeconomic Trends & Historical FOMC Sentiment Overlay', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # 2. Hawkish / Dovish Sentiment Track
    sent_dates = [pd.to_datetime(s.meeting_date) for s in sentiments]
    sent_scores = [s.hawkish_score for s in sentiments]
    colors = ['#d62728' if score > 0.1 else '#2ca02c' if score < -0.1 else '#7f7f7f' for score in sent_scores]

    ax2.scatter(sent_dates, sent_scores, color=colors, s=120, zorder=3)
    for i, s in enumerate(sentiments):
        ax2.annotate(f"{s.action_taken}\n({s.hawkish_score:+.2f})",
                     (sent_dates[i], sent_scores[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, weight='bold')

    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel('Hawkish (+) / Dovish (-)', fontsize=11)
    ax2.set_title('FOMC Statement Stance Progression', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------
# 6. Pattern Recognition & Policy Forecasting
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # Cache results for 1 hour
def generate_macro_pattern_synthesis(macro_df: pd.DataFrame, sentiments: list[FOMCSentiment]) -> str:
    """
    Feeds time-series summary + narrative timeline into Gemini to identify reaction functions.
    """
    st.info("\n🔍 Synthesizing patterns across macro time-series and FOMC narratives...")

    # Format macro context
    macro_summary = macro_df.tail(6).to_string()

    # Format qualitative timeline
    sentiment_summary = ""
    for s in sentiments:
        sentiment_summary += (
            f"- Date: {s.meeting_date} | Action: {s.action_taken} | Hawkishness: {s.hawkish_score:+.2f}\n"
            f"  Inflation View: {s.inflation_assessment}\n"
            f"  Labor View: {s.labor_assessment}\n"
            f"  Key Guidance: \"{s.key_forward_guidance_quote}\"\n\n"
        )

    prompt = f"""
    You are a Lead Macroeconomic Strategist. Analyze the following dual-stream dataset:

    1. RECENT MACROECONOMIC DATA (Last 6 months):
    {macro_summary}

    2. HISTORICAL FOMC STATEMENT TRAJECTORY:
    {sentiment_summary}

    TASKS:
    1. **Identify the Fed's Reaction Function:** How sensitive has FOMC language been to changes in Core CPI vs. Unemployment over this timeframe?
    2. **Detect Policy Lags / Inflection Points:** Where did the narrative shift before the interest rate action occurred (or vice versa)?
    3. **Forward Macro Outlook:** Based on current macro trajectories and the latest guidance tone, forecast the monetary policy path over the next 6 months.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )
    return response.text

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(layout="wide")
st.title("🏛️ FOMC Statement and Macro Indicator Analyzer")
st.markdown("This app scrapes FOMC statements, analyzes sentiment, pulls macroeconomic data, and uses an LLM to synthesize patterns and forecast policy.")

# Sidebar for user inputs
st.sidebar.header("Configuration")
selected_dates_input = st.sidebar.text_area(
    "Enter FOMC Statement Dates (YYYYMMDD, one per line):",
    value="20230503\n20231101\n20240131\n20240501\n20240918\n20241106\n20241218"
)
lookback_months = st.sidebar.slider("Macro Data Lookback (months):", min_value=12, max_value=60, value=24)

# Process dates from input
sample_dates = [d.strip() for d in selected_dates_input.split('\n') if d.strip()]

if st.sidebar.button("Run Analysis"):
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        st.error("Please provide your GEMINI_API_KEY in Colab secrets or environment variables.")
    if not FRED_API_KEY or FRED_API_KEY == "YOUR_FRED_API_KEY":
        st.error("Please provide your FRED_API_KEY in Colab secrets or environment variables.")
    
    if (GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY") and \
       (FRED_API_KEY and FRED_API_KEY != "YOUR_FRED_API_KEY"):
        st.subheader("1. Scraping FOMC Statements")
        fomc_corpus = {}
        for d in sample_dates:
            with st.spinner(f"Scraping statement for {d}..."):
                text = scrape_fomc_statement(d)
            if text:
                fomc_corpus[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = text

        st.subheader("2. Fetching Macroeconomic Data")
        macro_timeseries = fetch_macro_series(lookback_months=lookback_months)
        st.write(macro_timeseries.tail())

        st.subheader("3. Analyzing FOMC Statement Sentiment")
        statement_sentiments = analyze_historical_statements(fomc_corpus)

        st.subheader("4. Visualizing Macro Trends and FOMC Stance")
        plot_macro_and_commentary_timeline(macro_timeseries, statement_sentiments)

        st.subheader("5. LLM-Powered Macro Pattern Recognition & Forecasting")
        pattern_analysis = generate_macro_pattern_synthesis(macro_timeseries, statement_sentiments)
        st.markdown("### INSTITUTIONAL MACRO REGIME & PATTERN ANALYSIS")
        st.write(pattern_analysis)
