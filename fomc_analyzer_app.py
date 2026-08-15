
import streamlit as st
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
from google.colab import userdata # For Colab secrets; for deployment, consider st.secrets

# Suppress matplotlib warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# ---------------------------------------------------------
# 1. API Configuration & Authentication
# (Re-defining for self-contained Streamlit app)
# ---------------------------------------------------------

gemini_api_key_from_userdata = None
try:
    gemini_api_key_from_userdata = userdata.get('GEMINI_API_KEY')
except userdata.SecretNotFoundError:
    pass

GEMINI_API_KEY = gemini_api_key_from_userdata or os.environ.get('GEMINI_API_KEY') or "YOUR_GEMINI_API_KEY"

fred_api_key_from_userdata = None
try:
    fred_api_key_from_userdata = userdata.get('FRED_API_KEY')
except userdata.SecretNotFoundError:
    pass

FRED_API_KEY = fred_api_key_from_userdata or os.environ.get('FRED_API_KEY') or "YOUR_FRED_API_KEY"

# Initialize APIs only if keys are available
fred = None
client = None
if FRED_API_KEY and FRED_API_KEY != "YOUR_FRED_API_KEY":
    fred = Fred(api_key=FRED_API_KEY)
else:
    st.warning("FRED API Key not found. Please set 'FRED_API_KEY' in Colab secrets or environment variables.")

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    st.warning("Gemini API Key not found. Please set 'GEMINI_API_KEY' in Colab secrets or environment variables.")

# ---------------------------------------------------------
# 2. Automated FOMC Statement Scraper (Re-defined)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Scraping FOMC statements...")
def scrape_fomc_statement(date_str: str) -> str:
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date_str}a.htm"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"⚠️ Could not fetch from URL: {url} (Status: {response.status_code})")
        return ""

    soup = BeautifulSoup(response.content, "html.parser")
    article = soup.find("div", {"id": "article"}) or soup.find("div", {"class": "col-xs-12 col-sm-8 col-md-8"})
    if not article:
        return ""

    paragraphs = article.find_all("p")
    statement_text = "
".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
    return statement_text

# ---------------------------------------------------------
# 3. Macro Data Fetcher (Hard Time-Series) (Re-defined)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Fetching macroeconomic indicators from FRED...")
def fetch_macro_series(fred_api_obj: Fred, lookback_months: int = 36) -> pd.DataFrame:
    if not fred_api_obj:
        st.error("FRED API is not initialized. Please provide a valid FRED API Key.")
        return pd.DataFrame()
    series_map = {
        'FedFundsRate': 'FEDFUNDS',
        'CoreCPI': 'CPILFESL',
        'Unemployment': 'UNRATE',
        'Treasury2Y': 'DGS2'
    }

    data = {}
    for name, series_id in series_map.items():
        s = fred_api_obj.get_series(series_id)
        data[name] = s

    df = pd.DataFrame(data).dropna()
    df['CoreCPI_YoY'] = df['CoreCPI'].pct_change(12) * 100
    df = df.dropna().tail(lookback_months)
    return df

# ---------------------------------------------------------
# 4. Qualitative NLP Scorer (Pydantic Schema) (Re-defined)
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

@st.cache_data(show_spinner="Processing and scoring FOMC statements with Gemini...")
def analyze_historical_statements(gemini_client: genai.Client, statements_dict: dict[str, str]) -> list[FOMCSentiment]:
    if not gemini_client:
        st.error("Gemini API is not initialized. Please provide a valid Gemini API Key.")
        return []
    parsed_results = []
    for date_str, text in statements_dict.items():
        if not text:
            continue
        prompt = f"""
        You are a Federal Reserve monetary analyst. Analyze this official FOMC statement from {date_str}.
        Quantify its tone, summarize inflation/labor assessments, and extract key forward guidance.

        STATEMENT TEXT:
        """{text[:4000]}"""
        """
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": FOMCSentiment,
                    "temperature": 0.0
                }
            )
            parsed_results.append(response.parsed)
            st.write(f"  ✓ Processed {date_str}: Score = {response.parsed.hawkish_score:+.2f} ({response.parsed.action_taken})")
        except Exception as e:
            st.error(f"Error processing {date_str}: {e}")

    return parsed_results

# ---------------------------------------------------------
# 5. Dual-Stream Visualization (Re-defined)
# ---------------------------------------------------------
def plot_macro_and_commentary_timeline(macro_df: pd.DataFrame, sentiments: list[FOMCSentiment]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(macro_df.index, macro_df['FedFundsRate'], label='Fed Funds Rate (%)', color='#1f77b4', linewidth=2.5)
    ax1.plot(macro_df.index, macro_df['CoreCPI_YoY'], label='Core CPI (YoY %)', color='#d62728', linestyle='--', linewidth=2)
    ax1.plot(macro_df.index, macro_df['Unemployment'], label='Unemployment Rate (%)', color='#2ca02c', alpha=0.8)
    ax1.set_ylabel('Rate / Percentage (%)', fontsize=11)
    ax1.set_title('Macroeconomic Trends & Historical FOMC Sentiment Overlay', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.5)

    sent_dates = [pd.to_datetime(s.meeting_date) for s in sentiments]
    sent_scores = [s.hawkish_score for s in sentiments]
    colors = ['#d62728' if score > 0.1 else '#2ca02c' if score < -0.1 else '#7f7f7f' for score in sent_scores]

    ax2.scatter(sent_dates, sent_scores, color=colors, s=120, zorder=3)
    for i, s in enumerate(sentiments):
        ax2.annotate(f"{s.action_taken}
({s.hawkish_score:+.2f})",
                     (sent_dates[i], sent_scores[i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, weight='bold')

    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel('Hawkish (+) / Dovish (-)', fontsize=11)
    ax2.set_title('FOMC Statement Stance Progression', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 6. Pattern Recognition & Policy Forecasting (Re-defined)
# ---------------------------------------------------------
@st.cache_data(show_spinner="Synthesizing patterns with Gemini...")
def generate_macro_pattern_synthesis(gemini_client: genai.Client, macro_df: pd.DataFrame, sentiments: list[FOMCSentiment]) -> str:
    if not gemini_client:
        st.error("Gemini API is not initialized. Please provide a valid Gemini API Key.")
        return ""
    st.write("
🔍 Synthesizing patterns across macro time-series and FOMC narratives...")

    macro_summary = macro_df.tail(6).to_string()

    sentiment_summary = ""
    for s in sentiments:
        sentiment_summary += (
            f"- Date: {s.meeting_date} | Action: {s.action_taken} | Hawkishness: {s.hawkish_score:+.2f}
"
            f"  Inflation View: {s.inflation_assessment}
"
            f"  Labor View: {s.labor_assessment}
"
            f"  Key Guidance: "{s.key_forward_guidance_quote}"

"
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

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"temperature": 0.2}
    )
    return response.text

# ---------------------------------------------------------
# Streamlit App Layout and Logic
# ---------------------------------------------------------
def main():
    st.title("🏛️ FOMC Statement & Macro Pattern Analyzer")
    st.write("Analyze Federal Reserve statements and macroeconomic data to forecast policy.")

    st.sidebar.header("Configuration")

    # Input for FOMC dates
    default_dates = ["20230503", "20231101", "20240131", "20240501", "20240918"]
    fomc_dates_input = st.sidebar.text_area(
        "Enter FOMC Meeting Dates (YYYYMMDD, one per line)",
        value="
".join(default_dates)
    )
    sample_dates = [d.strip() for d in fomc_dates_input.split('
') if d.strip()]

    # Input for lookback months
    lookback_months = st.sidebar.slider(
        "Macro Data Lookback (months)",
        min_value=12,
        max_value=60,
        value=24,
        step=6
    )

    if st.sidebar.button("Run Analysis"):
        if not fred or not client:
            st.error("API keys are not configured correctly. Please check your 'FRED_API_KEY' and 'GEMINI_API_KEY'.")
            return

        st.subheader("1. Scraping FOMC Statements")
        fomc_corpus = {}
        for d in sample_dates:
            with st.spinner(f"Scraping statement for {d}..."):
                text = scrape_fomc_statement(d)
                if text:
                    fomc_corpus[f"{d[:4]}-{d[4:6]}-{d[6:]}"] = text
        st.success(f"Scraped {len(fomc_corpus)} FOMC statements.")
        st.json({date: text[:100] + '...' for date, text in fomc_corpus.items()})

        st.subheader("2. Fetching Macroeconomic Time-Series Data")
        macro_timeseries = fetch_macro_series(fred, lookback_months=lookback_months)
        st.success("Macroeconomic data fetched successfully.")
        st.dataframe(macro_timeseries.tail())

        st.subheader("3. Extracting Sentiment from FOMC Statements")
        statement_sentiments = analyze_historical_statements(client, fomc_corpus)
        st.success("Sentiment extraction complete.")
        for s in statement_sentiments:
            st.write(f"  - Date: {s.meeting_date}, Hawkishness: {s.hawkish_score:+.2f}, Action: {s.action_taken}")

        st.subheader("4. Visualizing Macro Trends and FOMC Commentary")
        if not macro_timeseries.empty and statement_sentiments:
            fig = plot_macro_and_commentary_timeline(macro_timeseries, statement_sentiments)
            st.pyplot(fig)
        else:
            st.warning("Insufficient data for visualization.")

        st.subheader("5. Generating Macro Pattern Recognition and Policy Forecast")
        if not macro_timeseries.empty and statement_sentiments:
            pattern_analysis = generate_macro_pattern_synthesis(client, macro_timeseries, statement_sentiments)
            st.write("
" + "="*70)
            st.write("INSTITUTIONAL MACRO REGIME & PATTERN ANALYSIS")
            st.write("="*70)
            st.markdown(pattern_analysis)
        else:
            st.warning("Insufficient data for pattern recognition and forecasting.")

if __name__ == '__main__':
    main()
