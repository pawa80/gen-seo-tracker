"""
WTA AI Ranking Tracker - Main Dashboard
Tracks how Perplexity AI cites wtatennis.com across different queries
"""

import os
import csv
import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import pandas as pd
import requests
from dotenv import load_dotenv
from queries import ALL_QUERIES, QUERY_CATEGORIES

# Load environment variables
load_dotenv()
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

# Constants
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "results.csv"
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
WTA_DOMAINS = ["wtatennis.com", "wta.com", "www.wtatennis.com", "www.wta.com"]

# Ensure results directory exists
RESULTS_DIR.mkdir(exist_ok=True)


def check_api_key():
    """Check if API key is configured"""
    return PERPLEXITY_API_KEY is not None and len(PERPLEXITY_API_KEY) > 0


def query_perplexity(query):
    """
    Query Perplexity API and return the response

    Args:
        query: The search query string

    Returns:
        dict: API response or None if error
    """
    if not PERPLEXITY_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    }

    try:
        response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error for query '{query}': {str(e)}")
        return None


def parse_citations(response, debug=False):
    """
    Parse Perplexity API response to find wtatennis.com citations

    Args:
        response: API response dictionary
        debug: If True, print debug information about response structure

    Returns:
        tuple: (appears, position, citation_url)
            - appears: bool, whether wtatennis.com appears
            - position: int or "Not found"
            - citation_url: str or None
    """
    if not response:
        return False, "Error", None

    try:
        # Check for search_results first (primary field used by Perplexity API)
        search_results = response.get("search_results", [])
        citations = response.get("citations", [])

        # Debug logging (only if requested)
        if debug:
            if search_results:
                st.info(f"DEBUG: Response has 'search_results' field with {len(search_results)} items")
                if search_results and isinstance(search_results[0], dict):
                    st.info(f"DEBUG: First result structure: {search_results[0]}")
            if citations:
                st.info(f"DEBUG: Response has 'citations' field with {len(citations)} items")
                if citations:
                    st.info(f"DEBUG: First citation: {citations[0]}")

        # Process search_results (array of objects with url, title, date)
        if search_results:
            for idx, result in enumerate(search_results, start=1):
                # Handle object structure
                if isinstance(result, dict):
                    citation_url = result.get("url", "")
                else:
                    # Handle string structure (fallback)
                    citation_url = str(result)

                # Check if any WTA domain is in the citation URL
                if citation_url and any(domain in citation_url.lower() for domain in WTA_DOMAINS):
                    if debug:
                        st.success(f"DEBUG: Found WTA at position {idx}: {citation_url}")
                    return True, idx, citation_url

        # Fallback to citations field (array of strings)
        if citations:
            for idx, citation in enumerate(citations, start=1):
                # Handle both string and object structures
                if isinstance(citation, dict):
                    citation_url = citation.get("url", "")
                else:
                    citation_url = str(citation)

                # Check if any WTA domain is in the citation URL
                if citation_url and any(domain in citation_url.lower() for domain in WTA_DOMAINS):
                    if debug:
                        st.success(f"DEBUG: Found WTA in citations at position {idx}: {citation_url}")
                    return True, idx, citation_url

        # Not found in either field
        if debug:
            st.warning("DEBUG: WTA domain not found in search_results or citations")
        return False, "Not found", None

    except Exception as e:
        st.error(f"Error parsing citations: {str(e)}")
        return False, "Error", None


def save_result(timestamp, query, appears, position, citation_url):
    """
    Save a single result to CSV file

    Args:
        timestamp: ISO format timestamp string
        query: The search query
        appears: "Yes" or "No"
        position: Citation position or "Not found"
        citation_url: The citation URL or empty string
    """
    # Create file with headers if it doesn't exist
    file_exists = RESULTS_FILE.exists()

    with open(RESULTS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header if new file
        if not file_exists:
            writer.writerow(['timestamp', 'query', 'appears', 'position', 'citation_url'])

        # Write result
        writer.writerow([timestamp, query, appears, position, citation_url or ''])


def load_results():
    """
    Load all historical results from CSV

    Returns:
        pd.DataFrame: Results dataframe or empty dataframe if file doesn't exist
    """
    if not RESULTS_FILE.exists():
        return pd.DataFrame(columns=['timestamp', 'query', 'appears', 'position', 'citation_url'])

    try:
        df = pd.read_csv(RESULTS_FILE, encoding='utf-8')
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'query', 'appears', 'position', 'citation_url'])


def run_weekly_check():
    """
    Run the weekly check for all queries

    Returns:
        list: List of result dictionaries
    """
    results = []
    total_queries = len(ALL_QUERIES)

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, query in enumerate(ALL_QUERIES):
        # Update progress
        progress = (idx + 1) / total_queries
        progress_bar.progress(progress)

        # Calculate estimated time remaining
        remaining_queries = total_queries - (idx + 1)
        estimated_seconds = remaining_queries * RATE_LIMIT_DELAY
        status_text.text(f"Checking query {idx + 1} of {total_queries}: '{query}' "
                        f"(~{int(estimated_seconds)}s remaining)")

        # Query Perplexity API
        response = query_perplexity(query)

        # Parse citations (enable debug for first query only)
        debug_mode = (idx == 0)
        appears, position, citation_url = parse_citations(response, debug=debug_mode)

        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save result
        appears_str = "Yes" if appears else "No"
        save_result(timestamp, query, appears_str, position, citation_url)

        # Store result for display
        results.append({
            'timestamp': timestamp,
            'query': query,
            'appears': appears_str,
            'position': position,
            'citation_url': citation_url or ''
        })

        # Rate limiting - wait before next query (except for last query)
        if idx < total_queries - 1:
            time.sleep(RATE_LIMIT_DELAY)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return results


def calculate_summary_stats(df):
    """
    Calculate summary statistics from historical results

    Args:
        df: Results dataframe

    Returns:
        dict: Summary statistics
    """
    if df.empty:
        return {
            'total_checks': 0,
            'appearance_rate': 0,
            'always_appears': [],
            'never_appears': [],
            'biggest_changes': []
        }

    # Get latest check timestamp
    latest_timestamp = df['timestamp'].max()
    latest_results = df[df['timestamp'] == latest_timestamp]

    # Calculate appearance rate
    total_queries_checked = len(latest_results)
    appears_count = len(latest_results[latest_results['appears'] == 'Yes'])
    appearance_rate = (appears_count / total_queries_checked * 100) if total_queries_checked > 0 else 0

    # Find queries that always appear
    query_stats = df.groupby('query')['appears'].apply(lambda x: (x == 'Yes').all())
    always_appears = query_stats[query_stats].index.tolist()

    # Find queries that never appear
    never_appears = query_stats[~query_stats].index.tolist()

    # Calculate position changes (if we have multiple checks)
    biggest_changes = []
    if len(df['timestamp'].unique()) >= 2:
        # Get the two most recent timestamps
        timestamps = sorted(df['timestamp'].unique(), reverse=True)[:2]
        recent = df[df['timestamp'] == timestamps[0]]
        previous = df[df['timestamp'] == timestamps[1]]

        for query in ALL_QUERIES:
            recent_row = recent[recent['query'] == query]
            previous_row = previous[previous['query'] == query]

            if not recent_row.empty and not previous_row.empty:
                recent_pos = recent_row.iloc[0]['position']
                previous_pos = previous_row.iloc[0]['position']

                # Only calculate change if both are numeric positions
                if isinstance(recent_pos, int) and isinstance(previous_pos, int):
                    change = previous_pos - recent_pos  # Positive means improved (moved up)
                    if change != 0:
                        biggest_changes.append({
                            'query': query,
                            'change': change,
                            'from': previous_pos,
                            'to': recent_pos
                        })

        # Sort by absolute change
        biggest_changes.sort(key=lambda x: abs(x['change']), reverse=True)
        biggest_changes = biggest_changes[:5]  # Top 5

    return {
        'total_checks': len(df['timestamp'].unique()),
        'appearance_rate': appearance_rate,
        'always_appears': always_appears,
        'never_appears': never_appears,
        'biggest_changes': biggest_changes
    }


def apply_wta_styling():
    """Apply WTA brand styling to the Streamlit app"""
    st.markdown("""
    <style>
        /* Import Inter font as fallback for General Sans */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

        /* Global Styles */
        .main {
            background-color: #FAFAF8;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #2C0046;
            max-width: 1400px;
            margin: 0 auto;
        }

        /* WTA Header */
        .wta-header {
            background: linear-gradient(135deg, #2C0046 0%, #7100B4 100%);
            padding: 32px 24px;
            border-radius: 0;
            border-bottom: 3px solid #00CD5A;
            margin: -60px -60px 32px -60px;
        }

        .wta-header h1 {
            color: #FFFFFF;
            font-size: 32px;
            font-weight: 900;
            margin: 0 0 8px 0;
            letter-spacing: -0.5px;
        }

        .wta-header .subtitle {
            color: #00CD5A;
            font-size: 18px;
            font-weight: 600;
            margin: 0;
        }

        /* Section Headers */
        .wta-section-header {
            color: #2C0046;
            font-size: 28px;
            font-weight: 800;
            margin: 32px 0 20px 0;
            padding-bottom: 12px;
            border-bottom: 3px solid;
            border-image: linear-gradient(135deg, #00CD5A 0%, #00A360 100%) 1;
        }

        /* Metric Boxes */
        .wta-metric-box {
            background: #FFFFFF;
            border: 2px solid #E8E8E8;
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .wta-metric-box:hover {
            border-color: #00CD5A;
            box-shadow: 0 4px 12px rgba(0, 163, 96, 0.1);
        }

        .wta-metric-box .metric-value {
            font-size: 28px;
            font-weight: 800;
            color: #00A360;
            margin-bottom: 8px;
        }

        .wta-metric-box .metric-label {
            font-size: 15px;
            font-weight: 600;
            color: #2C0046;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #00CD5A 0%, #00A360 100%);
            color: #2C0046;
            font-size: 18px;
            font-weight: 700;
            border: none;
            border-radius: 12px;
            padding: 16px 32px;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #00A360 0%, #008C4C 100%);
            box-shadow: 0 6px 16px rgba(0, 163, 96, 0.3);
            transform: translateY(-2px);
        }

        .stButton>button:active {
            transform: translateY(0);
        }

        /* Data Tables */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #E8E8E8;
        }

        .stDataFrame thead tr {
            background: linear-gradient(135deg, #00CD5A 0%, #00A360 100%);
        }

        .stDataFrame thead tr th {
            color: #2C0046 !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            font-size: 13px;
            letter-spacing: 0.5px;
            padding: 16px 12px;
        }

        .stDataFrame tbody tr:nth-child(even) {
            background-color: #FAFAF8;
        }

        .stDataFrame tbody tr:nth-child(odd) {
            background-color: #FFFFFF;
        }

        /* Metrics */
        .stMetric {
            background: #FFFFFF;
            border: 2px solid #E8E8E8;
            border-radius: 12px;
            padding: 20px;
        }

        .stMetric label {
            color: #2C0046 !important;
            font-weight: 600 !important;
            font-size: 15px;
        }

        .stMetric [data-testid="stMetricValue"] {
            color: #00A360 !important;
            font-weight: 800 !important;
            font-size: 24px;
        }

        /* Info/Success/Error Boxes */
        .stAlert {
            border-radius: 8px;
            border-left: 5px solid;
        }

        .stSuccess {
            background-color: rgba(0, 205, 90, 0.1);
            border-left-color: #00CD5A;
        }

        .stInfo {
            background-color: rgba(44, 0, 70, 0.05);
            border-left-color: #7100B4;
        }

        .stError {
            background-color: rgba(255, 107, 107, 0.1);
            border-left-color: #FF6B6B;
        }

        .stWarning {
            background-color: rgba(255, 193, 7, 0.1);
            border-left-color: #FFC107;
        }

        /* Dividers */
        hr {
            border: none;
            border-top: 2px solid #E8E8E8;
            margin: 32px 0;
        }

        /* Headers (fallback for non-custom headers) */
        h1 {
            color: #2C0046;
            font-weight: 900;
            font-size: 32px;
        }

        h2 {
            color: #2C0046;
            font-weight: 800;
            font-size: 28px;
            border-bottom: 3px solid;
            border-image: linear-gradient(135deg, #00CD5A 0%, #00A360 100%) 1;
            padding-bottom: 12px;
            margin-top: 32px;
        }

        h3 {
            color: #2C0046;
            font-weight: 700;
            font-size: 24px;
        }

        /* Download Button */
        .stDownloadButton>button {
            background: linear-gradient(135deg, #2C0046 0%, #7100B4 100%);
            color: #FFFFFF;
            font-weight: 700;
            border-radius: 12px;
            padding: 16px 32px;
            border: none;
        }

        .stDownloadButton>button:hover {
            background: linear-gradient(135deg, #7100B4 0%, #9D00E8 100%);
            box-shadow: 0 6px 16px rgba(113, 0, 180, 0.3);
        }

        /* Summary Box */
        .wta-summary-box {
            background: #FFFFFF;
            border-left: 5px solid #00CD5A;
            border-radius: 8px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }

        .wta-summary-box h4 {
            color: #2C0046;
            font-weight: 700;
            margin-top: 0;
        }

        /* Status Indicators */
        .status-yes {
            color: #00CD5A;
            font-weight: 600;
        }

        .status-no {
            color: #FF6B6B;
            font-weight: 600;
        }

        .position-number {
            color: #7100B4;
            font-weight: 700;
        }

        /* Code blocks */
        .stCode {
            background-color: #2C0046;
            color: #00CD5A;
            border-radius: 8px;
            font-family: 'Monaco', 'Courier New', monospace;
        }

        /* Progress bar */
        .stProgress > div > div {
            background: linear-gradient(135deg, #00CD5A 0%, #00A360 100%);
        }

        /* Columns spacing */
        [data-testid="column"] {
            padding: 0 12px;
        }

        /* Text elements */
        p {
            font-size: 15px;
            line-height: 1.6;
            color: #2C0046;
        }

        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 60px;
            padding-bottom: 60px;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application"""

    # Set page configuration
    st.set_page_config(
        page_title="WTA AI Ranking Tracker",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Apply WTA brand styling
    apply_wta_styling()

    # Page Header
    st.markdown("""
    <div class="wta-header">
        <h1>🎾 WTA AI Ranking Tracker</h1>
        <p class="subtitle">Track how Perplexity AI cites wtatennis.com across different queries</p>
    </div>
    """, unsafe_allow_html=True)

    # Configuration Section
    st.header("⚙️ Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Queries", len(ALL_QUERIES))

    with col2:
        api_status = "✅ Connected" if check_api_key() else "❌ Not Connected"
        st.metric("API Key Status", api_status)

    with col3:
        df = load_results()
        if not df.empty:
            last_check = df['timestamp'].max().strftime("%Y-%m-%d %H:%M")
            st.metric("Last Check", last_check)
        else:
            st.metric("Last Check", "Never")

    # API Key validation
    if not check_api_key():
        st.error("⚠️ Perplexity API key not found!")
        st.info("Please create a `.env` file with your API key. See `.env.example` for the format.")
        st.code("PERPLEXITY_API_KEY=your_api_key_here")
        return

    st.divider()

    # Run Check Section
    st.header("🔄 Run Check")

    if st.button("▶️ Run Weekly Check", type="primary", use_container_width=True):
        st.info(f"Starting check for {len(ALL_QUERIES)} queries... "
                f"This will take approximately {int(len(ALL_QUERIES) * RATE_LIMIT_DELAY)} seconds.")

        # Run the check
        results = run_weekly_check()

        st.success(f"✅ Check completed! Processed {len(results)} queries.")

        # Reload results to show latest data
        df = load_results()

    st.divider()

    # Latest Results Section
    st.header("📊 Latest Results")

    if not df.empty:
        # Get latest results
        latest_timestamp = df['timestamp'].max()
        latest_results = df[df['timestamp'] == latest_timestamp].copy()

        # Add category column
        latest_results['category'] = latest_results['query'].map(QUERY_CATEGORIES)

        # Reorder columns
        display_df = latest_results[['query', 'category', 'appears', 'position', 'citation_url', 'timestamp']]

        # Format timestamp
        display_df['timestamp'] = display_df['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Display with filtering
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "query": "Query",
                "category": "Category",
                "appears": "Appears?",
                "position": "Position",
                "citation_url": st.column_config.LinkColumn("Citation URL"),
                "timestamp": "Timestamp"
            }
        )

        # Summary metrics
        appears_count = len(latest_results[latest_results['appears'] == 'Yes'])
        st.metric("Queries with wtatennis.com citation",
                 f"{appears_count} / {len(latest_results)} ({appears_count/len(latest_results)*100:.1f}%)")
    else:
        st.info("No results yet. Click 'Run Weekly Check' to start tracking!")

    st.divider()

    # Historical Summary Section
    st.header("📈 Historical Summary")

    if not df.empty:
        stats = calculate_summary_stats(df)

        # Display summary metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Checks Run", stats['total_checks'])
            st.metric("Current Appearance Rate", f"{stats['appearance_rate']:.1f}%")

        with col2:
            st.metric("Always Appears", len(stats['always_appears']))
            st.metric("Never Appears", len(stats['never_appears']))

        # Show lists
        col1, col2 = st.columns(2)

        with col1:
            if stats['always_appears']:
                st.subheader("✅ Always Appears")
                for query in stats['always_appears']:
                    st.text(f"• {query}")
            else:
                st.subheader("✅ Always Appears")
                st.text("None yet")

        with col2:
            if stats['never_appears']:
                st.subheader("❌ Never Appears")
                for query in stats['never_appears'][:10]:  # Limit to 10
                    st.text(f"• {query}")
            else:
                st.subheader("❌ Never Appears")
                st.text("None")

        # Biggest position changes
        if stats['biggest_changes']:
            st.subheader("📊 Biggest Position Changes (Week over Week)")
            for change in stats['biggest_changes']:
                direction = "⬆️" if change['change'] > 0 else "⬇️"
                st.text(f"{direction} {change['query']}: #{change['from']} → #{change['to']} "
                       f"({change['change']:+d})")
    else:
        st.info("No historical data yet. Run at least one check to see summary statistics.")

    st.divider()

    # Export Section
    st.header("💾 Export Data")

    if not df.empty:
        # Convert dataframe to CSV
        csv_data = df.to_csv(index=False, encoding='utf-8')

        st.download_button(
            label="⬇️ Download Complete Results (CSV)",
            data=csv_data,
            file_name=f"wta_tracker_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Show total records
        st.text(f"Total records: {len(df)}")
    else:
        st.info("No data available to export yet.")


if __name__ == "__main__":
    main()
