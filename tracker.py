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


def parse_citations(response):
    """
    Parse Perplexity API response to find wtatennis.com citations

    Args:
        response: API response dictionary

    Returns:
        tuple: (appears, position, citation_url)
            - appears: bool, whether wtatennis.com appears
            - position: int or "Not found"
            - citation_url: str or None
    """
    if not response:
        return False, "Error", None

    # Try to find citations in the response
    citations = response.get("citations", [])

    # Check each citation for WTA domains
    for idx, citation_url in enumerate(citations, start=1):
        # Check if any WTA domain is in the citation URL
        if any(domain in citation_url.lower() for domain in WTA_DOMAINS):
            return True, idx, citation_url

    # Not found
    return False, "Not found", None


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

        # Parse citations
        appears, position, citation_url = parse_citations(response)

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


def main():
    """Main Streamlit application"""

    st.title("🎾 WTA AI Ranking Tracker")
    st.markdown("Track how Perplexity AI cites wtatennis.com across different queries")

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
