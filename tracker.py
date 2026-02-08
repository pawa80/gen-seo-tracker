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
import plotly.graph_objects as go
import plotly.express as px
from supabase import create_client, Client
from queries import ALL_QUERIES, QUERY_CATEGORIES, CATEGORY_ORDER, CATEGORY_GOALS

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

# Pål's Scandinavian Depth Palette
PAL_CHARCOAL = "#2C3135"  # Main backgrounds, headers
PAL_AMBER = "#FF9500"     # CTAs, links, highlights
PAL_TEAL = "#0A7C7C"      # Secondary elements
PAL_WARM_WHITE = "#FAFAF8"  # Light backgrounds
PAL_STONE_GRAY = "#8B8B8B"  # Borders, secondary text
PAL_TEXT_DARK = "#1A1D1F"   # Dark text
PAL_TEXT_LIGHT = "#FFFFFF"  # Light text


def check_api_key():
    """Check if API key is configured"""
    return PERPLEXITY_API_KEY is not None and len(PERPLEXITY_API_KEY) > 0


def get_supabase_client():
    """
    Initialize and return Supabase client using Streamlit secrets.

    Returns:
        Client: Supabase client or None if credentials not configured
    """
    try:
        supabase_url = st.secrets.get("supabase_url")
        supabase_key = st.secrets.get("supabase_key")

        if not supabase_url or not supabase_key:
            return None

        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.warning(f"Could not initialize Supabase client: {str(e)}")
        return None


def save_result_to_supabase(supabase_client, check_date, query, category, appears, position, citation_url):
    """
    Save a single result to Supabase database using UPSERT.

    Uses UPSERT with (query, engine, check_date_only) as the conflict key.
    This ensures that if a check is restarted on the same day, results are
    updated rather than duplicated.

    Args:
        supabase_client: Initialized Supabase client
        check_date: ISO format timestamp string
        query: The search query
        category: Query category
        appears: Boolean - whether wtatennis.com appears
        position: Citation position (int) or None
        citation_url: The citation URL or None

    Returns:
        bool: True if save successful, False otherwise
    """
    if not supabase_client:
        return False

    try:
        # Extract date-only for the unique constraint
        check_date_only = check_date.split(" ")[0] if " " in check_date else check_date.split("T")[0]

        # Prepare data for upsert
        data = {
            "check_date": check_date,
            "check_date_only": check_date_only,
            "query": query,
            "category": category,
            "appears": appears,
            "position": position if isinstance(position, int) else None,
            "citation_url": citation_url if citation_url else None,
            "engine": "perplexity"
        }

        # UPSERT: Insert or update if (query, engine, check_date_only) already exists
        # This prevents duplicates from app restarts while allowing one row per query per day
        supabase_client.table("check_results").upsert(
            data,
            on_conflict="query,engine,check_date_only"
        ).execute()
        return True
    except Exception as e:
        # Log error but don't crash - database storage is supplementary
        st.warning(f"Failed to save to database: {str(e)}")
        return False


def get_historical_data(supabase_client):
    """
    Fetch all historical check results from Supabase.

    Args:
        supabase_client: Initialized Supabase client

    Returns:
        pd.DataFrame: Historical results or empty DataFrame if not available
    """
    if not supabase_client:
        return pd.DataFrame()

    try:
        response = supabase_client.table('check_results').select('*').order('check_date').execute()
        if response.data:
            df = pd.DataFrame(response.data)
            # Parse check_date as datetime
            df['check_date'] = pd.to_datetime(df['check_date'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not fetch historical data: {str(e)}")
        return pd.DataFrame()


def load_results_from_supabase(supabase_client):
    """
    Load results from Supabase in CSV-compatible format for dashboard display.

    This function converts Supabase data to match the format expected by
    existing dashboard functions (which were designed for CSV data).

    Args:
        supabase_client: Initialized Supabase client

    Returns:
        pd.DataFrame: Results in CSV-compatible format with columns:
                      timestamp, query, appears, position, citation_url
    """
    hist_df = get_historical_data(supabase_client)

    if hist_df.empty:
        return pd.DataFrame(columns=['timestamp', 'query', 'appears', 'position', 'citation_url'])

    # Convert to CSV-compatible format
    df = hist_df.copy()

    # IMPORTANT: Normalize timestamps to start-of-day for each check date
    # This ensures all rows from the same check run have identical timestamps,
    # which is required for filtering "latest results" correctly.
    # Without this, df['timestamp'].max() would return only 1 row (the last query of the run).
    df['timestamp'] = pd.to_datetime(df['check_date'].dt.date)

    df['appears'] = df['appears'].apply(lambda x: 'Yes' if x else 'No')

    # Ensure all expected columns exist
    result_df = df[['timestamp', 'query', 'appears', 'position', 'citation_url']].copy()

    return result_df


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
        tuple: (list of result dictionaries, dict with database save stats)
    """
    results = []
    total_queries = len(ALL_QUERIES)
    db_success_count = 0
    db_fail_count = 0

    # Initialize Supabase client (may be None if not configured)
    supabase_client = get_supabase_client()

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

        # Get category for this query
        category = QUERY_CATEGORIES.get(query, "Unknown")

        # Save result to CSV (backup storage)
        appears_str = "Yes" if appears else "No"
        save_result(timestamp, query, appears_str, position, citation_url)

        # Save result to Supabase database
        if supabase_client:
            db_saved = save_result_to_supabase(
                supabase_client,
                timestamp,
                query,
                category,
                appears,  # Boolean for database
                position,
                citation_url
            )
            if db_saved:
                db_success_count += 1
            else:
                db_fail_count += 1

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

    # Return results and database stats
    db_stats = {
        'enabled': supabase_client is not None,
        'success': db_success_count,
        'failed': db_fail_count
    }

    return results, db_stats


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


def render_executive_summary(df):
    """Render executive summary with key metrics"""
    if df.empty:
        return

    latest_timestamp = df['timestamp'].max()
    latest = df[df['timestamp'] == latest_timestamp]

    # Calculate metrics
    total_queries = len(latest)
    cited_queries = len(latest[latest['appears'] == 'Yes'])
    cite_rate = (cited_queries / total_queries * 100) if total_queries > 0 else 0

    # Average position (only for cited queries)
    cited_positions = latest[latest['appears'] == 'Yes']['position']
    numeric_positions = [p for p in cited_positions if isinstance(p, (int, float))]
    avg_position = sum(numeric_positions) / len(numeric_positions) if numeric_positions else 0

    # Calculate trend (if we have previous check)
    trend = ""
    if len(df['timestamp'].unique()) >= 2:
        timestamps = sorted(df['timestamp'].unique(), reverse=True)[:2]
        previous = df[df['timestamp'] == timestamps[1]]
        prev_cited = len(previous[previous['appears'] == 'Yes'])
        prev_cite_rate = (prev_cited / len(previous) * 100)
        change = cite_rate - prev_cite_rate
        if change > 0:
            trend = f"↑ +{change:.1f}%"
        elif change < 0:
            trend = f"↓ {change:.1f}%"
        else:
            trend = "→ No change"

    # Display in prominent box with Pål's colors
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {PAL_CHARCOAL} 0%, {PAL_TEAL} 100%);
                padding: 24px; border-radius: 12px; border-bottom: 3px solid {PAL_AMBER};
                margin-bottom: 24px;">
        <h2 style="color: {PAL_TEXT_LIGHT}; margin: 0 0 16px 0;">📊 Executive Summary</h2>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div>
                <div style="color: {PAL_AMBER}; font-size: 36px; font-weight: 800;">{cite_rate:.1f}%</div>
                <div style="color: {PAL_TEXT_LIGHT}; font-size: 14px;">Overall Authority Score</div>
                <div style="color: {PAL_TEXT_LIGHT}; font-size: 12px; opacity: 0.8;">{cited_queries}/{total_queries} queries cite WTA</div>
            </div>
            <div>
                <div style="color: {PAL_AMBER}; font-size: 36px; font-weight: 800;">{avg_position:.1f}</div>
                <div style="color: {PAL_TEXT_LIGHT}; font-size: 14px;">Average Authority Rank</div>
                <div style="color: {PAL_TEXT_LIGHT}; font-size: 12px; opacity: 0.8;">In 6-source authority set</div>
            </div>
            <div>
                <div style="color: {PAL_AMBER}; font-size: 36px; font-weight: 800;">{trend if trend else 'N/A'}</div>
                <div style="color: {PAL_TEXT_LIGHT}; font-size: 14px;">Trend vs. Previous</div>
                <div style="color: {PAL_TEXT_LIGHT}; font-size: 12px; opacity: 0.8;">Bi-weekly comparison</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_position_distribution(df):
    """Render position distribution bar chart with Pål's colors"""
    if df.empty:
        return

    latest_timestamp = df['timestamp'].max()
    latest = df[df['timestamp'] == latest_timestamp]

    # Count positions
    position_counts = {}
    for i in range(1, 7):
        position_counts[f"Position {i}"] = len(latest[latest['position'] == i])
    position_counts["Not Found"] = len(latest[latest['appears'] == 'No'])

    # Previous check comparison (if available)
    if len(df['timestamp'].unique()) >= 2:
        timestamps = sorted(df['timestamp'].unique(), reverse=True)[:2]
        previous = df[df['timestamp'] == timestamps[1]]

        prev_counts = {}
        for i in range(1, 7):
            prev_counts[f"Position {i}"] = len(previous[previous['position'] == i])
        prev_counts["Not Found"] = len(previous[previous['appears'] == 'No'])
    else:
        prev_counts = {k: 0 for k in position_counts.keys()}

    # Create figure with Pål's colors
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Current',
        x=list(position_counts.keys()),
        y=list(position_counts.values()),
        marker_color=PAL_AMBER
    ))

    if any(v > 0 for v in prev_counts.values()):
        fig.add_trace(go.Bar(
            name='Previous',
            x=list(prev_counts.keys()),
            y=list(prev_counts.values()),
            marker_color=PAL_TEAL
        ))

    fig.update_layout(
        title="Position Distribution",
        xaxis_title="Authority Rank",
        yaxis_title="Number of Queries",
        barmode='group',
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)


def render_category_performance(df):
    """Render category-level performance metrics"""
    if df.empty:
        return

    latest_timestamp = df['timestamp'].max()
    latest = df[df['timestamp'] == latest_timestamp]

    category_stats = []
    for category in CATEGORY_ORDER:
        cat_queries = latest[latest['category'] == category]
        if len(cat_queries) == 0:
            continue

        cited = len(cat_queries[cat_queries['appears'] == 'Yes'])
        total = len(cat_queries)
        cite_rate = (cited / total * 100) if total > 0 else 0

        # Average position
        positions = [p for p in cat_queries['position'] if isinstance(p, (int, float))]
        avg_pos = sum(positions) / len(positions) if positions else 0

        # Goal
        goal = CATEGORY_GOALS.get(category, 70)

        # Trend (if available)
        trend = "→"
        if len(df['timestamp'].unique()) >= 2:
            timestamps = sorted(df['timestamp'].unique(), reverse=True)[:2]
            prev = df[df['timestamp'] == timestamps[1]]
            prev_cat = prev[prev['category'] == category]
            if len(prev_cat) > 0:
                prev_cited = len(prev_cat[prev_cat['appears'] == 'Yes'])
                prev_cite_rate = (prev_cited / len(prev_cat) * 100)
                change = cite_rate - prev_cite_rate
                if change > 0:
                    trend = f"↑ +{change:.1f}%"
                elif change < 0:
                    trend = f"↓ {change:.1f}%"

        category_stats.append({
            'Category': category,
            'Citation %': f"{cite_rate:.1f}%",
            'Avg Position': f"{avg_pos:.1f}" if avg_pos > 0 else "N/A",
            'Trend': trend,
            'Queries': total,
            'Goal': f"{goal}%"
        })

    category_df = pd.DataFrame(category_stats)
    st.dataframe(category_df, use_container_width=True, hide_index=True)


def render_historical_citation_trend(hist_df):
    """
    Render a line chart showing overall citation rate over time.

    Args:
        hist_df: DataFrame with historical data from Supabase
    """
    if hist_df.empty:
        st.info("No historical data available from Supabase.")
        return

    # Group by date (ignore time component)
    hist_df['check_date_only'] = hist_df['check_date'].dt.date

    # Calculate citation rate per check date
    trend_data = []
    for date in sorted(hist_df['check_date_only'].unique()):
        date_df = hist_df[hist_df['check_date_only'] == date]
        total = len(date_df)
        cited = len(date_df[date_df['appears'] == True])
        rate = (cited / total * 100) if total > 0 else 0
        trend_data.append({
            'Date': pd.to_datetime(date),
            'Citation Rate (%)': rate,
            'Cited': cited,
            'Total': total
        })

    if len(trend_data) < 1:
        st.info("Not enough data points to show trend.")
        return

    trend_df = pd.DataFrame(trend_data)

    # Create line chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=trend_df['Date'],
        y=trend_df['Citation Rate (%)'],
        mode='lines+markers',
        name='Citation Rate',
        line=dict(color=PAL_AMBER, width=3),
        marker=dict(size=10, color=PAL_AMBER),
        hovertemplate='%{x|%b %d, %Y}<br>Rate: %{y:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Overall Citation Rate Over Time",
        xaxis_title="Check Date",
        yaxis_title="Citation Rate (%)",
        yaxis=dict(range=[0, 100]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickformat='%b %d, %Y'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show data table below
    display_trend = trend_df.copy()
    display_trend['Date'] = display_trend['Date'].dt.strftime('%b %d, %Y')
    display_trend['Citation Rate (%)'] = display_trend['Citation Rate (%)'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_trend, use_container_width=True, hide_index=True)


def render_category_trend_comparison(hist_df):
    """
    Render category performance comparison across check dates.

    Args:
        hist_df: DataFrame with historical data from Supabase
    """
    if hist_df.empty:
        st.info("No historical data available from Supabase.")
        return

    # Group by date (ignore time component)
    hist_df['check_date_only'] = hist_df['check_date'].dt.date
    dates = sorted(hist_df['check_date_only'].unique())

    if len(dates) < 1:
        st.info("Not enough data to compare.")
        return

    # Build comparison data
    comparison_data = []
    for category in CATEGORY_ORDER:
        row = {'Category': category}
        prev_rate = None

        for date in dates:
            date_df = hist_df[(hist_df['check_date_only'] == date) & (hist_df['category'] == category)]
            total = len(date_df)
            cited = len(date_df[date_df['appears'] == True])
            rate = (cited / total * 100) if total > 0 else 0

            date_label = pd.to_datetime(date).strftime('%b %d, %Y')
            row[date_label] = f"{rate:.1f}%"

            # Calculate change from previous date
            if prev_rate is not None:
                change = rate - prev_rate
                if change > 0:
                    row[f'{date_label} Δ'] = f"↑ +{change:.1f}%"
                elif change < 0:
                    row[f'{date_label} Δ'] = f"↓ {change:.1f}%"
                else:
                    row[f'{date_label} Δ'] = "→ 0%"
            prev_rate = rate

        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Create grouped bar chart if we have multiple dates
    if len(dates) >= 2:
        fig = go.Figure()

        colors = [PAL_TEAL, PAL_AMBER, '#6B8E23', '#CD853F', '#4682B4']  # Extended palette

        for i, date in enumerate(dates):
            date_label = pd.to_datetime(date).strftime('%b %d, %Y')
            rates = []
            for category in CATEGORY_ORDER:
                date_df = hist_df[(hist_df['check_date_only'] == date) & (hist_df['category'] == category)]
                total = len(date_df)
                cited = len(date_df[date_df['appears'] == True])
                rate = (cited / total * 100) if total > 0 else 0
                rates.append(rate)

            fig.add_trace(go.Bar(
                name=date_label,
                x=CATEGORY_ORDER,
                y=rates,
                marker_color=colors[i % len(colors)]
            ))

        fig.update_layout(
            title="Category Citation Rates by Check Date",
            xaxis_title="Category",
            yaxis_title="Citation Rate (%)",
            barmode='group',
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)


def render_query_status_changes(hist_df):
    """
    Render a table showing queries that changed status between check dates.

    Args:
        hist_df: DataFrame with historical data from Supabase
    """
    if hist_df.empty:
        st.info("No historical data available from Supabase.")
        return

    # Group by date (ignore time component)
    hist_df['check_date_only'] = hist_df['check_date'].dt.date
    dates = sorted(hist_df['check_date_only'].unique())

    if len(dates) < 2:
        st.info("Need at least 2 check dates to show changes.")
        return

    # Compare first and last dates (or last two if preferred)
    first_date = dates[0]
    last_date = dates[-1]

    first_df = hist_df[hist_df['check_date_only'] == first_date]
    last_df = hist_df[hist_df['check_date_only'] == last_date]

    first_label = pd.to_datetime(first_date).strftime('%b %d, %Y')
    last_label = pd.to_datetime(last_date).strftime('%b %d, %Y')

    changes = []
    queries = hist_df['query'].unique()

    for query in queries:
        first_row = first_df[first_df['query'] == query]
        last_row = last_df[last_df['query'] == query]

        if first_row.empty or last_row.empty:
            continue

        first_status = "Cited" if first_row.iloc[0]['appears'] else "Not Cited"
        last_status = "Cited" if last_row.iloc[0]['appears'] else "Not Cited"

        if first_status != last_status:
            if last_status == "Cited":
                change_type = "🟢 Gained"
            else:
                change_type = "🔴 Lost"

            changes.append({
                'Query': query,
                'Category': first_row.iloc[0].get('category', 'Unknown'),
                first_label: first_status,
                last_label: last_status,
                'Change': change_type
            })

    if not changes:
        st.success("No queries changed status between checks.")
        return

    changes_df = pd.DataFrame(changes)

    # Sort: losses first, then gains
    changes_df['sort_key'] = changes_df['Change'].apply(lambda x: 0 if 'Lost' in x else 1)
    changes_df = changes_df.sort_values('sort_key').drop('sort_key', axis=1)

    # Summary counts
    gained = len([c for c in changes if 'Gained' in c['Change']])
    lost = len([c for c in changes if 'Lost' in c['Change']])

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries Gained Citation", gained, delta=f"+{gained}" if gained > 0 else None)
    with col2:
        st.metric("Queries Lost Citation", lost, delta=f"-{lost}" if lost > 0 else None, delta_color="inverse")

    st.dataframe(changes_df, use_container_width=True, hide_index=True)


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
        page_title="AI Search Authority Tracker",
        page_icon="🎾",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    # Show login screen if not authenticated
    if not st.session_state.authenticated:
        st.title("🎾 AI Search Authority Tracker")
        st.subheader("Login Required")

        password = st.text_input("Password", type="password", key="password_input")

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            login_button = st.button("Login", use_container_width=True)

        if login_button:
            if password == st.secrets.get("auth_password", ""):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password")

        st.stop()  # Don't show any content below this if not authenticated

    # Apply WTA brand styling
    apply_wta_styling()

    # If authenticated, add logout button in sidebar
    with st.sidebar:
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.rerun()

    # Page Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {PAL_CHARCOAL} 0%, {PAL_TEAL} 100%);
                padding: 32px; border-radius: 12px; border-bottom: 3px solid {PAL_AMBER};
                margin-bottom: 24px;">
        <h1 style="color: {PAL_TEXT_LIGHT}; margin: 0 0 8px 0; font-size: 42px; font-weight: 700;">
            🎾 AI Search Authority Tracker
        </h1>
        <p style="color: {PAL_TEXT_LIGHT}; font-size: 18px; margin: 0; opacity: 0.9;">
            Track wtatennis.com's authority in AI-generated answers
        </p>
    </div>
    """, unsafe_allow_html=True)

    # About This Tool Section
    with st.expander("ℹ️ About This Tool"):
        st.markdown("""
        ### What We Track
        - Position in Perplexity's "authority set" (6 core sources used to generate AI answers)
        - NOT full web UI display (which shows 10-20 total sources)
        - Being in the top 6 = AI trusts you as a primary authority for that topic

        ### Why This Matters
        - Traditional SEO tracks findability in search results
        - AI Search Authority tracks trustworthiness in answer generation
        - As users shift to ChatGPT/Perplexity, this becomes the new SEO

        ### Understanding Positions
        - **Position 1-2:** Primary authority - AI's go-to source
        - **Position 3-4:** Supporting authority - trusted secondary source
        - **Position 5-6:** Referenced authority - included but not primary
        - **Not Found:** AI doesn't cite you for this query (content gap)

        ### About API vs. Web UI Differences
        This tracker uses Perplexity's official API, which returns the 6 "core sources"
        used to generate AI answers. The web UI displays additional sources (10-20 total)
        for user reference.

        We track the core sources because:
        - ✓ These are the sources AI actually uses to create answers
        - ✓ Being in this set indicates higher authority/trust
        - ✓ API-based tracking enables automation and trend analysis

        **Position 2/6 in API authority set > Position 8/13 in web UI display**
        """)

    # Configuration Section
    st.header("⚙️ Configuration")

    # Initialize Supabase client early - used for both data loading and saving
    supabase_client = get_supabase_client()

    # Load data: Supabase (primary) with CSV fallback
    if supabase_client:
        df = load_results_from_supabase(supabase_client)
        data_source = "supabase"
    else:
        df = load_results()
        data_source = "csv"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", len(ALL_QUERIES))

    with col2:
        api_status = "✅ Connected" if check_api_key() else "❌ Not Connected"
        st.metric("API Status", api_status)

    with col3:
        if not df.empty:
            last_check = df['timestamp'].max().strftime("%Y-%m-%d %H:%M")
            st.metric("Last Check", last_check)
        else:
            st.metric("Last Check", "Never")

    with col4:
        if not df.empty:
            latest = df[df['timestamp'] == df['timestamp'].max()]
            cited = len(latest[latest['appears'] == 'Yes'])
            cite_rate = (cited / len(latest) * 100)
            st.metric("Citation Rate", f"{cite_rate:.1f}%")
        else:
            st.metric("Citation Rate", "N/A")

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
        results, db_stats = run_weekly_check()

        st.success(f"✅ Check completed! Processed {len(results)} queries.")

        # Show database storage status
        if db_stats['enabled']:
            if db_stats['failed'] == 0:
                st.success(f"📦 Database: All {db_stats['success']} results saved to Supabase")
            else:
                st.warning(f"📦 Database: {db_stats['success']} saved, {db_stats['failed']} failed")
        else:
            st.info("📦 Database: Supabase not configured (results saved to CSV only)")

        # Reload results to show latest data (from same source)
        if supabase_client:
            df = load_results_from_supabase(supabase_client)
        else:
            df = load_results()

    # Display Executive Summary and Visualizations
    if not df.empty:
        # Add category column to df for visualizations
        df['category'] = df['query'].map(QUERY_CATEGORIES)

        render_executive_summary(df)

        st.markdown("---")
        st.subheader("📊 Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Position Distribution")
            render_position_distribution(df)

        with col2:
            st.markdown("### Category Performance")
            render_category_performance(df)

    st.divider()

    # Historical Trends Section (from Supabase)
    # supabase_client already initialized above
    hist_df = get_historical_data(supabase_client)

    if not hist_df.empty:
        with st.expander("📈 Historical Trends", expanded=True):
            st.markdown(f"""
            <div style="background: {PAL_CHARCOAL}; padding: 12px 16px; border-radius: 8px;
                        border-left: 4px solid {PAL_AMBER}; margin-bottom: 16px;">
                <span style="color: {PAL_TEXT_LIGHT}; font-size: 14px;">
                    📊 Showing data from <strong>{len(hist_df['check_date'].dt.date.unique())}</strong> check dates
                    ({len(hist_df)} total records)
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Tab layout for different visualizations
            tab1, tab2, tab3 = st.tabs(["📈 Citation Rate Trend", "📊 Category Comparison", "🔄 Status Changes"])

            with tab1:
                st.markdown("### Overall Citation Rate Over Time")
                render_historical_citation_trend(hist_df)

            with tab2:
                st.markdown("### Category Performance Across Check Dates")
                render_category_trend_comparison(hist_df)

            with tab3:
                st.markdown("### Queries That Changed Status")
                render_query_status_changes(hist_df)
    else:
        with st.expander("📈 Historical Trends", expanded=False):
            st.info("No historical data available. Configure Supabase to enable trend tracking across check dates.")

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
                "appears": "In Authority Set?",
                "position": "Authority Rank",
                "citation_url": st.column_config.LinkColumn("Source URL"),
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
