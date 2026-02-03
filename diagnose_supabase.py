#!/usr/bin/env python3
"""
Supabase Diagnostic Script for GEO Rank & Citation Tracker

Run this locally to diagnose Supabase connection and data issues.
Requires: pip install supabase python-dotenv

Usage:
    python diagnose_supabase.py

Or set environment variables:
    export SUPABASE_URL="https://your-project.supabase.co"
    export SUPABASE_KEY="your_supabase_anon_key"
    python diagnose_supabase.py
"""

import os
import sys
from datetime import datetime

try:
    from supabase import create_client
    import pandas as pd
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install supabase pandas")
    sys.exit(1)

# Try to load from .streamlit/secrets.toml or environment variables
def get_credentials():
    """Get Supabase credentials from various sources."""
    # Try environment variables first
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if url and key:
        return url, key

    # Try .streamlit/secrets.toml
    secrets_path = ".streamlit/secrets.toml"
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'supabase_url' in line and '=' in line:
                        url = line.split('=')[1].strip().strip('"').strip("'")
                    if 'supabase_key' in line and '=' in line:
                        key = line.split('=')[1].strip().strip('"').strip("'")
            if url and key:
                return url, key
        except Exception as e:
            print(f"Warning: Could not parse secrets.toml: {e}")

    return None, None


def run_diagnostics():
    """Run all diagnostic checks."""
    print("=" * 60)
    print("SUPABASE DIAGNOSTIC REPORT")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check 1: Get credentials
    print("\n[CHECK 1] Supabase Credentials")
    print("-" * 40)
    url, key = get_credentials()

    if not url or not key:
        print("❌ FAIL: Could not find Supabase credentials")
        print("   Set SUPABASE_URL and SUPABASE_KEY environment variables")
        print("   Or create .streamlit/secrets.toml with supabase_url and supabase_key")
        return

    print(f"✅ URL: {url[:40]}...")
    print(f"✅ Key: {key[:20]}...")

    # Check 2: Connect to Supabase
    print("\n[CHECK 2] Supabase Connection")
    print("-" * 40)
    try:
        client = create_client(url, key)
        print("✅ Connected to Supabase successfully")
    except Exception as e:
        print(f"❌ FAIL: Could not connect to Supabase: {e}")
        return

    # Check 3: Count total rows
    print("\n[CHECK 3] Total Row Count")
    print("-" * 40)
    try:
        response = client.table('check_results').select('*', count='exact').execute()
        total_count = len(response.data)
        print(f"✅ Total rows in check_results: {total_count}")

        if total_count == 0:
            print("⚠️  WARNING: Table is empty!")
            return
    except Exception as e:
        print(f"❌ FAIL: Could not query table: {e}")
        return

    # Check 4: Data by date
    print("\n[CHECK 4] Data by Check Date")
    print("-" * 40)
    try:
        df = pd.DataFrame(response.data)
        df['check_date'] = pd.to_datetime(df['check_date'])
        df['date_only'] = df['check_date'].dt.date

        date_counts = df.groupby('date_only').size().sort_index()
        print(f"{'Date':<15} {'Count':<10} {'Expected':<10} {'Status'}")
        print("-" * 50)

        for date, count in date_counts.items():
            expected = 149  # Based on 149 queries
            status = "✅" if count >= expected * 0.9 else "⚠️ INCOMPLETE"
            print(f"{str(date):<15} {count:<10} {expected:<10} {status}")

        print(f"\nTotal unique dates: {len(date_counts)}")
        print(f"Total rows: {len(df)}")
    except Exception as e:
        print(f"❌ FAIL: Could not analyze dates: {e}")

    # Check 5: Today's data
    print("\n[CHECK 5] Today's Data (2026-02-03)")
    print("-" * 40)
    try:
        today = datetime.now().date()
        today_df = df[df['date_only'] == today]
        print(f"Rows for today: {len(today_df)}")

        if len(today_df) > 0:
            cited = len(today_df[today_df['appears'] == True])
            print(f"Cited: {cited}/{len(today_df)} ({cited/len(today_df)*100:.1f}%)")
        else:
            print("⚠️  No data for today yet")
    except Exception as e:
        print(f"❌ FAIL: Could not check today's data: {e}")

    # Check 6: Citation rates by date
    print("\n[CHECK 6] Citation Rates by Date")
    print("-" * 40)
    try:
        print(f"{'Date':<15} {'Cited':<8} {'Total':<8} {'Rate'}")
        print("-" * 45)

        for date in sorted(df['date_only'].unique()):
            date_df = df[df['date_only'] == date]
            cited = len(date_df[date_df['appears'] == True])
            total = len(date_df)
            rate = (cited / total * 100) if total > 0 else 0
            print(f"{str(date):<15} {cited:<8} {total:<8} {rate:.1f}%")
    except Exception as e:
        print(f"❌ FAIL: Could not calculate citation rates: {e}")

    # Check 7: Check for duplicates
    print("\n[CHECK 7] Duplicate Check")
    print("-" * 40)
    try:
        # Check if there are duplicate (query, date) combinations
        df['query_date'] = df['query'] + '_' + df['date_only'].astype(str)
        duplicates = df[df.duplicated(subset=['query_date'], keep=False)]

        if len(duplicates) > 0:
            print(f"⚠️  Found {len(duplicates)} rows with duplicate (query, date) combinations")
            dup_counts = duplicates.groupby('query_date').size()
            print(f"   {len(dup_counts)} unique query/date pairs have duplicates")
        else:
            print("✅ No duplicate (query, date) combinations found")
    except Exception as e:
        print(f"❌ FAIL: Could not check duplicates: {e}")

    # Check 8: Schema check
    print("\n[CHECK 8] Table Schema")
    print("-" * 40)
    try:
        print("Columns found:")
        for col in df.columns:
            if col != 'query_date':  # Skip our computed column
                sample = df[col].iloc[0] if len(df) > 0 else None
                dtype = type(sample).__name__ if sample is not None else 'unknown'
                print(f"  - {col}: {dtype}")
    except Exception as e:
        print(f"❌ FAIL: Could not check schema: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
