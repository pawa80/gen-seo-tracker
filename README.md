# WTA AI Ranking Tracker MVP

A tool to track how Perplexity AI cites wtatennis.com across different search queries. The tool runs manual weekly checks and tracks citation positions over time.

## Overview

This application queries Perplexity AI with 41 predefined searches related to WTA (Women's Tennis Association) and tracks whether wtatennis.com appears in the citations, at what position, and which specific URL is cited.

### Query Categories (41 Total)

- **WTA Brand Variants** (6 queries): WTA, WTA Tennis, Women's Tennis Association, etc.
- **Top 10 Players - Full Names** (10 queries): Aryna Sabalenka, Iga Swiatek, Coco Gauff, etc.
- **Top 10 Players - Last Names** (10 queries): Sabalenka, Swiatek, Gauff, etc.
- **Generic Tennis Queries** (15 queries): "women's tennis rankings", "WTA schedule", etc.

## Features

- ✅ Manual weekly check execution with progress tracking
- 📊 Real-time results display with filtering and sorting
- 📈 Historical data tracking and trend analysis
- 💾 CSV data export functionality
- ⚡ Rate limiting (1 query/second) to respect API limits
- 🎯 Citation position tracking (1-10)
- 🔍 Multi-domain matching (wtatennis.com, wta.com, and variants)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd wta-ai-tracker
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Perplexity API key:

```
PERPLEXITY_API_KEY=your_actual_api_key_here
```

**To get a Perplexity API key:**
1. Visit https://www.perplexity.ai/settings/api
2. Sign up or log in
3. Generate an API key
4. Copy the key to your `.env` file

### 4. Run the Dashboard

```bash
streamlit run tracker.py
```

The dashboard will open in your browser at http://localhost:8501

## How to Use

### Running a Check

1. Open the dashboard
2. Verify the API key status shows "✅ Connected"
3. Click the "▶️ Run Weekly Check" button
4. Wait approximately 45 seconds for all 41 queries to complete
5. View results in the "Latest Results" section

### Viewing Results

The dashboard displays:

- **Configuration**: Total queries, API status, and last check timestamp
- **Latest Results**: Sortable table with all queries from the most recent check
- **Historical Summary**: Appearance rate, always/never appearing queries, position changes
- **Export**: Download complete historical data as CSV

### Understanding the Results

- **Appears?**: "Yes" if wtatennis.com was found in citations, "No" otherwise
- **Position**: Citation rank (1-10) or "Not found"
- **Citation URL**: The specific wtatennis.com URL that was cited

## Data Storage

Results are stored in `results/results.csv` with the following format:

```csv
timestamp,query,appears,position,citation_url
2025-11-08 18:30:00,Aryna Sabalenka,Yes,3,https://www.wtatennis.com/players/...
2025-11-08 18:30:05,Iga Swiatek,No,Not found,
```

## Cost Estimate

Based on Perplexity API pricing:
- 41 queries per check
- 1 check per week = ~164 queries/month
- Estimated cost: £8-12/month

## File Structure

```
wta-ai-tracker/
├── tracker.py           # Main Streamlit dashboard
├── requirements.txt     # Python dependencies
├── queries.py          # Hardcoded query list (41 queries)
├── results/            # CSV files with historical results
│   └── results.csv     # Historical data (created on first run)
├── .env.example        # Example environment variables
├── .env               # Your actual API key (not committed to git)
└── README.md          # This file
```

## Technical Details

### Stack
- **Python 3.11+**
- **Streamlit**: Dashboard interface
- **Perplexity API**: AI search with citations
- **Pandas**: Data manipulation and analysis
- **CSV**: Persistent data storage

### API Model
Uses `llama-3.1-sonar-small-128k-online` for search queries with real-time web access.

### Domain Matching
Detects all variants:
- wtatennis.com
- wta.com
- www.wtatennis.com
- www.wta.com

### Rate Limiting
Enforces 1-second delay between queries to respect API limits and prevent rate limiting.

## Troubleshooting

### API Key Issues

**Problem**: "API Key Status: ❌ Not Connected"

**Solution**:
1. Ensure `.env` file exists in project root
2. Check that the file contains `PERPLEXITY_API_KEY=your_key`
3. Verify no extra spaces or quotes around the key
4. Restart the Streamlit app

### No Results Showing

**Problem**: Dashboard shows "No results yet"

**Solution**:
1. Click "Run Weekly Check" to execute your first check
2. Wait for the progress bar to complete
3. Results will appear automatically

### Import Errors

**Problem**: Module not found errors

**Solution**:
```bash
pip install -r requirements.txt
```

## Limitations (MVP)

This is an MVP (Minimum Viable Product) with intentional limitations:

- ❌ No automated scheduling (manual execution only)
- ❌ No query management UI (queries are hardcoded)
- ❌ No database (CSV storage only)
- ❌ No email/Slack alerts
- ❌ No competitor comparison
- ❌ No advanced visualizations

See ROADMAP.md for planned future features.

## Future Roadmap

### v2.0 (Planned)
- Automated weekly scheduling via GitHub Actions
- Query management UI (add/edit/delete queries)
- SQLite database migration

### v3.0 (Planned)
- Email/Slack alerts for ranking changes
- Multi-domain support
- Competitor comparison

### v4.0 (Planned)
- Trend visualization graphs
- PDF report generation
- Query categories and tags

## Support

For issues, questions, or feature requests, please open an issue in the repository.

## License

MIT License - See LICENSE file for details
