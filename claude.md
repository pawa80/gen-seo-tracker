# Project: GEO Rank & Citation Tracker

## Overview
Tracks AI search engine citation visibility — monitors whether Perplexity (and future: ChatGPT, Google AI) cites a domain for specific queries. Currently WTA-focused. Part of the Search Intelligence Suite (PRCS009).

## Current Version
v2.1 — Auth + Supabase storage

## Live Deployment
https://gen-seo-tracker-dedj43axdprniwyv9tns5c.streamlit.app/
Password: 24$zUaY3_43sR

## Architecture
```
├── tracker.py           # Main Streamlit app (52KB — large file)
├── queries.py           # 149 queries across 10 categories
├── diagnose_supabase.py # Supabase diagnostic script
├── requirements.txt     # Dependencies
├── .streamlit/          # Streamlit config + secrets
├── .devcontainer/       # Dev container config
└── README.md            # Original MVP readme
```

## Tech Stack
- Python + Streamlit (web framework)
- Perplexity API (semantic search / citation checking)
- Supabase PostgreSQL (data storage)
- Streamlit Cloud (hosting, auto-deploys from main)
- GitHub: github.com/pawa80/gen-seo-tracker (private)

## Key Data
- 149 queries across 10 categories
- 745 historical records (WTA domain)
- 89% overall citation rate
- Educational category: 53% (content gap identified)
- 8 never-cited queries = content opportunities

## Key Discovery
Perplexity returns a 6-source "authority set" (what AI actually uses for answers) vs 13-source display list. Product reframed as "AI Authority Tracker" not "Citation Position Tracker."

## Current Blocker
Phase 3B Part 2 (historical data import) blocked by Windows Python dependency issues (pyroaring requires C++ Build Tools). 4 weeks of CSV data (590 rows) waiting to be imported to Supabase.

## Supabase
- **Project**: dxduneaizaxnynsmsvbx.supabase.co
- **Shared with**: AEO Agent (when it reaches v1.0)
- **Tables**: workspaces, workspace_members, projects, queries, geo_check_results (+ check_results from MVP)
- **RLS**: enabled on all tables with `user_in_workspace()` helper
- **GEO leads on schema** — AEO adopts this when adding persistence

## Suite Context
- Part of Search Intelligence Suite (PRCS007)
- Sibling: AEO Audit Agent (PRCS008)
- GEO = "Where do I show up?" (query portfolio view)
- AEO = "How do I fix this page?" (page fixer view)
- Handoff: GEO shows page not cited → user takes to AEO to fix
- Unified Dev Comms: https://www.notion.so/3049fa1ce4f5809a8b0bd45a287f4e06

## Notion Pages
- Product Page: https://www.notion.so/2f29fa1ce4f58095ac8efe75a0056764
- Product Spec: https://www.notion.so/2f29fa1ce4f581d28cbff670f0bda45f
- GEO Rolling Handover: https://www.notion.so/2fc9fa1ce4f580e5a411c004857ad78b

## Configuration
`.streamlit/secrets.toml`:
```toml
PERPLEXITY_API_KEY = "pplx-..."
SUPABASE_URL = "https://dxduneaizaxnynsmsvbx.supabase.co"
SUPABASE_KEY = "..."
password = "24$zUaY3_43sR"
```

## Git Workflow
Commit and push to main. Auto-deploys to Streamlit Cloud.

## Safety
- `v2.1-stable` tag on current state (pushed to remote 14/02/2026)

## Version History
- v1.0 (Nov 15, 2025): 41-query MVP
- v1.1 (Nov 22, 2025): 149 queries, category organisation
- v2.0 (Dec 6, 2025): Visualisations, brand styling ("Scandinavian Depth")
- v2.1 (Jan 9, 2026): Authentication, Supabase Week 5 data

## Roadmap
- v2.2: Historical data import (blocked)
- v3.0: Dynamic keywords (not hardcoded WTA) — may require new repo
- Future: Multi-engine tracking, GEO→AEO handoff, Quick Wins automation

## Rolling Handover
**Last session:** 14 Feb 2026 (pal-ops chat — Search Intelligence Suite master dev)
- Cloned repo locally, created v2.1-stable safety tag
- No code changes to GEO this session
- Posted sequencing plan to Unified Dev Comms: AEO v1.0 → GEO v3.0 → Suite MVP
- GEO's citation data (745 records) now feeds into AEO Agent's intelligence feed
- **Next**: Resolve historical import blocker, then dynamic keywords
