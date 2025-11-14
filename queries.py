"""
WTA AI Ranking Tracker - Query List
Total: 41 queries organized by category
"""

# WTA Brand variants (6 queries)
WTA_BRAND_QUERIES = [
    "WTA",
    "WTA Tennis",
    "Women's Tennis Association",
    "wtatennis",
    "WTA Tour",
    "WTA Unlocked"
]

# Top 10 players - Full names (10 queries)
PLAYER_FULL_NAME_QUERIES = [
    "Aryna Sabalenka",
    "Iga Swiatek",
    "Coco Gauff",
    "Amanda Anisimova",
    "Elena Rybakina",
    "Jessica Pegula",
    "Madison Keys",
    "Jasmine Paolini",
    "Mirra Andreeva",
    "Ekaterina Alexandrova"
]

# Top 10 players - Last names only (10 queries)
PLAYER_LAST_NAME_QUERIES = [
    "Sabalenka",
    "Swiatek",
    "Gauff",
    "Anisimova",
    "Rybakina",
    "Pegula",
    "Keys",
    "Paolini",
    "Andreeva",
    "Alexandrova"
]

# Generic queries (15 queries)
GENERIC_QUERIES = [
    "how to watch tennis",
    "women's tennis rankings",
    "best female tennis players",
    "tennis grand slam winners women",
    "women's tennis tournaments 2025",
    "WTA schedule",
    "tennis streaming",
    "women's tennis live scores",
    "tennis player rankings",
    "professional women's tennis",
    "tennis tournaments today",
    "women's tennis news",
    "tennis grand slams",
    "tennis ATP WTA",
    "world tennis rankings"
]

# Combined list of all queries
ALL_QUERIES = (
    WTA_BRAND_QUERIES +
    PLAYER_FULL_NAME_QUERIES +
    PLAYER_LAST_NAME_QUERIES +
    GENERIC_QUERIES
)

# Verify total count
assert len(ALL_QUERIES) == 41, f"Expected 41 queries, got {len(ALL_QUERIES)}"

# Category mapping for analysis
QUERY_CATEGORIES = {}
for query in WTA_BRAND_QUERIES:
    QUERY_CATEGORIES[query] = "WTA Brand"
for query in PLAYER_FULL_NAME_QUERIES:
    QUERY_CATEGORIES[query] = "Player Full Name"
for query in PLAYER_LAST_NAME_QUERIES:
    QUERY_CATEGORIES[query] = "Player Last Name"
for query in GENERIC_QUERIES:
    QUERY_CATEGORIES[query] = "Generic"
