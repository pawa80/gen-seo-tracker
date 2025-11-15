# queries.py - Complete 150-query expansion
# Phase 2A: AI Search Authority Tracker

# Category 1: Brand Protection (6 queries)
BRAND_QUERIES = [
    "WTA",
    "WTA Tennis",
    "Women's Tennis Association",
    "wtatennis",
    "WTA Tour",
    "WTA Unlocked"
]

# Category 2: Top 20 Players - Full Names (20 queries)
PLAYER_FULL_NAMES = [
    "Aryna Sabalenka",
    "Iga Swiatek",
    "Coco Gauff",
    "Amanda Anisimova",
    "Elena Rybakina",
    "Jessica Pegula",
    "Madison Keys",
    "Jasmine Paolini",
    "Mirra Andreeva",
    "Ekaterina Alexandrova",
    "Qinwen Zheng",
    "Danielle Collins",
    "Beatriz Haddad Maia",
    "Daria Kasatkina",
    "Veronika Kudermetova",
    "Maria Sakkari",
    "Liudmila Samsonova",
    "Barbora Krejcikova",
    "Jelena Ostapenko",
    "Marketa Vondrousova"
]

# Category 3: Top 20 Players - Last Names (20 queries)
PLAYER_LAST_NAMES = [
    "Sabalenka",
    "Swiatek",
    "Gauff",
    "Anisimova",
    "Rybakina",
    "Pegula",
    "Keys",
    "Paolini",
    "Andreeva",
    "Alexandrova",
    "Zheng",
    "Collins",
    "Haddad Maia",
    "Kasatkina",
    "Kudermetova",
    "Sakkari",
    "Samsonova",
    "Krejcikova",
    "Ostapenko",
    "Vondrousova"
]

# Category 4: Grand Slam Tournaments (20 queries)
GRAND_SLAM_QUERIES = [
    "Australian Open women's singles",
    "Australian Open women's draw 2025",
    "Australian Open women's champion",
    "French Open women's singles",
    "French Open women's results",
    "Roland Garros women's draw",
    "Wimbledon women's singles",
    "Wimbledon women's champion",
    "Wimbledon women's draw 2025",
    "US Open women's singles",
    "US Open women's champion",
    "US Open women's results",
    "Grand Slam women's winners",
    "Grand Slam women's records",
    "women's tennis majors",
    "Grand Slam women's singles champions list",
    "most Grand Slam titles women",
    "youngest Grand Slam winner women",
    "oldest Grand Slam winner women",
    "Grand Slam prize money women"
]

# Category 5: WTA Tour Events (15 queries)
WTA_TOUR_QUERIES = [
    "WTA Finals",
    "WTA Finals schedule",
    "WTA Finals prize money",
    "WTA 1000 tournaments",
    "WTA 500 tournaments",
    "Indian Wells women's tennis",
    "Miami Open women's tennis",
    "Madrid Open women's tennis",
    "Rome Masters women's tennis",
    "Dubai Tennis Championships women",
    "Cincinnati Masters women",
    "Canada Masters women",
    "Billie Jean King Cup",
    "United Cup tennis",
    "WTA Tour schedule 2025"
]

# Category 6: Rankings & Statistics (15 queries)
RANKINGS_STATS_QUERIES = [
    "women's tennis rankings",
    "WTA rankings live",
    "tennis world rankings women",
    "WTA race to finals",
    "WTA ranking points",
    "WTA ranking system explained",
    "tennis elo ratings women",
    "WTA prize money rankings",
    "head to head Sabalenka Swiatek",
    "youngest number 1 WTA",
    "best tennis serve women",
    "fastest serve women's tennis",
    "most aces women's tennis",
    "best return of serve women's tennis",
    "most consecutive wins women's tennis"
]

# Category 7: How-To & Educational (15 queries)
EDUCATIONAL_QUERIES = [
    "how to watch tennis",
    "how to watch WTA tennis",
    "tennis streaming services",
    "where to watch tennis online",
    "WTA ranking points system",
    "how to qualify for WTA Finals",
    "tennis scoring system explained",
    "tennis rules women vs men",
    "how to become a professional tennis player",
    "WTA tennis academy",
    "tennis training programs women",
    "how to get into WTA Tour",
    "WTA qualifying rules",
    "tennis wildcards explained",
    "protected ranking tennis"
]

# Category 8: News & Trends (15 queries)
NEWS_TRENDS_QUERIES = [
    "women's tennis news",
    "WTA news today",
    "tennis injury news women",
    "WTA player of the month",
    "women's tennis upsets 2025",
    "WTA retirement news",
    "tennis comeback stories women",
    "WTA prize money increase",
    "equal pay tennis women",
    "women's tennis scheduling",
    "WTA tour changes",
    "tennis fashion women",
    "tennis equipment women players use",
    "tennis training innovations women",
    "women's tennis popularity"
]

# Category 9: Player Comparisons (12 queries)
COMPARISON_QUERIES = [
    "Swiatek vs Sabalenka comparison",
    "Swiatek vs Gauff stats",
    "best women tennis player 2025",
    "greatest female tennis players",
    "women's tennis GOAT debate",
    "Serena Williams vs Steffi Graf",
    "youngest top 10 player WTA",
    "oldest active WTA player",
    "tallest women tennis players",
    "left handed women tennis players",
    "two-handed backhand women",
    "best clay court player women"
]

# Category 10: Historical & Records (12 queries)
HISTORICAL_QUERIES = [
    "women's tennis history",
    "WTA tour history",
    "first WTA champion",
    "most WTA titles",
    "longest match women's tennis",
    "highest ranked women's tennis player",
    "most weeks at number 1 women",
    "tennis records women",
    "WTA prize money history",
    "women's tennis pioneers",
    "tennis legends women",
    "WTA hall of fame"
]

# Combine all queries
ALL_QUERIES = (
    BRAND_QUERIES +
    PLAYER_FULL_NAMES +
    PLAYER_LAST_NAMES +
    GRAND_SLAM_QUERIES +
    WTA_TOUR_QUERIES +
    RANKINGS_STATS_QUERIES +
    EDUCATIONAL_QUERIES +
    NEWS_TRENDS_QUERIES +
    COMPARISON_QUERIES +
    HISTORICAL_QUERIES
)

# Map each query to its category
QUERY_CATEGORIES = {}

for query in BRAND_QUERIES:
    QUERY_CATEGORIES[query] = "Brand Protection"

for query in PLAYER_FULL_NAMES:
    QUERY_CATEGORIES[query] = "Player Authority (Full Names)"

for query in PLAYER_LAST_NAMES:
    QUERY_CATEGORIES[query] = "Player Authority (Last Names)"

for query in GRAND_SLAM_QUERIES:
    QUERY_CATEGORIES[query] = "Grand Slam Tournaments"

for query in WTA_TOUR_QUERIES:
    QUERY_CATEGORIES[query] = "WTA Tour Events"

for query in RANKINGS_STATS_QUERIES:
    QUERY_CATEGORIES[query] = "Rankings & Statistics"

for query in EDUCATIONAL_QUERIES:
    QUERY_CATEGORIES[query] = "How-To & Educational"

for query in NEWS_TRENDS_QUERIES:
    QUERY_CATEGORIES[query] = "News & Trends"

for query in COMPARISON_QUERIES:
    QUERY_CATEGORIES[query] = "Player Comparisons"

for query in HISTORICAL_QUERIES:
    QUERY_CATEGORIES[query] = "Historical & Records"

# Category goals (for documentation/reporting)
CATEGORY_GOALS = {
    "Brand Protection": 100,  # 100% citation target
    "Player Authority (Full Names)": 95,
    "Player Authority (Last Names)": 90,
    "Grand Slam Tournaments": 85,
    "WTA Tour Events": 80,
    "Rankings & Statistics": 80,
    "How-To & Educational": 70,
    "News & Trends": 65,
    "Player Comparisons": 60,
    "Historical & Records": 70
}

# Category display order
CATEGORY_ORDER = [
    "Brand Protection",
    "Player Authority (Full Names)",
    "Player Authority (Last Names)",
    "Grand Slam Tournaments",
    "WTA Tour Events",
    "Rankings & Statistics",
    "How-To & Educational",
    "News & Trends",
    "Player Comparisons",
    "Historical & Records"
]

# Validation
print(f"Total queries: {len(ALL_QUERIES)}")
print(f"Total categories mapped: {len(QUERY_CATEGORIES)}")
print(f"\nCategory breakdown:")
for category in CATEGORY_ORDER:
    count = sum(1 for q in QUERY_CATEGORIES.values() if q == category)
    goal = CATEGORY_GOALS[category]
    print(f"  {category}: {count} queries (goal: {goal}%)")

# Check for duplicates
if len(ALL_QUERIES) != len(set(ALL_QUERIES)):
    duplicates = [q for q in ALL_QUERIES if ALL_QUERIES.count(q) > 1]
    print(f"\n⚠️  WARNING: Found duplicate queries: {set(duplicates)}")
else:
    print(f"\n✅ No duplicate queries found")
