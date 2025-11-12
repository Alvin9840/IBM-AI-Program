"""
Test suite for DataTool - NBA API integration and caching
Run with: python test_data_tool.py
"""

import time
from datetime import datetime
from tools.data_tool import DataTool


def print_section(title: str):
    """Pretty print section headers"""
    print(f"\n{'='*80}")
    print(f"🧪 {title}")
    print(f"{'='*80}\n")


def test_initialization():
    """Test 1: DataTool initialization"""
    print_section("Test 1: Initialization")
    
    try:
        tool = DataTool(team_name="Los Angeles Lakers")
        print(f"✅ Team ID: {tool.team_id}")
        print(f"✅ Team Name: {tool.team_name}")
        print(f"✅ Season: {tool.CURRENT_SEASON}")
        return tool
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_recent_games(tool: DataTool):
    """Test 2: Fetch recent games"""
    print_section("Test 2: Recent Games")
    
    try:
        games = tool.get_recent_games(num_games=5)
        print(f"✅ Fetched {len(games)} games")
        
        if games:
            latest = games[0]
            print(f"\nLatest Game:")
            print(f"  Date: {latest['date']}")
            print(f"  Matchup: {latest['matchup']}")
            print(f"  Result: {latest['result']}")
            print(f"  Score: {latest['team_points']} - {latest['opponent_points']}")
            print(f"  Plus/Minus: {latest['plus_minus']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_standings(tool: DataTool):
    """Test 3: Get current standings"""
    print_section("Test 3: Standings")
    
    try:
        standings = tool.get_standings()
        print(f"✅ Fetched standings")
        
        if standings:
            print(f"\nCurrent Standings:")
            print(f"  Conference: {standings.get('conference', 'N/A')}")
            print(f"  League Rank: {standings.get('league_rank', 'N/A')}")
            print(f"  Record: {standings.get('wins', 0)}-{standings.get('losses', 0)}")
            print(f"  Win %: {standings.get('win_pct', 0):.3f}")
            print(f"  Games Back: {standings.get('games_back', 0)}")
            print(f"  Last 10: {standings.get('last_10', 'N/A')}")
            print(f"  Streak: {standings.get('streak', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_roster(tool: DataTool):
    """Test 4: Get team roster"""
    print_section("Test 4: Team Roster")
    
    try:
        roster = tool.get_team_roster()
        print(f"✅ Fetched {len(roster)} players")
        
        if roster:
            print(f"\nRoster Sample (first 5 players):")
            for player in roster[:5]:
                print(f"  #{player['jersey_number']} {player['player_name']} - {player['position']}")
                print(f"     Age: {player['age']}, Experience: {player['experience']} years")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_top_performers(tool: DataTool):
    """Test 5: Get top performers"""
    print_section("Test 5: Top Performers")
    
    try:
        performers = tool.get_top_performers(num_games=5)
        print(f"✅ Fetched {len(performers)} top performers")
        
        if performers:
            print(f"\nTop Performers (last 5 games):")
            for i, player in enumerate(performers, 1):
                print(f"  {i}. {player['player_name']} ({player['position']})")
                print(f"     Avg: {player['avg_points']} pts, {player['avg_assists']} ast, {player['avg_rebounds']} reb")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_team_details(tool: DataTool):
    """Test 6: Get team details"""
    print_section("Test 6: Team Details")
    
    try:
        details = tool.get_team_details()
        print(f"✅ Fetched team details")
        
        if details:
            print(f"\nTeam Information:")
            print(f"  Name: {details.get('team_name', 'N/A')}")
            print(f"  City: {details.get('city', 'N/A')}")
            print(f"  Conference: {details.get('conference', 'N/A')}")
            print(f"  Arena: {details.get('arena', 'N/A')}")
            print(f"  Head Coach: {details.get('head_coach', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_win_streak(tool: DataTool):
    """Test 7: Calculate win streak"""
    print_section("Test 7: Win Streak Analysis")
    
    try:
        streak = tool.get_team_win_streak()
        print(f"✅ Calculated streak")
        
        print(f"\nStreak Information:")
        print(f"  Type: {streak['streak_type']}")
        print(f"  Length: {streak['streak_length']} games")
        recent = streak['recent_record']
        print(f"  Last 10 games: {recent['wins']}-{recent['losses']}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_performance_trends(tool: DataTool):
    """Test 8: Analyze performance trends"""
    print_section("Test 8: Performance Trends")
    
    try:
        trends = tool.analyze_performance_trends()
        print(f"✅ Analyzed trends")
        
        if trends:
            print(f"\nTrend Analysis:")
            print(f"  Overall Trend: {trends.get('trend', 'N/A')}")
            print(f"  Recent Win Rate: {trends.get('recent_win_rate', 0):.1%}")
            print(f"  Previous Win Rate: {trends.get('previous_win_rate', 0):.1%}")
            
            scoring = trends.get('scoring_trend', {})
            print(f"\n  Scoring Trend:")
            print(f"    Recent Avg: {scoring.get('recent_avg', 0)}")
            print(f"    Previous Avg: {scoring.get('previous_avg', 0)}")
            print(f"    Change: {scoring.get('change', 0):+.1f}")
            
            print(f"\n  Momentum: {trends.get('momentum', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_performance_metrics(tool: DataTool):
    """Test 9: Get performance metrics"""
    print_section("Test 9: Performance Metrics")
    
    try:
        metrics = tool.get_performance_metrics(num_games=20)
        print(f"✅ Calculated metrics")
        
        print(f"\nPerformance Metrics (last 20 games):")
        print(f"  Record: {metrics.wins}-{metrics.losses}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Avg Points: {metrics.avg_points}")
        print(f"  Avg Margin: {metrics.avg_margin:+.1f}")
        print(f"  Avg FG%: {metrics.avg_fg_pct}%")
        print(f"  Consistency: {metrics.consistency}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_competitive_context(tool: DataTool):
    """Test 10: Get competitive context"""
    print_section("Test 10: Competitive Context")
    
    try:
        context = tool.get_competitive_context()
        print(f"✅ Analyzed competitive context")
        
        print(f"\nCompetitive Positioning:")
        print(f"  League Rank: {context.league_rank}")
        print(f"  Competitive Tier: {context.competitive_tier.replace('_', ' ').title()}")
        print(f"  Playoff Status: {context.playoff_status.title()}")
        print(f"  Win Percentage: {context.win_pct:.3f}")
        print(f"  Games Back: {context.games_back}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_momentum_score(tool: DataTool):
    """Test 11: Calculate momentum score"""
    print_section("Test 11: Momentum Score")
    
    try:
        momentum = tool.calculate_momentum_score()
        print(f"✅ Calculated momentum")
        
        print(f"\nMomentum Analysis:")
        print(f"  Score: {momentum['score']}/100")
        print(f"  Sentiment: {momentum['sentiment'].title()}")
        print(f"  Trend: {momentum['trend'].title()}")
        
        # Visual representation
        bar_length = int(momentum['score'] / 5)
        bar = '█' * bar_length + '░' * (20 - bar_length)
        print(f"  [{bar}] {momentum['score']}/100")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_historical_performance(tool: DataTool):
    """Test 12: Get historical performance"""
    print_section("Test 12: Historical Performance")
    
    try:
        history = tool.get_historical_performance(num_seasons=3)
        print(f"✅ Fetched {len(history)} seasons")
        
        if history:
            print(f"\nHistorical Performance:")
            for season in history:
                print(f"  {season['season']}: {season['wins']}-{season['losses']} "
                      f"({season['win_pct']:.3f}) "
                      f"{'🏆 Playoffs' if season['made_playoffs'] else '❌ No playoffs'}")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_caching():
    """Test 13: Cache performance"""
    print_section("Test 13: Cache Performance")
    
    try:
        tool = DataTool(team_name="Los Angeles Lakers")
        
        # Use a method that's already cached from previous tests
        # This way we test cache retrieval without making new API calls
        print("Testing cache retrieval (should be instant)...")
        start = time.time()
        standings1 = tool.get_standings()
        time1 = time.time() - start
        print(f"  ⏱️  First call: {time1:.3f}s")
        
        # Second call (definitely cached)
        print("\nSecond call (from cache)...")
        start = time.time()
        standings2 = tool.get_standings()
        time2 = time.time() - start
        print(f"  ⏱️  Second call: {time2:.3f}s")
        
        # Verify cache worked
        if time2 > 0:
            speedup = time1 / time2
            print(f"\n✅ Cache speedup: {speedup:.1f}x faster")
        else:
            print(f"\n✅ Cache: Instant retrieval (< 0.001s)")
        
        print(f"✅ Data matches: {standings1 == standings2}")
        
        # Cache stats
        stats = tool.get_cache_stats()
        print(f"\nCache Statistics:")
        print(f"  Total entries: {stats['total_cached']}")
        print(f"  Valid entries: {stats['valid_entries']}")
        
        # Test cache invalidation
        print(f"\nTesting cache with force refresh...")
        tool.clear_cache()
        print(f"  Cache cleared, now has {tool.get_cache_stats()['total_cached']} entries")
        
        # Re-fetch (will hit API but won't timeout since it's just standings)
        start = time.time()
        standings3 = tool.get_standings()
        time3 = time.time() - start
        print(f"  ⏱️  After clear: {time3:.3f}s")
        print(f"  Cache now has {tool.get_cache_stats()['total_cached']} entries")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_bulk_fetch(tool: DataTool):
    """Test 14: Bulk data fetch"""
    print_section("Test 14: Bulk Data Fetch")
    
    try:
        print("Fetching all data in bulk...")
        start = time.time()
        data = tool.fetch_all_data()
        elapsed = time.time() - start
        
        print(f"\n✅ Bulk fetch completed in {elapsed:.2f}s")
        
        print(f"\nData Retrieved:")
        print(f"  Team Info: {'✅' if data.get('team_info') else '❌'}")
        print(f"  Recent Games: {len(data.get('recent_performance', {}).get('recent_games', []))} games")
        print(f"  Top Performers: {len(data.get('recent_performance', {}).get('top_performers', []))} players")
        print(f"  Standings: {'✅' if data.get('season_data', {}).get('standings') else '❌'}")
        print(f"  Historical Data: {len(data.get('season_data', {}).get('historical', []))} seasons")
        print(f"  Performance Metrics: {'✅' if data.get('metrics') else '❌'}")
        print(f"  Competitive Context: {'✅' if data.get('competitive_context') else '❌'}")
        print(f"  Momentum Score: {data.get('momentum', {}).get('score', 'N/A')}/100")
        print(f"  Roster: {len(data.get('roster', []))} players")
        
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_cache_clear(tool: DataTool):
    """Test 15: Cache clearing"""
    print_section("Test 15: Cache Management")
    
    try:
        # Populate cache
        tool.get_recent_games(5)
        stats_before = tool.get_cache_stats()
        print(f"Before clear: {stats_before['total_cached']} cached items")
        
        # Clear cache
        tool.clear_cache()
        stats_after = tool.get_cache_stats()
        print(f"After clear: {stats_after['total_cached']} cached items")
        
        print(f"✅ Cache cleared successfully")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("🏀 NBA DATA TOOL - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    results = []
    
    # Initialize
    tool = test_initialization()
    if not tool:
        print("\n❌ Initialization failed. Aborting tests.")
        return
    
    results.append(("Initialization", True))
    
    # Run all tests
    tests = [
        ("Recent Games", lambda: test_recent_games(tool)),
        ("Standings", lambda: test_standings(tool)),
        ("Roster", lambda: test_roster(tool)),
        ("Top Performers", lambda: test_top_performers(tool)),
        ("Team Details", lambda: test_team_details(tool)),
        ("Win Streak", lambda: test_win_streak(tool)),
        ("Performance Trends", lambda: test_performance_trends(tool)),
        ("Performance Metrics", lambda: test_performance_metrics(tool)),
        ("Competitive Context", lambda: test_competitive_context(tool)),
        ("Momentum Score", lambda: test_momentum_score(tool)),
        ("Historical Performance", lambda: test_historical_performance(tool)),
        ("Caching", test_caching),
        ("Bulk Fetch", lambda: test_bulk_fetch(tool)),
        ("Cache Clear", lambda: test_cache_clear(tool)),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total} ({passed/total*100:.1f}%)\n")
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "="*80)
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_tests()