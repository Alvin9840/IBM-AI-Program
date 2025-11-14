"""NBA Data Tool - Multi-Team Support with Intelligent Caching

Optimized for AI agent usage with support for querying any NBA team.
"""

import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict

from nba_api.stats.endpoints import (
    leaguegamelog, playergamelog, commonteamroster, teamdetails,
    leaguestandings, teamyearbyyearstats
)
from nba_api.stats.static import teams


@dataclass
class PerformanceMetrics:
    win_rate: float
    wins: int
    losses: int
    avg_points: float
    avg_margin: float
    avg_fg_pct: float
    consistency: str


@dataclass
class CompetitiveContext:
    league_rank: int
    competitive_tier: str
    playoff_status: str
    win_pct: float
    games_back: float


class DataTool:
    """NBA data tool with multi-team support and intelligent caching."""
    
    CURRENT_SEASON = '2025-26'
    TTL_FRESH = 300
    TTL_SESSION = 3600

    def __init__(self, default_team_name: str = "Los Angeles Lakers"):
        self.default_team_name = default_team_name
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._team_lookup = {t['full_name']: t for t in teams.get_teams()}
        
        if default_team_name not in self._team_lookup:
            raise ValueError(f"Team '{default_team_name}' not found")
        
        self.default_team_id = self._team_lookup[default_team_name]['id']
        print(f"📊 DataTool initialized (default: {default_team_name})")
    
    def _resolve_team(self, team_name: Optional[str] = None) -> Tuple[str, int]:
        """Resolve team name to (name, id). Uses default if None."""
        if team_name is None:
            return self.default_team_name, self.default_team_id
        
        team_info = self._team_lookup.get(team_name)
        if not team_info:
            raise ValueError(f"Team '{team_name}' not found")
        return team_name, team_info['id']
    
    def _fetch_cached(self, key: str, fetch_fn: Callable, ttl: Optional[int]) -> Any:
        """Cache-or-fetch pattern with TTL expiry."""
        if key in self._cache:
            data, expiry = self._cache[key]
            if ttl is None or datetime.now() < expiry:
                return data
            del self._cache[key]
        
        data = fetch_fn()
        expiry = datetime.now() + timedelta(seconds=ttl) if ttl else datetime.max
        self._cache[key] = (data, expiry)
        return data
    
    def get_recent_games(self, num_games: int = 10, team_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent game results with detailed statistics.
        
        Args:
            num_games: Number of recent games to retrieve (will convert string to int if needed)
            team_name: Team name (uses default if None)
        """
        # Type conversion: handle string numbers from AI agents
        num_games = int(num_games) if isinstance(num_games, str) else num_games
        
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                game_log = leaguegamelog.LeagueGameLog(
                    season=self.CURRENT_SEASON,
                    season_type_all_star='Regular Season',
                    player_or_team_abbreviation='T'
                )
                df = game_log.get_data_frames()[0]
                team_games = df[df['TEAM_ID'] == team_id]
                
                if team_games.empty:
                    return []
                
                team_games = team_games.sort_values('GAME_DATE', ascending=False)
                return [
                    {
                        "game_id": r['GAME_ID'], "date": r['GAME_DATE'], "matchup": r['MATCHUP'],
                        "result": r['WL'], "team_points": int(r['PTS']),
                        "opponent_points": int(r['PTS']) - int(r['PLUS_MINUS']),
                        "plus_minus": int(r['PLUS_MINUS']), "field_goal_pct": float(r['FG_PCT']),
                        "three_point_pct": float(r['FG3_PCT']), "rebounds": int(r['REB']),
                        "assists": int(r['AST']), "turnovers": int(r['TOV'])
                    }
                    for _, r in team_games.head(num_games).iterrows()
                ]
            except Exception as e:
                print(f"⚠️ Error fetching games: {e}")
                return []
        
        return self._fetch_cached(f"games_{team_id}_{num_games}", fetch, self.TTL_FRESH)
    
    def get_team_win_streak(self, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current win/loss streak and recent record."""
        games = self.get_recent_games(20, team_name)
        if not games:
            return {"streak_type": "unknown", "streak_length": 0}
        
        current_result = games[0]['result']
        streak_length = sum(1 for g in games if g['result'] == current_result)
        recent_wins = sum(1 for g in games[:10] if g['result'] == 'W')
        
        return {
            "streak_type": "winning" if current_result == 'W' else "losing",
            "streak_length": streak_length,
            "recent_record": {"wins": recent_wins, "losses": 10 - recent_wins}
        }
    
    def get_standings(self, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current standings and playoff positioning."""
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                df = leaguestandings.LeagueStandings().get_data_frames()[0]
                team_row = df[df['TeamID'] == team_id]
                
                if team_row.empty:
                    return {}
                
                s = team_row.iloc[0]
                return {
                    "conference": s['Conference'], "league_rank": int(s['LeagueRank']),
                    "wins": int(s['WINS']), "losses": int(s['LOSSES']),
                    "win_pct": float(s['WinPCT']), "games_back": float(s['ConferenceGamesBack']),
                    "last_10": s['L10'], "streak": s['CurrentStreak']
                }
            except Exception as e:
                print(f"⚠️ Error fetching standings: {e}")
                return {}
        
        return self._fetch_cached(f"standings_{team_id}", fetch, self.TTL_FRESH)
    
    def get_team_roster(self, team_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current team roster."""
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                df = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
                return [
                    {
                        "player_id": r['PLAYER_ID'], "player_name": r['PLAYER'],
                        "position": r['POSITION'], "jersey_number": r['NUM'],
                        "age": r['AGE'], "experience": r['EXP']
                    }
                    for _, r in df.iterrows()
                ]
            except Exception as e:
                print(f"⚠️ Error fetching roster: {e}")
                return []
        
        return self._fetch_cached(f"roster_{team_id}", fetch, self.TTL_SESSION)
    
    def get_top_performers(self, num_games: int = 5, team_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get top performing players with recent statistical averages.
        
        Args:
            num_games: Number of recent games to analyze (will convert string to int if needed)
            team_name: Team name (uses default if None)
        """
        # Type conversion: handle string numbers from AI agents
        num_games = int(num_games) if isinstance(num_games, str) else num_games
        
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                roster = self.get_team_roster(team_name)
                performers = []
                
                for player in roster[:10]:
                    try:
                        df = playergamelog.PlayerGameLog(
                            player_id=player['player_id'], season=self.CURRENT_SEASON
                        ).get_data_frames()[0]
                        
                        if not df.empty:
                            recent = df.head(num_games)
                            performers.append({
                                "player_name": player['player_name'], "position": player['position'],
                                "avg_points": round(recent['PTS'].mean(), 1),
                                "avg_assists": round(recent['AST'].mean(), 1),
                                "avg_rebounds": round(recent['REB'].mean(), 1)
                            })
                    except:
                        continue
                
                return sorted(performers, key=lambda x: x['avg_points'], reverse=True)[:5]
            except Exception as e:
                print(f"⚠️ Error fetching performers: {e}")
                return []
        
        return self._fetch_cached(f"performers_{team_id}_{num_games}", fetch, self.TTL_SESSION)
    
    def get_team_details(self, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Get team details (arena, coach, ownership, etc.)."""
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                df = teamdetails.TeamDetails(team_id=team_id).get_data_frames()[0]
                if df.empty:
                    return {}
                
                t = df.iloc[0]
                return {
                    "team_name": f"{t['CITY']} {t['NICKNAME']}", "abbreviation": t['ABBREVIATION'],
                    "city": t['CITY'], "arena": t['ARENA'], "arena_capacity": int(t['ARENACAPACITY']),
                    "head_coach": t['HEADCOACH'], "general_manager": t['GENERALMANAGER'],
                    "owner": t['OWNER'], "year_founded": int(t['YEARFOUNDED']),
                    "d_league_affiliation": t['DLEAGUEAFFILIATION']
                }
            except Exception as e:
                print(f"⚠️ Error fetching details: {e}")
                return {}
        
        return self._fetch_cached(f"details_{team_id}", fetch, 86400)
    
    def get_season_stats(self, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current season overall statistics."""
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                df = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id).get_data_frames()[0]
                current = df[df['YEAR'] == self.CURRENT_SEASON]
                
                if current.empty:
                    return {}
                
                s = current.iloc[0]
                return {
                    "season": self.CURRENT_SEASON, "wins": int(s['WINS']),
                    "losses": int(s['LOSSES']), "win_pct": float(s['WIN_PCT']),
                    "conf_rank": int(s['CONF_RANK'])
                }
            except Exception as e:
                print(f"⚠️ Error fetching season stats: {e}")
                return {}
        
        return self._fetch_cached(f"season_{team_id}", fetch, 86400)
    
    def get_historical_performance(self, num_seasons: int = 3, team_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get historical performance across multiple seasons.
        
        Args:
            num_seasons: Number of seasons to retrieve (will convert string to int if needed)
            team_name: Team name (uses default if None)
        """
        # Type conversion: handle string numbers from AI agents
        num_seasons = int(num_seasons) if isinstance(num_seasons, str) else num_seasons
        
        _, team_id = self._resolve_team(team_name)
        
        def fetch():
            try:
                df = teamyearbyyearstats.TeamYearByYearStats(team_id=team_id).get_data_frames()[0]
                return [
                    {
                        "season": r['YEAR'], "wins": int(r['WINS']), "losses": int(r['LOSSES']),
                        "win_pct": float(r['WIN_PCT']), "made_playoffs": r['PO_WINS'] > 0
                    }
                    for _, r in df.head(num_seasons).iterrows()
                ]
            except Exception as e:
                print(f"⚠️ Error fetching history: {e}")
                return []
        
        return self._fetch_cached(f"history_{team_id}_{num_seasons}", fetch, None)
    
    def analyze_performance_trends(self, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze performance trends comparing recent vs previous 10 games."""
        games = self.get_recent_games(20, team_name)
        if not games:
            return {}
        
        prev, recent = games[10:], games[:10]
        prev_wins = sum(1 for g in prev if g['result'] == 'W')
        recent_wins = sum(1 for g in recent if g['result'] == 'W')
        prev_avg = statistics.mean(g['team_points'] for g in prev)
        recent_avg = statistics.mean(g['team_points'] for g in recent)
        
        return {
            "trend": "improving" if recent_wins > prev_wins else "declining",
            "recent_win_rate": recent_wins / 10,
            "previous_win_rate": prev_wins / 10,
            "scoring_trend": {
                "recent_avg": round(recent_avg, 1),
                "previous_avg": round(prev_avg, 1),
                "change": round(recent_avg - prev_avg, 1)
            },
            "momentum": "positive" if recent_wins > prev_wins and recent_avg > prev_avg else "negative"
        }
    
    def get_performance_metrics(self, num_games: int = 20, team_name: Optional[str] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            num_games: Number of games to analyze (will convert string to int if needed)
            team_name: Team name (uses default if None)
        """
        # Type conversion: handle string numbers from AI agents
        num_games = int(num_games) if isinstance(num_games, str) else num_games
        
        games = self.get_recent_games(num_games, team_name)
        if not games:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, "unknown")
        
        wins = sum(1 for g in games if g['result'] == 'W')
        points = [g['team_points'] for g in games]
        margins = [g['team_points'] - g['opponent_points'] for g in games]
        fg_pcts = [g['field_goal_pct'] for g in games]
        
        return PerformanceMetrics(
            win_rate=round(wins / len(games), 3),
            wins=wins,
            losses=len(games) - wins,
            avg_points=round(statistics.mean(points), 1),
            avg_margin=round(statistics.mean(margins), 1),
            avg_fg_pct=round(statistics.mean(fg_pcts) * 100, 1),
            consistency="high" if statistics.stdev(points) < 8 else "moderate"
        )
    
    def get_competitive_context(self, team_name: Optional[str] = None) -> CompetitiveContext:
        """Get competitive positioning and playoff status classification."""
        standings = self.get_standings(team_name)
        if not standings:
            return CompetitiveContext(15, "unknown", "unknown", 0.5, 0)
        
        rank = standings['league_rank']
        tier, status = (
            ("championship_contender", "guaranteed") if rank <= 6 else
            ("play_in_team", "likely") if rank <= 10 else
            ("rebuild_mode", "unlikely")
        )
        
        return CompetitiveContext(
            league_rank=rank,
            competitive_tier=tier,
            playoff_status=status,
            win_pct=standings['win_pct'],
            games_back=standings['games_back']
        )
    
    def calculate_momentum_score(self, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Calculate comprehensive momentum score (0-100)."""
        trends = self.analyze_performance_trends(team_name)
        streak = self.get_team_win_streak(team_name)
        
        score = 50.0 + (trends.get("recent_win_rate", 0.5) - 0.5) * 60
        
        streak_type = streak.get("streak_type", "")
        streak_len = streak.get("streak_length", 0)
        streak_impact = min(streak_len * 3, 20)
        score += streak_impact if streak_type == "winning" else -streak_impact
        score += {"positive": 10, "negative": -10}.get(trends.get("momentum"), 0)
        score = max(0, min(100, score))
        
        sentiment = (
            "excellent" if score >= 75 else
            "positive" if score >= 60 else
            "neutral" if score >= 40 else
            "concerning"
        )
        
        return {
            "score": round(score, 1),
            "sentiment": sentiment,
            "trend": trends.get("trend", "stable")
        }
    
    def fetch_all_data(self, force_refresh: bool = False, team_name: Optional[str] = None) -> Dict[str, Any]:
        """Bulk fetch all available data for a team."""
        if force_refresh:
            self._cache.clear()
        
        resolved_name, _ = self._resolve_team(team_name)
        print(f"📥 Fetching comprehensive data for {resolved_name}...")
        
        data = {
            "team_info": self.get_team_details(team_name),
            "recent_performance": {
                "recent_games": self.get_recent_games(10, team_name),
                "win_streak": self.get_team_win_streak(team_name),
                "top_performers": self.get_top_performers(5, team_name)
            },
            "season_data": {
                "current_season": self.get_season_stats(team_name),
                "standings": self.get_standings(team_name),
                "historical": self.get_historical_performance(3, team_name)
            },
            "trends": self.analyze_performance_trends(team_name),
            "metrics": asdict(self.get_performance_metrics(20, team_name)),
            "competitive_context": asdict(self.get_competitive_context(team_name)),
            "momentum": self.calculate_momentum_score(team_name),
            "roster": self.get_team_roster(team_name)
        }
        
        print("✅ Complete!")
        return data
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        valid = sum(1 for _, (_, exp) in self._cache.items() if datetime.now() < exp)
        return {"total_cached": len(self._cache), "valid_entries": valid}
    
    def get_all_nba_teams(self) -> List[Dict[str, Any]]:
        """Get list of all NBA teams with their full names and abbreviations.
        
        Returns:
            List of dictionaries containing team information (id, full_name, abbreviation, city, nickname)
        """
        return [
            {
                "id": team['id'],
                "full_name": team['full_name'],
                "abbreviation": team['abbreviation'],
                "city": team['city'],
                "nickname": team['nickname']
            }
            for team in teams.get_teams()
        ]
    
    def get_teams_by_rank(self, start_rank: int = 1, end_rank: int = 5) -> List[Dict[str, Any]]:
        """Get teams within a specific league rank range with their key performance data.
        
        Args:
            start_rank: Starting league rank (1 = best team)
            end_rank: Ending league rank (inclusive)
        
        Returns:
            List of team dictionaries sorted by rank, each containing:
            - rank, team_name, wins, losses, win_pct, conference
            - performance metrics (win_rate, avg_points, avg_margin)
            - momentum score and trend
        """
        # Convert to int if string was passed
        start_rank = int(start_rank) if isinstance(start_rank, str) else start_rank
        end_rank = int(end_rank) if isinstance(end_rank, str) else end_rank
        
        # Validate inputs
        if start_rank < 1:
            start_rank = 1
        if end_rank > 30:
            end_rank = 30
        if start_rank > end_rank:
            start_rank, end_rank = end_rank, start_rank
        
        def fetch():
            try:
                # Get all standings
                df = leaguestandings.LeagueStandings().get_data_frames()[0]
                
                # Sort by league rank and filter
                df = df.sort_values('LeagueRank')
                filtered = df[(df['LeagueRank'] >= start_rank) & (df['LeagueRank'] <= end_rank)]
                
                teams_data = []
                for _, row in filtered.iterrows():
                    team_name = f"{row['TeamCity']} {row['TeamName']}"
                    
                    # Get performance metrics for this team
                    try:
                        metrics = self.get_performance_metrics(num_games=20, team_name=team_name)
                        momentum = self.calculate_momentum_score(team_name=team_name)
                        trends = self.analyze_performance_trends(team_name=team_name)
                        
                        teams_data.append({
                            "rank": int(row['LeagueRank']),
                            "team_name": team_name,
                            "wins": int(row['WINS']),
                            "losses": int(row['LOSSES']),
                            "win_pct": float(row['WinPCT']),
                            "conference": row['Conference'],
                            "games_back": float(row['ConferenceGamesBack']),
                            "performance": {
                                "win_rate": metrics.win_rate,
                                "avg_points": metrics.avg_points,
                                "avg_margin": metrics.avg_margin,
                                "avg_fg_pct": metrics.avg_fg_pct,
                                "consistency": metrics.consistency
                            },
                            "momentum": {
                                "score": momentum['score'],
                                "sentiment": momentum['sentiment'],
                                "trend": momentum['trend']
                            },
                            "recent_trend": trends.get('trend', 'stable')
                        })
                    except Exception as e:
                        print(f"⚠️ Error getting detailed stats for {team_name}: {e}")
                        # Add basic info even if detailed stats fail
                        teams_data.append({
                            "rank": int(row['LeagueRank']),
                            "team_name": team_name,
                            "wins": int(row['WINS']),
                            "losses": int(row['LOSSES']),
                            "win_pct": float(row['WinPCT']),
                            "conference": row['Conference'],
                            "games_back": float(row['ConferenceGamesBack'])
                        })
                
                return teams_data
                
            except Exception as e:
                print(f"⚠️ Error fetching teams by rank: {e}")
                return []
        
        cache_key = f"teams_rank_{start_rank}_{end_rank}"
        return self._fetch_cached(cache_key, fetch, self.TTL_FRESH)
    
    def clear_cache(self, team_name: Optional[str] = None) -> None:
        """Clear cached data."""
        if team_name:
            _, team_id = self._resolve_team(team_name)
            keys_to_remove = [k for k in self._cache if str(team_id) in k]
            for key in keys_to_remove:
                del self._cache[key]
            print(f"🧹 Cache cleared for {team_name}")
        else:
            self._cache.clear()
            print("🧹 All cache cleared")