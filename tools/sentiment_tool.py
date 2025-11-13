"""NBA Sentiment Tool - Twitter Sentiment Analysis Engine

Analyzes NBA fan sentiment from Twitter datasets to provide insights on fan emotions,
engagement patterns, and trending topics. Designed for integration with AI agents.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


@dataclass
class SentimentDistribution:
    positive: float
    negative: float
    neutral: float
    total_posts: int
    avg_polarity: float


@dataclass
class PlatformEngagement:
    platform: str
    total_posts: int
    avg_followers: float
    avg_retweets: float
    sentiment_breakdown: Dict[str, float]


@dataclass
class TopicAnalysis:
    topic: str
    mention_count: int
    avg_sentiment: float
    engagement_rate: float
    trending_score: float


@dataclass
class PlayerSentiment:
    player_name: str
    mention_count: int
    avg_polarity: float
    sentiment_category: str
    top_keywords: List[str]


class SentimentTool:
    """Production-grade sentiment analysis tool for NBA Twitter data."""
    
    CACHE_TTL_SECONDS = 3600
    POLARITY_THRESHOLDS = {'positive': 0.1, 'negative': -0.1}
    MIN_ENTITY_MENTIONS = 5
    MIN_PLAYER_MENTIONS = 3
    
    def __init__(self, data_dir: str = "data/sentiment_dataset"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._load_datasets()
        print(f"📱 Sentiment Tool initialized with {len(self.datasets)} datasets")
    
    def _load_datasets(self) -> None:
        """Load all CSV files from sentiment dataset directory."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Sentiment dataset directory not found: {self.data_dir}")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    df.columns = df.columns.str.lower().str.strip()
                    self.datasets[csv_file.stem] = df
                    print(f"  ✓ Loaded {csv_file.name}: {len(df)} records")
            except Exception as e:
                print(f"  ⚠️ Failed to load {csv_file.name}: {e}")
        
        if not self.datasets:
            raise ValueError("No valid datasets loaded")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Retrieve cached result if valid."""
        if key not in self._cache:
            return None
        
        data, timestamp = self._cache[key]
        if (datetime.now() - timestamp).total_seconds() < self.CACHE_TTL_SECONDS:
            return data
        
        del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any) -> None:
        """Store result in cache with timestamp."""
        self._cache[key] = (data, datetime.now())
    
    def _get_dataframe(self, dataset_name: Optional[str]) -> pd.DataFrame:
        """Get dataframe for analysis - single dataset or concatenated."""
        if dataset_name and dataset_name in self.datasets:
            return self.datasets[dataset_name]
        return pd.concat(self.datasets.values(), ignore_index=True)
    
    def _categorize_sentiment(self, polarity: float) -> str:
        """Categorize sentiment based on polarity score."""
        if polarity >= self.POLARITY_THRESHOLDS['positive']:
            return 'positive'
        elif polarity <= self.POLARITY_THRESHOLDS['negative']:
            return 'negative'
        return 'neutral'
    
    def _calculate_sentiment_breakdown(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate percentage breakdown of sentiment categories."""
        df_clean = df.dropna(subset=['polarity'])
        if df_clean.empty:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        sentiments = df_clean['polarity'].apply(self._categorize_sentiment)
        sentiment_counts = sentiments.value_counts()
        total = len(df_clean)
        
        return {
            'positive': round((sentiment_counts.get('positive', 0) / total) * 100, 2),
            'negative': round((sentiment_counts.get('negative', 0) / total) * 100, 2),
            'neutral': round((sentiment_counts.get('neutral', 0) / total) * 100, 2)
        }
    
    def _safe_column_mean(self, df: pd.DataFrame, column: str, default: float = 0.0) -> float:
        """Safely calculate mean for a column, returning default if missing."""
        return float(df[column].mean()) if column in df.columns else default
    
    def get_overall_sentiment(self, dataset_name: Optional[str] = None,
                             timeframe: Optional[str] = None) -> SentimentDistribution:
        """Calculate overall sentiment distribution across tweets."""
        cache_key = f"overall_{dataset_name}_{timeframe}"
        if cached := self._get_cached(cache_key):
            return cached
        
        df = self._get_dataframe(dataset_name)
        if 'polarity' not in df.columns:
            raise ValueError("Dataset missing 'polarity' column")
        
        df_clean = df.dropna(subset=['polarity'])
        breakdown = self._calculate_sentiment_breakdown(df)
        
        result = SentimentDistribution(
            positive=breakdown['positive'],
            negative=breakdown['negative'],
            neutral=breakdown['neutral'],
            total_posts=len(df_clean),
            avg_polarity=round(float(df_clean['polarity'].mean()), 4)
        )
        
        self._set_cache(cache_key, result)
        return result
    
    def get_platform_breakdown(self, dataset_name: Optional[str] = None) -> List[PlatformEngagement]:
        """Analyze engagement metrics by platform/partition."""
        cache_key = f"platform_{dataset_name}"
        if cached := self._get_cached(cache_key):
            return cached
        
        df = self._get_dataframe(dataset_name)
        if not all(col in df.columns for col in ['partition_0', 'polarity']):
            return []
        
        platforms = []
        for platform, group in df.groupby('partition_0'):
            group_clean = group.dropna(subset=['polarity'])
            if group_clean.empty:
                continue
            
            breakdown = self._calculate_sentiment_breakdown(group)
            
            platforms.append(PlatformEngagement(
                platform=str(platform),
                total_posts=len(group_clean),
                avg_followers=round(self._safe_column_mean(group, 'followers'), 0),
                avg_retweets=round(self._safe_column_mean(group, 'retweet_count'), 1),
                sentiment_breakdown={k: round(v, 1) for k, v in breakdown.items()}
            ))
        
        platforms.sort(key=lambda x: x.total_posts, reverse=True)
        self._set_cache(cache_key, platforms)
        return platforms
    
    def get_top_entities(self, entity_type: str = "teams", 
                        limit: int = 10,
                        dataset_name: Optional[str] = None) -> List[TopicAnalysis]:
        """Get top mentioned entities (teams, players, locations) with sentiment."""
        cache_key = f"entities_{entity_type}_{limit}_{dataset_name}"
        if cached := self._get_cached(cache_key):
            return cached
        
        df = self._get_dataframe(dataset_name)
        
        column_map = {
            'teams': 'partition_1',
            'players': 'screenname',
            'locations': 'location'
        }
        
        col_name = column_map.get(entity_type)
        if not col_name or col_name not in df.columns:
            return []
        
        df_clean = df.dropna(subset=[col_name, 'polarity'])
        topics = []
        
        for entity, group in df_clean.groupby(col_name):
            if pd.isna(entity) or entity == '' or len(group) < self.MIN_ENTITY_MENTIONS:
                continue
            
            avg_engagement = self._safe_column_mean(group, 'retweet_count')
            trending_score = (len(group) * 0.6) + (avg_engagement * 0.4)
            
            topics.append(TopicAnalysis(
                topic=str(entity),
                mention_count=len(group),
                avg_sentiment=round(float(group['polarity'].mean()), 4),
                engagement_rate=round(avg_engagement, 2),
                trending_score=round(trending_score, 2)
            ))
        
        topics.sort(key=lambda x: x.trending_score, reverse=True)
        result = topics[:limit]
        
        self._set_cache(cache_key, result)
        return result
    
    def analyze_player_sentiment(self, player_name: Optional[str] = None,
                                dataset_name: Optional[str] = None) -> List[PlayerSentiment]:
        """Analyze sentiment for specific player or all players."""
        cache_key = f"player_{player_name}_{dataset_name}"
        if cached := self._get_cached(cache_key):
            return cached
        
        df = self._get_dataframe(dataset_name)
        if not all(col in df.columns for col in ['screenname', 'polarity']):
            return []
        
        df_clean = df.dropna(subset=['screenname', 'polarity'])
        if player_name:
            df_clean = df_clean[df_clean['screenname'].str.contains(
                player_name, case=False, na=False)]
        
        players = []
        for player, group in df_clean.groupby('screenname'):
            if len(group) < self.MIN_PLAYER_MENTIONS:
                continue
            
            avg_polarity = float(group['polarity'].mean())
            keywords = self._extract_keywords(group)
            
            players.append(PlayerSentiment(
                player_name=str(player),
                mention_count=len(group),
                avg_polarity=round(avg_polarity, 4),
                sentiment_category=self._categorize_sentiment(avg_polarity),
                top_keywords=keywords
            ))
        
        players.sort(key=lambda x: x.mention_count, reverse=True)
        result = players[:20]
        
        self._set_cache(cache_key, result)
        return result
    
    def _extract_keywords(self, df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """Extract top keywords from text content."""
        if 'text' not in df.columns:
            return []
        
        all_text = ' '.join(df['text'].astype(str).tolist())
        words = [w for w in all_text.lower().split() if len(w) > 4]
        return [word for word, _ in Counter(words).most_common(top_n)]
    
    def get_sentiment_trends(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sentiment trends and patterns."""
        cache_key = f"trends_{dataset_name}"
        if cached := self._get_cached(cache_key):
            return cached
        
        df = self._get_dataframe(dataset_name)
        df_clean = df.dropna(subset=['polarity'])
        
        engagement_corr = self._calculate_engagement_correlation(df_clean)
        top_users = self._get_top_users(df_clean, limit=5)
        
        result = {
            "sentiment_volatility": round(float(df_clean['polarity'].std()), 4),
            "engagement_sentiment_correlation": round(engagement_corr, 4),
            "total_unique_users": int(df_clean['username'].nunique()) if 'username' in df_clean.columns else 0,
            "avg_post_length": round(self._safe_column_mean(df_clean, 'text', 0), 1) if 'text' in df_clean.columns else 0,
            "most_active_users": top_users,
            "dataset_coverage": {
                "total_records": len(df_clean),
                "date_range": "Historical NBA Twitter data",
                "platforms_covered": list(df_clean['partition_0'].unique()) if 'partition_0' in df_clean.columns else []
            }
        }
        
        self._set_cache(cache_key, result)
        return result
    
    def _calculate_engagement_correlation(self, df: pd.DataFrame) -> float:
        """Calculate correlation between sentiment and engagement."""
        if 'retweet_count' not in df.columns:
            return 0.0
        
        corr_df = df[['polarity', 'retweet_count']].dropna()
        if len(corr_df) < 10:
            return 0.0
        
        return float(corr_df.corr().iloc[0, 1])
    
    def _get_top_users(self, df: pd.DataFrame, limit: int = 5) -> List[Dict[str, Any]]:
        """Get most active users from dataframe."""
        if 'username' not in df.columns:
            return []
        
        user_counts = df['username'].value_counts().head(limit)
        return [{"username": str(user), "post_count": int(count)} 
                for user, count in user_counts.items()]
    
    def search_by_keyword(self, keyword: str, 
                         dataset_name: Optional[str] = None,
                         limit: int = 50) -> Dict[str, Any]:
        """Search tweets by keyword and analyze sentiment."""
        df = self._get_dataframe(dataset_name)
        
        if 'text' not in df.columns:
            return {"error": "Text column not found in dataset"}
        
        results = df[df['text'].str.contains(keyword, case=False, na=False)].head(limit)
        
        if results.empty:
            return {
                "keyword": keyword,
                "matches_found": 0,
                "message": "No tweets found containing this keyword"
            }
        
        breakdown = self._calculate_sentiment_breakdown(results) if 'polarity' in results.columns else {}
        
        return {
            "keyword": keyword,
            "matches_found": len(results),
            "avg_sentiment": round(self._safe_column_mean(results, 'polarity'), 4),
            "sentiment_breakdown": {k: int(v * len(results) / 100) for k, v in breakdown.items()},
            "sample_tweets": results['text'].head(5).tolist()
        }
    
    def get_dataset_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary statistics for all loaded datasets."""
        return {
            name: {
                "records": len(df),
                "columns": list(df.columns),
                "date_range": "Historical NBA Twitter data",
                "has_sentiment": 'polarity' in df.columns,
                "has_engagement": 'retweet_count' in df.columns,
                "unique_users": int(df['username'].nunique()) if 'username' in df.columns else 0
            }
            for name, df in self.datasets.items()
        }
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._cache.clear()
        print("🧹 Sentiment cache cleared")