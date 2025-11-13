"""NBA Forecast Tool - Statistical Prediction Engine

Data-driven forecasting engine providing quantitative predictions and trend analysis.
Designed for integration with AI agents that make strategic recommendations.
"""

import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats, signal

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


class TrendDirection(Enum):
    STRONG_GROWTH = "strong_growth"
    MODERATE_GROWTH = "moderate_growth"
    STABLE = "stable"
    MODERATE_DECLINE = "moderate_decline"
    STRONG_DECLINE = "strong_decline"


@dataclass
class StatisticalForecast:
    metric_name: str
    current_value: float
    predicted_value: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    confidence_score: float
    trend_direction: str
    percent_change: float
    volatility: float
    sample_size: int
    model_accuracy: float


@dataclass
class TrendAnalysis:
    metric: str
    historical_values: List[float]
    moving_average: float
    trend_coefficient: float
    acceleration: float
    seasonality_detected: bool
    anomaly_score: float


@dataclass
class CorrelationInsight:
    variable_x: str
    variable_y: str
    correlation_strength: float
    statistical_significance: float
    relationship_type: str


class ForecastTool:
    """Production-grade statistical forecasting engine for NBA analytics.
    
    Provides quantitative predictions, trend analysis, and correlation insights
    without prescriptive recommendations. AI agents interpret results for strategy.
    """
    
    CACHE_TTL_SECONDS = 3600
    MIN_SAMPLE_SIZE = 10
    
    def __init__(self, data_dir: str = "data/forecast_dataset"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        
        self._initialize_datasets()
    
    def _initialize_datasets(self) -> None:
        """Load and validate all datasets from directory"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    self.datasets[csv_file.stem] = df
            except Exception as e:
                print(f"Warning: Failed to load {csv_file.name}: {e}")
        
        if not self.datasets:
            raise ValueError("No valid datasets loaded")
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Retrieve cached result if valid"""
        if key not in self._cache:
            return None
        
        data, timestamp = self._cache[key]
        if datetime.now() - timestamp < timedelta(seconds=self.CACHE_TTL_SECONDS):
            return data
        
        del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any) -> None:
        """Store result in cache with timestamp"""
        self._cache[key] = (data, datetime.now())
    
    def _calculate_trend_direction(self, percent_change: float) -> TrendDirection:
        """Classify trend based on percent change"""
        if percent_change > 10:
            return TrendDirection.STRONG_GROWTH
        elif percent_change > 3:
            return TrendDirection.MODERATE_GROWTH
        elif percent_change > -3:
            return TrendDirection.STABLE
        elif percent_change > -10:
            return TrendDirection.MODERATE_DECLINE
        else:
            return TrendDirection.STRONG_DECLINE
    
    def _calculate_confidence(self, r2: float, sample_size: int, volatility: float) -> float:
        """Calculate prediction confidence based on multiple factors"""
        base_confidence = max(0, r2)
        size_factor = min(sample_size / 50, 1.0)
        volatility_penalty = max(0, 1 - (volatility / 100))
        
        confidence = base_confidence * 0.6 + size_factor * 0.2 + volatility_penalty * 0.2
        return round(max(0.1, min(0.99, confidence)), 2)
    
    def _train_ensemble(self, X: np.ndarray, y: np.ndarray, model_key: str) -> Dict[str, float]:
        """Train ensemble of models and select best performer"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        best_metrics = {}
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            predictions = model.predict(X_test_scaled)
            
            r2 = r2_score(y_test, predictions)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                       cv=min(5, len(X_train)), scoring='r2')
            cv_mean = cv_scores.mean()
            
            if cv_mean > best_score:
                best_score = cv_mean
                best_model = model
                best_metrics = {
                    'mae': mean_absolute_error(y_test, predictions),
                    'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                    'r2': r2,
                    'cv_r2': cv_mean,
                    'model_type': name
                }
        
        self.models[model_key] = best_model
        self.scalers[model_key] = scaler
        self.model_performance[model_key] = best_metrics
        
        return best_metrics
    
    def list_available_metrics(self) -> Dict[str, List[str]]:
        """List all available metrics organized by category
        
        Returns:
            Dictionary with categories and their available metrics
        """
        categorized_metrics = {
            'social_engagement': [],
            'performance': [],
            'financial': [],
            'attendance': [],
            'other': []
        }
        
        for dataset_name, df in self.datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_lower = col.lower()
                
                if any(word in col_lower for word in ['twitter', 'wiki', 'social', 'favorite', 'pageview']):
                    category = 'social_engagement'
                elif any(word in col_lower for word in ['win', 'loss', 'elo', 'rpm', 'performance', 'pts', 'ast']):
                    category = 'performance'
                elif any(word in col_lower for word in ['salary', 'endorsement', 'value', 'revenue']):
                    category = 'financial'
                elif any(word in col_lower for word in ['attendance', 'crowd', 'fan']):
                    category = 'attendance'
                else:
                    category = 'other'
                
                metric_entry = f"{dataset_name}.{col}"
                if metric_entry not in categorized_metrics[category]:
                    categorized_metrics[category].append(metric_entry)
        
        return {k: v for k, v in categorized_metrics.items() if v}
    
    def get_metric_info(self, dataset_name: str, metric_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific metric
        
        Args:
            dataset_name: Dataset identifier
            metric_name: Metric column name
        
        Returns:
            Dictionary with metric statistics and metadata
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        if metric_name not in df.columns:
            raise ValueError(f"Metric '{metric_name}' not in dataset")
        
        col_data = df[metric_name].dropna()
        
        if not pd.api.types.is_numeric_dtype(df[metric_name]):
            return {"error": "Metric is not numeric"}
        
        return {
            "metric": metric_name,
            "dataset": dataset_name,
            "mean": round(float(col_data.mean()), 3),
            "median": round(float(col_data.median()), 3),
            "std_dev": round(float(col_data.std()), 3),
            "min": round(float(col_data.min()), 3),
            "max": round(float(col_data.max()), 3),
            "sample_size": len(col_data),
            "missing_pct": round((df[metric_name].isnull().sum() / len(df)) * 100, 2)
        }
    
    def auto_forecast(self, dataset_name: str, target_metric: str,
                     current_context: Optional[Dict[str, float]] = None) -> StatisticalForecast:
        """Automatically select features and generate forecast
        
        Args:
            dataset_name: Dataset identifier
            target_metric: Metric to predict
            current_context: Optional current values for prediction
        
        Returns:
            StatisticalForecast with automatically selected features
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        if target_metric not in df.columns:
            raise ValueError(f"Target metric '{target_metric}' not in dataset")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_metric in numeric_cols:
            numeric_cols.remove(target_metric)
        
        exclude_keywords = ['id', 'ID', 'index', 'Index']
        feature_cols = [col for col in numeric_cols 
                       if not any(keyword in col for keyword in exclude_keywords)]
        
        if not feature_cols:
            raise ValueError(f"No suitable features found for predicting {target_metric}")
        
        correlations = []
        target_data = df[target_metric].dropna()
        
        for feature in feature_cols:
            feature_data = df[feature].dropna()
            common_idx = target_data.index.intersection(feature_data.index)
            
            if len(common_idx) < self.MIN_SAMPLE_SIZE:
                continue
            
            corr = df.loc[common_idx, [target_metric, feature]].corr().iloc[0, 1]
            if not np.isnan(corr) and abs(corr) >= 0.2:
                correlations.append((feature, abs(corr)))
        
        if not correlations:
            raise ValueError(f"No features meet minimum correlation threshold for {target_metric}")
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in correlations[:5]]
        
        return self.forecast_metric(dataset_name, target_metric, selected_features, current_context)
    
    def forecast_metric(self, dataset_name: str, target_metric: str,
                       feature_columns: List[str],
                       current_values: Optional[Dict[str, float]] = None) -> StatisticalForecast:
        """Generate statistical forecast for any metric using ML models
        
        Args:
            dataset_name: Name of dataset to use
            target_metric: Column name to predict
            feature_columns: List of feature column names
            current_values: Optional dict of current feature values for prediction
        
        Returns:
            StatisticalForecast with prediction and confidence metrics
        """
        cache_key = f"{dataset_name}_{target_metric}_{'_'.join(feature_columns)}"
        cached = self._get_cached(cache_key)
        if cached and current_values is None:
            return cached
        
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        if target_metric not in df.columns:
            raise ValueError(f"Target metric '{target_metric}' not in dataset")
        
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not in dataset: {missing_features}")
        
        df_clean = df.dropna(subset=[target_metric] + feature_columns)
        
        if len(df_clean) < self.MIN_SAMPLE_SIZE:
            raise ValueError(f"Insufficient samples: {len(df_clean)} < {self.MIN_SAMPLE_SIZE}")
        
        X = df_clean[feature_columns].values
        y = df_clean[target_metric].values
        sample_size = len(df_clean)
        
        model_key = f"{dataset_name}_{target_metric}"
        if model_key not in self.models:
            metrics = self._train_ensemble(X, y, model_key)
        else:
            metrics = self.model_performance[model_key]
        
        current_value = float(df[target_metric].mean())
        volatility = float(df[target_metric].std() / current_value * 100) if current_value != 0 else 0
        
        if current_values:
            feature_vector = np.array([current_values.get(f, df[f].mean()) for f in feature_columns])
            X_scaled = self.scalers[model_key].transform(feature_vector.reshape(1, -1))
            predicted = float(self.models[model_key].predict(X_scaled)[0])
            
            if hasattr(self.models[model_key], 'estimators_'):
                predictions = np.array([est.predict(X_scaled)[0] for est in self.models[model_key].estimators_])
                std_dev = predictions.std()
            else:
                std_dev = predicted * 0.1
            
            margin = 1.96 * std_dev
            lower = predicted - margin
            upper = predicted + margin
        else:
            mean_features = np.array([df[f].mean() for f in feature_columns])
            X_scaled = self.scalers[model_key].transform(mean_features.reshape(1, -1))
            predicted = float(self.models[model_key].predict(X_scaled)[0])
        
        percent_change = ((predicted - current_value) / current_value) * 100 if current_value != 0 else 0
        trend = self._calculate_trend_direction(percent_change)
        confidence = self._calculate_confidence(metrics['r2'], sample_size, volatility)
        
        forecast = StatisticalForecast(
            metric_name=target_metric,
            current_value=round(current_value, 2),
            predicted_value=round(predicted, 2),
            prediction_interval_lower=round(lower, 2),
            prediction_interval_upper=round(upper, 2),
            confidence_score=confidence,
            trend_direction=trend.value,
            percent_change=round(percent_change, 2),
            volatility=round(volatility, 2),
            sample_size=sample_size,
            model_accuracy=round(metrics['r2'], 3)
        )
        
        self._set_cache(cache_key, forecast)
        return forecast
    
    def analyze_trend(self, dataset_name: str, metric: str, window_size: int = 5) -> TrendAnalysis:
        """Perform time-series trend analysis on metric
        
        Args:
            dataset_name: Dataset identifier
            metric: Metric column name
            window_size: Moving average window size
        
        Returns:
            TrendAnalysis with historical patterns and trends
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        if metric not in df.columns:
            raise ValueError(f"Metric '{metric}' not in dataset")
        
        values = df[metric].dropna().values
        
        if len(values) < window_size:
            raise ValueError(f"Insufficient data points: {len(values)} < {window_size}")
        
        moving_avg = pd.Series(values).rolling(window=window_size).mean().iloc[-1]
        
        X = np.arange(len(values)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, values)
        trend_coef = model.coef_[0]
        
        if len(values) > 2:
            mid = len(values) // 2
            first_trend = np.polyfit(range(mid), values[:mid], 1)[0]
            second_trend = np.polyfit(range(len(values) - mid), values[mid:], 1)[0]
            acceleration = second_trend - first_trend
        else:
            acceleration = 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        z_scores = np.abs((values - mean_val) / std_val) if std_val > 0 else np.zeros_like(values)
        anomaly_score = float(np.max(z_scores))
        
        seasonality = self._detect_seasonality(values)
        
        return TrendAnalysis(
            metric=metric,
            historical_values=values.tolist(),
            moving_average=round(float(moving_avg), 3),
            trend_coefficient=round(float(trend_coef), 4),
            acceleration=round(float(acceleration), 4),
            seasonality_detected=seasonality,
            anomaly_score=round(anomaly_score, 2)
        )
    
    def _detect_seasonality(self, values: np.ndarray, threshold: float = 0.3) -> bool:
        """Detect if time series exhibits seasonal patterns"""
        if len(values) < 8:
            return False
        
        try:
            autocorr = signal.correlate(values, values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            peaks = signal.find_peaks(autocorr[1:], height=threshold)[0]
            return len(peaks) > 0
        except ImportError:
            return False
    
    def calculate_correlations(self, dataset_name: str, variables: List[str]) -> List[CorrelationInsight]:
        """Calculate correlation matrix and identify significant relationships
        
        Args:
            dataset_name: Dataset identifier
            variables: List of variable names to analyze
        
        Returns:
            List of CorrelationInsight objects with significant relationships
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        df = self.datasets[dataset_name]
        
        missing_vars = [v for v in variables if v not in df.columns]
        if missing_vars:
            raise ValueError(f"Variables not in dataset: {missing_vars}")
        
        corr_matrix = df[variables].corr()
        insights = []
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i >= j:
                    continue
                
                corr_value = corr_matrix.loc[var1, var2]
                
                if abs(corr_value) < 0.3:
                    continue
                
                if abs(corr_value) > 0.8:
                    relationship = "very_strong"
                elif abs(corr_value) > 0.6:
                    relationship = "strong"
                elif abs(corr_value) > 0.4:
                    relationship = "moderate"
                else:
                    relationship = "weak"
                
                n = len(df[[var1, var2]].dropna())
                if n > 3:
                    t_stat = corr_value * np.sqrt(n - 2) / np.sqrt(1 - corr_value**2)
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    significance = 1 - p_value
                else:
                    significance = 0.5
                
                insights.append(CorrelationInsight(
                    variable_x=var1,
                    variable_y=var2,
                    correlation_strength=round(float(corr_value), 3),
                    statistical_significance=round(float(significance), 3),
                    relationship_type=relationship
                ))
        
        return sorted(insights, key=lambda x: abs(x.correlation_strength), reverse=True)
    
    def get_dataset_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get statistical summary of all loaded datasets"""
        return {
            name: {
                "rows": len(df),
                "columns": list(df.columns),
                "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(df.select_dtypes(include=['object']).columns),
                "missing_data_pct": round((df.isnull().sum().sum() / df.size) * 100, 2)
            }
            for name, df in self.datasets.items()
        }
    
    def clear_cache(self) -> None:
        """Clear prediction cache"""
        self._cache.clear()