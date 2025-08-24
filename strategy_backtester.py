"""
Strateji Backtester Modülü
Farklı trading stratejilerini test eder
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def backtest_strategy(data: pd.DataFrame, strategy_name: str, initial_capital: float = 10000) -> Dict:
    """
    Tek bir stratejiyi backtest eder
    
    Args:
        data: İşlenmiş veri
        strategy_name: Strateji adı
        initial_capital: Başlangıç sermayesi
    
    Returns:
        Backtest sonuçları
    """
    
    # Strateji sinyallerini al
    if 'Final_Signal' in data.columns:
        signals = data['Final_Signal'].map({'AL': 1, 'SAT': -1, 'TUT': 0})
    else:
        signals = pd.Series(0, index=data.index)
    
    # Returns hesapla
    returns = data['Returns'].fillna(0)
    strategy_returns = returns * signals.shift(1)
    
    # Kümülatif getiri
    cumulative_returns = (1 + strategy_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    # Win rate
    winning_trades = strategy_returns[strategy_returns > 0].count()
    losing_trades = strategy_returns[strategy_returns < 0].count()
    total_trades = winning_trades + losing_trades
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Maximum drawdown
    cummax = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cummax) / cummax
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (yıllık)
    sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
    return {
        'strategy_name': strategy_name,
        'total_return': total_return,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades
    }

def backtest_multiple_strategies(data: pd.DataFrame, strategies: Dict = None) -> pd.DataFrame:
    """
    Birden fazla stratejiyi karşılaştırır
    
    Args:
        data: İşlenmiş veri
        strategies: Strateji fonksiyonları dict'i
    
    Returns:
        Karşılaştırma tablosu
    """
    
    if strategies is None:
        # Default stratejiler
        strategies = {
            'Buy & Hold': lambda d: pd.Series(1, index=d.index),
            'RSI': lambda d: np.where(d['RSI'] < 30, 1, np.where(d['RSI'] > 70, -1, 0)),
            'MACD': lambda d: np.where(d['MACD'] > d['MACD_Signal'], 1, -1),
            'Moving Average': lambda d: np.where(d['SMA_50'] > d['SMA_200'], 1, -1),
            'Combined': lambda d: d['Final_Signal'].map({'AL': 1, 'SAT': -1, 'TUT': 0})
        }
    
    results = []
    
    for name, strategy_func in strategies.items():
        try:
            # Strateji sinyallerini hesapla
            signals = strategy_func(data)
            
            # Backtest yap
            data['Strategy_Signal'] = signals
            result = backtest_strategy(data, name)
            results.append(result)
        except Exception as e:
            print(f"Strateji {name} hata verdi: {e}")
            continue
    
    # DataFrame'e çevir
    df_results = pd.DataFrame(results)
    
    # Sıralama
    df_results = df_results.sort_values('total_return', ascending=False)
    
    return df_results

def calculate_portfolio_metrics(data: pd.DataFrame, weights: Dict[str, float]) -> Dict:
    """
    Portfolio metriklerini hesaplar
    
    Args:
        data: Fiyat verisi
        weights: Varlık ağırlıkları
    
    Returns:
        Portfolio metrikleri
    """
    
    portfolio_return = sum(data[ticker] * weight for ticker, weight in weights.items())
    
    return {
        'total_return': portfolio_return.sum(),
        'volatility': portfolio_return.std() * np.sqrt(252),
        'sharpe_ratio': np.sqrt(252) * portfolio_return.mean() / portfolio_return.std()
    }
