name: "stock-analyst"
description: "Analyzes stock volatility and provides buy/sell timing recommendations. Invoke when user asks for stock analysis, volatility charts, or trading advice."
----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Stock Analyst

This skill provides stock visualization and trading recommendations based on volatility analysis.

## Core Functionality

### 1. Data Requirements

- Stock ticker symbol
- Date range for analysis
- Data source: Yahoo Finance (yfinance library)

### 2. Visualization Features

#### Daily Volatility Chart

- Calculates daily price changes (percentage change)
- Displays as a time series line chart
- Highlights extreme volatility days (>1 standard deviation)

#### Weekly Volatility Chart

- Calculates weekly price changes from daily data
- Aggregated and displayed on the same chart
- Uses secondary y-axis for scale comparison

#### Combined Chart

- X-axis: Date
- Left Y-axis: Daily volatility (%)
- Right Y-axis: Weekly volatility (%)
- Color coding:
  - Green markers: Low volatility (< 0.5 std) → Potential buy zone
  - Yellow markers: Normal volatility (0.5-1.5 std)
  - Red markers: High volatility (> 1.5 std) → Potential sell zone

### 3. Buy/Sell Recommendation Logic

| Volatility Level | Condition           | Recommendation                              |
| ---------------- | ------------------- | ------------------------------------------- |
| Very Low         | < 0.5 std deviation | **BUY** - Price stable, potential upside    |
| Low              | 0.5 - 1.0 std       | **BUY with caution** - Moderate opportunity |
| Normal           | 1.0 - 1.5 std       | **HOLD** - Normal market behavior           |
| High             | 1.5 - 2.0 std       | **SELL with caution** - Increased risk      |
| Very High        | > 2.0 std           | **SELL** - High volatility, exit positions  |

### 4. Implementation Example

```python
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_stock(ticker: str, period: str = "3mo"):
    """
    Analyze stock volatility and provide trading recommendations.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'TSLA')
        period: Analysis period (1mo, 3mo, 6mo, 1y, etc.)
    
    Returns:
        Dictionary with volatility metrics and recommendations
    """
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    
    daily_returns = data['Close'].pct_change().dropna()
    weekly_returns = data['Close'].resample('W').last().pct_change().dropna()
    
    daily_vol = daily_returns.std()
    weekly_vol = weekly_returns.std()
    
    daily_mean = daily_returns.mean()
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax2 = ax1.twinx()
    ax1.plot(daily_returns.index, daily_returns * 100, 'b-', alpha=0.7, label='Daily Volatility')
    ax2.plot(weekly_returns.index, weekly_returns * 100, 'r-', linewidth=2, label='Weekly Volatility')
    
    threshold_buy = daily_mean - 0.5 * daily_vol
    threshold_sell = daily_mean + 1.5 * daily_vol
    
    ax1.axhline(y=threshold_buy * 100, color='green', linestyle='--', label='Buy Threshold')
    ax1.axhline(y=threshold_sell * 100, color='red', linestyle='--', label='Sell Threshold')
    
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily Volatility (%)', color='blue')
    ax2.set_ylabel('Weekly Volatility (%)', color='red')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title(f'{ticker} Volatility Analysis - {period.upper()}')
    plt.tight_layout()
    plt.savefig(f'{ticker}_volatility.png')
    plt.show()
    
    latest_daily = daily_returns.iloc[-1]
    latest_weekly = weekly_returns.iloc[-1]
    
    if latest_daily < threshold_buy:
        recommendation = "BUY"
        reason = "Volatility is below normal, potential entry point"
    elif latest_daily > threshold_sell:
        recommendation = "SELL"
        reason = "Volatility is elevated, consider taking profits"
    else:
        recommendation = "HOLD"
        reason = "Volatility within normal range"
    
    return {
        'ticker': ticker,
        'daily_volatility': daily_vol * 100,
        'weekly_volatility': weekly_vol * 100,
        'latest_daily_return': latest_daily * 100,
        'latest_weekly_return': latest_weekly * 100,
        'recommendation': recommendation,
        'reason': reason
    }
```

### 5. Usage Instructions

1. Import the required libraries
2. Call `analyze_stock(ticker, period)` with appropriate parameters
3. Review the generated chart
4. Check the recommendation dictionary for trading signals

### 6. Output Interpretation

- **Chart**: Blue line shows daily volatility, red line shows weekly volatility
- **Green dashed line**: Buy threshold
- **Red dashed line**: Sell threshold
- **Recommendation**: Based on the latest volatility reading relative to historical norms

### 7. Limitations

- Past volatility does not guarantee future performance
- This is an analytical tool, not financial advice
- Always consider fundamental analysis alongside technical indicators
- Market conditions can change rapidly
