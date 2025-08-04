import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf


def get_portfolio_data(tickers, period = "5y", price_col = "Close"):


    try:
        print(f"Downloading data for tickers: {tickers}")
        # Fetch all data
        data = yf.download(tickers, period = period, progress = False, auto_adjust = True)
        

        if data.empty:
            raise ValueError(f"No data downloaded for tickers: {tickers}")

        if len(tickers) == 1:
            prices = data[price_col].to_frame()
            prices.columns = tickers
        else:
            prices = data.get(price_col)
            if prices is None or prices.empty:
                raise ValueError(f"No '{price_col}' data found in downloaded data")

        
        # Risk-free rate with fallback options
        risk_free_rate = 0.04  # Default fallback rate (4%)
        try: 
            irx_hist = yf.Ticker("^IRX").history(period="5d")
            risk_free_rate = float(irx_hist["Close"].iloc[-1]) / 100
            print(f"Risk-free rate fetched: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
        except Exception as e:
            print(f"Warning: Failed to fetch risk-free rate ({e}), using fallback: {risk_free_rate:.4f}")
        
        # Sector data - fetch for each ticker individually
        sectors = {}
        try:
            print("Fetching sector data for each ticker...")
            for ticker in tickers:
                try:
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    sector = info.get('sector', 'Unknown')
                    industry = info.get('industry', 'Unknown')
                    sectors[ticker] = {
                        'sector': sector,
                        'industry': industry,
                        'company_name': info.get('longName', ticker)
                    }
                except Exception as e:
                    print(f"  {ticker}: Failed to fetch sector data ({e})")
                    sectors[ticker] = {
                        'sector': 'Unknown',
                        'industry': 'Unknown', 
                        'company_name': ticker
                    }
            
            print(f"Sector data fetched for {len(sectors)} tickers")
        except Exception as e:
            print(f"Warning: Failed to fetch sector data ({e}), sector data will be None")
            sectors = None

        # Also check if main stock data is valid
        if prices is None or prices.empty:
            raise ValueError("No stock price data was downloaded. Check ticker symbols and internet connection.")
        
        
        print(f"Successfully downloaded data for: {list(prices.columns)}")

        # Calculate
        returns = prices.pct_change().dropna() # relative change in prices, skipping NaN
        cov = returns.cov()
        corr = returns.corr()

        # Statistics
        stats = {
            "risk_free_rate": risk_free_rate,
            "min_return": returns.min(),
            "max_return": returns.max(),
            "mean_returns": returns.mean(), # estimate of daily expected returns E[r]
            "annualized_mean_returns": returns.mean() * 252,
            'volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252)
        } 

        return {
            "prices": prices,
            "returns": returns,
            "covariance": cov,
            "annualized_covariance": cov * np.sqrt(252),
            "correlation": corr,
            "annualized_correlation": corr * np.sqrt(252),
            "sectors": sectors,  # Updated to include all sector data
            "stats": stats
        }
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None
    

