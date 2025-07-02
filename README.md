# Probability Crypto Scalping & Compounding Trading Dashboard

An intelligent trading dashboard built using Dash & Plotly, designed for **crypto scalping** and **compounding profit strategies**. Combines **real-time data**, **ML models**, and **statistical simulations** to provide actionable insights with profit breakdowns tailored for Indian users.

---

##  Features

-  **Real-time crypto data** from Binance (converted to INR)
-  **ML Predictions** using:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
  - SVR
  - MLP (Neural Network)
-  Candlestick chart for historical prices
-  **Profit & tax breakdown** (Platform Fee, GST, TDS)
-  Monte Carlo, Bayesian, Chebyshev & GBM simulations
-  ADF Test for stationarity
-  Risk meter & range recommendations
-  Estimated holding time for potential profits

---

## Tech Stack

- Dash
- Plotly
- Pandas, NumPy
- Scikit-learn
- Statsmodels
- Binance API
- ExchangeRate API

---

## Getting Started

```bash
git clone https://github.com/Vallurikeerthiram/Probability-Based-Crypto-For-Scalping-and-Compounding.git
cd Probability-Based-Crypto-For-Scalping-and-Compounding
pip install -r requirements.txt
```

### Add `.env` file (for Binance API)

```bash
BINANCE_API_KEY=xxxxxxxxxxxxxx
BINANCE_API_SECRET=xxxxxxxxxxx
```

### Run the App

```bash
python app.py
```

Then open in your browser: [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## How It Works

- Select a coin and time interval (1H, 24H, 7D, 1M, 6M)
- Fetches real-time OHLC data from Binance
- Applies:
  - ML models for next-step prediction
  - Statistical simulations for fluctuation estimation
  - Profit estimation with INR conversion & Indian tax logic
- Visualizes:
  - Candlestick chart
  - Risk meter
  - Suggested buy/sell range
  - Detailed tax/fee breakdown
  - Profit potential & expected holding time

---

## Strategy Fit

Perfectly suited for:

-  **Scalping** (fast, intraday trades)
-  **Compounding gains** (frequent re-entry using updated predictions)
- 🇮🇳 **Indian traders** (supports INR + Platform Fee, GST, TDS)

---

##  Author

**Valluri Keerthi Ram**  
B.Tech CSE @ Amrita Vishwa Vidyapeetham  
keerthiramvalluri@gmail.com  
bangalore, India

---

## License

This project is licensed under the **MIT License**.
