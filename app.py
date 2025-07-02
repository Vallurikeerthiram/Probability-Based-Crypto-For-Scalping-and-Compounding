import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
# Machine Learning and AI Models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Random Processes and Simulations
import random
from sklearn.preprocessing import StandardScaler

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Crypto Prediction Tool"

# Constants for tax calculation
PLATFORM_FEE_PERCENT = 0.5
GST_PERCENT = 18
TDS_PERCENT = 1
CYCLES_PER_DAY = 24 * 60 

# Function to fetch crypto data from Binance API
def get_crypto_data(crypto_symbol, interval):
    url = f'https://api.binance.com/api/v3/klines'
    interval_map = {
        '1h': '1s',
        '24h': '5m',
        '7d': '1h',
        '1m': '1d',
        '6m': '1d'
    }
    adjusted_interval = interval_map[interval]
    
    limit_map = {
        '1h': 3600,
        '24h': 288,
        '7d': 168,
        '1m': 30,
        '6m': 180
    }
    limit = limit_map[interval]
    
    response = requests.get(url, params={
        'symbol': crypto_symbol,
        'interval': adjusted_interval,
        'limit': limit
    })
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                     'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                     'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close']].astype(float)
    return df

# Function to fetch all coins from Binance API
def get_all_coins():
    url = 'https://api.binance.com/api/v3/ticker/price'
    response = requests.get(url)
    data = response.json()
    coins = [item['symbol'] for item in data if item['symbol'].endswith('USDT')]
    coin_options = [{'label': coin.replace('USDT', ''), 'value': coin} for coin in coins]
    return coin_options

# Function to calculate INR conversion rate
def get_inr_conversion_rate():
    url = 'https://api.exchangerate-api.com/v4/latest/USD'
    response = requests.get(url)
    data = response.json()
    return data['rates']['INR']

# Function to calculate final profit with breakdown
def calculate_profit(investment, buy_price, sell_price):
    platform_fee_buy = (PLATFORM_FEE_PERCENT / 100) * investment
    gst_buy = (GST_PERCENT / 100) * platform_fee_buy
    total_cost = investment + platform_fee_buy + gst_buy

    units_bought = total_cost / buy_price
    sell_amount = units_bought * sell_price

    platform_fee_sell = (PLATFORM_FEE_PERCENT / 100) * sell_amount
    gst_sell = (GST_PERCENT / 100) * platform_fee_sell
    tds = (TDS_PERCENT / 100) * sell_amount

    final_amount = sell_amount - platform_fee_sell - gst_sell - tds
    profit = final_amount - total_cost

    return final_amount, profit, platform_fee_buy, gst_buy, platform_fee_sell, gst_sell, tds

# Function to analyze fluctuation ranges
def analyze_fluctuations(df):
    # 1. **Descriptive Statistics**
    mean_price = df['close'].mean()
    std_dev = df['close'].std()

    # 2. **Chebyshev's Theorem-based Bounds**
    lower_bound = mean_price - 2 * std_dev
    upper_bound = mean_price + 2 * std_dev

    # 3. **Bayesian Analysis**
    bayesian_mean = np.mean(df['close'])
    bayesian_std = np.std(df['close'])
    bayesian_lower_bound = bayesian_mean - 1.96 * bayesian_std
    bayesian_upper_bound = bayesian_mean + 1.96 * bayesian_std

    # 4. **Rolling Averages**
    df['rolling_mean'] = df['close'].rolling(window=10).mean()
    df['rolling_std'] = df['close'].rolling(window=10).std()

    # 5. **Monte Carlo Simulation**
    simulations = 1000
    monte_carlo_predictions = [np.random.normal(bayesian_mean, bayesian_std) for _ in range(simulations)]
    monte_carlo_lower_bound = np.percentile(monte_carlo_predictions, 5)
    monte_carlo_upper_bound = np.percentile(monte_carlo_predictions, 95)

    # 6. **Stationarity Testing**
    adf_result = adfuller(df['close'])
    is_stationary = adf_result[1] <= 0.05

    # 7. **Machine Learning Models for Pattern Prediction**
    df['target'] = df['close'].shift(-1)  # Predict the next close price
    features = df[['open', 'high', 'low', 'close']].iloc[:-1]
    target = df['target'].iloc[:-1]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # **Linear Regression**
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)

    # **Random Forest Regressor**
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    # **Gradient Boosting Regressor**
    gbr_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbr_model.fit(X_train, y_train)
    gbr_predictions = gbr_model.predict(X_test)

    # **Support Vector Regression**
    svr_model = SVR(kernel='rbf')
    svr_model.fit(X_train, y_train)
    svr_predictions = svr_model.predict(X_test)

    # **Neural Network - Multi-Layer Perceptron**
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    mlp_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)
    mlp_predictions = mlp_model.predict(X_test_scaled)

    # Aggregate Results from ML Models
    all_predictions = np.vstack((lr_predictions, rf_predictions, gbr_predictions, svr_predictions, mlp_predictions))
    final_prediction_mean = np.mean(all_predictions, axis=0)
    final_prediction_lower_bound = np.percentile(final_prediction_mean, 5)
    final_prediction_upper_bound = np.percentile(final_prediction_mean, 95)

    # 8. **Random Walk Prediction using GBM**
    drift = mean_price * (std_dev ** 2) / 2
    random_walks = drift + std_dev * np.random.normal(0, 1, simulations)
    gbm_lower_bound = np.percentile(random_walks, 5)
    gbm_upper_bound = np.percentile(random_walks, 95)

    # 9. **Risk Assessment**
    risk_score = (upper_bound - lower_bound) / mean_price * 100

    # 10. **Most Frequent Range**
    most_frequent_range = df['close'].round(5).value_counts().idxmax()

    # 11. **Recommendations**
    recommended_buy = most_frequent_range
    recommended_sell = max(final_prediction_upper_bound, gbm_upper_bound)

    return {
        "predicted_lower_bound": lower_bound,
        "predicted_upper_bound": upper_bound,
        "bayesian_bounds": (bayesian_lower_bound, bayesian_upper_bound),
        "monte_carlo_bounds": (monte_carlo_lower_bound, monte_carlo_upper_bound),
        "ml_bounds": (final_prediction_lower_bound, final_prediction_upper_bound),
        "gbm_bounds": (gbm_lower_bound, gbm_upper_bound),
        "most_frequent_range": most_frequent_range,
        "risk_score": risk_score,
        "is_stationary": is_stationary,
        "recommended_buy": recommended_buy,
        "recommended_sell": recommended_sell,
    }

# Function to determine time frame for profit prediction
def estimate_time_frame(crypto_symbol):
    # Dynamically determine time frames based on volatility and clustering
    if crypto_symbol.startswith('BTC') or crypto_symbol.startswith('ETH'):
        return "Profits are likely to materialize over weeks or months due to moderate volatility."
    else:
        return "Profits may occur within hours to days due to high volatility."  

# Fetch all coins for dropdown
coin_options = get_all_coins()

# Layout definition
app.layout = html.Div(
    style={
        'backgroundColor': '#1E1E1E',
        'color': '#FFFFFF',
        'fontFamily': 'Arial',
        'padding': '20px'
    },
    children=[
        html.H1(
            'Crypto Trading Prediction Tool',
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),

        html.Div([
            html.Label('Select Cryptocurrency:', style={'color': '#FFFFFF'}),
            dcc.Dropdown(
                id='coin-dropdown',
                options=coin_options,
                value='BTCUSDT',
                style={'width': '50%', 'margin': 'auto', 'color': '#000'},
                searchable=True
            )
        ], style={'textAlign': 'center', 'marginBottom': '30px'}),

        html.Div([
            html.Label('Select Time Interval:', style={'color': '#FFFFFF'}),
            dcc.RadioItems(
                id='time-interval',
                options=[
                    {'label': '1 Hour', 'value': '1h'},
                    {'label': '24 Hours', 'value': '24h'},
                    {'label': '7 Days', 'value': '7d'},
                    {'label': '1 Month', 'value': '1m'},
                    {'label': '6 Months', 'value': '6m'}
                ],
                value='24h',
                style={'display': 'inline-block', 'padding': '10px'}
            )
        ], style={'textAlign': 'center', 'marginBottom': '30px'}),

        dcc.Graph(id='crypto-graph'),

        html.Div([
            html.Label('Investment Amount (INR):', style={'color': '#FFFFFF'}),
            dcc.Input(
                id='investment-input',
                type='number',
                value=1000,
                style={'width': '100px'}
            ),
            html.Label(
                'Platform Fee (%):',
                style={'color': '#FFFFFF', 'marginLeft': '20px'}
            ),
            dcc.Input(
                id='platform-fee',
                type='number',
                value=0.5,
                style={'width': '100px'}
            )
        ], style={'textAlign': 'center', 'marginBottom': '30px'}),

        dcc.Graph(id='risk-meter', style={'height': '300px'}),

        html.Div(id='profit-analysis', style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Div(id='time-frame-prediction', style={'textAlign': 'center', 'marginTop': '20px'}),
    ]
)

# Callback to update graph and analysis
@app.callback(
    [
        Output('crypto-graph', 'figure'),
        Output('profit-analysis', 'children'),
        Output('time-frame-prediction', 'children'),
        Output('risk-meter', 'figure')
    ],
    [
        Input('coin-dropdown', 'value'),
        Input('time-interval', 'value'),
        Input('investment-input', 'value'),
        Input('platform-fee', 'value')
    ]
)
def update_dashboard(coin, interval, investment, platform_fee):
    try:
        # Global platform fee (if needed)
        global PLATFORM_FEE_PERCENT
        PLATFORM_FEE_PERCENT = platform_fee

        # Fetch cryptocurrency data
        df = get_crypto_data(coin, interval)
        if df.empty:
            raise ValueError("No data available for the selected coin and interval.")

        # Fetch INR conversion rate and convert prices
        inr_rate = get_inr_conversion_rate()
        df['close'] = df['close'] * inr_rate
        df['open'] = df['open'] * inr_rate
        df['high'] = df['high'] * inr_rate
        df['low'] = df['low'] * inr_rate

        # Analyze fluctuations and calculate key metrics
        analysis = analyze_fluctuations(df)

        # Extract values from the analysis
        buy_price = analysis['most_frequent_range']
        sell_price = analysis['predicted_upper_bound']
        risk_score = analysis['risk_score']
        is_stationary = analysis['is_stationary']

        # Calculate profits and breakdown
        final_amount, profit, platform_fee_buy, gst_buy, platform_fee_sell, gst_sell, tds = calculate_profit(
            investment, buy_price, sell_price
        )

        # Build profit analysis content
        profit_analysis = html.Div([
            html.P(f"Suggested Buy Price: ₹{buy_price:.5f}", style={'color': '#00FF00'}),
            html.P(f"Suggested Sell Price: ₹{sell_price:.5f}", style={'color': '#FF0000'}),
            html.P(f"Profit After Taxes: ₹{profit:.2f} ({(profit / investment) * 100:.2f}%)",
                   style={'color': '#FFFFFF'}),
            html.P("Breakdown:", style={'color': '#FFFFFF'}),
            html.P(f"  Platform Fee on Buy: ₹{platform_fee_buy:.2f}", style={'color': '#FFFFFF'}),
            html.P(f"  GST on Buy Fee: ₹{gst_buy:.2f}", style={'color': '#FFFFFF'}),
            html.P(f"  Platform Fee on Sell: ₹{platform_fee_sell:.2f}", style={'color': '#FFFFFF'}),
            html.P(f"  GST on Sell Fee: ₹{gst_sell:.2f}", style={'color': '#FFFFFF'}),
            html.P(f"  TDS on Sell: ₹{tds:.2f}", style={'color': '#FFFFFF'})
        ])

        # Estimate time frame for profit realization
        time_frame = html.P(f"Expected Time Frame: {estimate_time_frame(coin)}", style={'color': '#FFFFFF'})

        # Create candlestick chart for price analysis
        figure = {
            'data': [
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name=coin
                )
            ],
            'layout': go.Layout(
                title=f"{coin.replace('USDT', '')} Price Analysis",
                xaxis={'title': 'Time'},
                yaxis={'title': 'Price (INR)'},
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font={'color': '#FFFFFF'}
            )
        }

        # Create risk meter gauge
        risk_meter = {
            'data': [
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': 'darkblue'},
                        'steps': [
                            {'range': [0, 3], 'color': 'green'},
                            {'range': [3, 7], 'color': 'yellow'},
                            {'range': [7, 10], 'color': 'red'}
                        ],
                        'threshold': {
                            'line': {'color': 'red', 'width': 4},
                            'thickness': 0.75,
                            'value': risk_score
                        }
                    },
                    title={'text': "Risk Meter"}
                )
            ],
            'layout': go.Layout(
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor='#1E1E1E',
                font={'color': '#FFFFFF'}
            )
        }

        return figure, profit_analysis, time_frame, risk_meter

    except Exception as e:
        # Handle errors and return fallback values
        empty_figure = {
            'data': [],
            'layout': go.Layout(
                title="Error",
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font={'color': '#FFFFFF'}
            )
        }
        error_message = html.P(f"Error: {str(e)}", style={'color': '#FF0000'})
        empty_risk_meter = {
            'data': [],
            'layout': go.Layout(
                title="Error",
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font={'color': '#FFFFFF'}
            )
        }
        return empty_figure, error_message, error_message, empty_risk_meter


if __name__ == '__main__':
    app.run_server(debug=True)
