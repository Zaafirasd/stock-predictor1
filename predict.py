import json
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

def handler(event, context):
    """
    This is the main function that Netlify will run.
    It takes a stock ticker, performs analysis, and returns the data.
    """
    try:
        # --- 1. Get Ticker from the URL Query ---
        ticker = event['queryStringParameters'].get('ticker')
        if not ticker:
            return {
                'statusCode': 400,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': 'Ticker symbol is required.'})
            }

        # --- 2. Fetch Data using yfinance ---
        start_date = "2020-01-01"
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            return {
                'statusCode': 404,
                'headers': { 'Content-Type': 'application/json' },
                'body': json.dumps({'error': f"Could not find data for ticker '{ticker}'. Check the symbol."})
            }

        # --- 3. Prepare Data for Modeling ---
        stock_data.reset_index(inplace=True)
        stock_data['DateOrdinal'] = stock_data['Date'].map(datetime.toordinal)
        
        X = stock_data[['DateOrdinal']]
        y = stock_data['Close']

        # --- 4. Train Linear Regression Model ---
        model = LinearRegression()
        model.fit(X, y)

        # --- 5. Generate Future Predictions ---
        last_date = stock_data['Date'].iloc[-1]
        future_dates_pd = pd.date_range(start=last_date + timedelta(days=1), end="2027-12-31")
        future_ordinals = np.array([d.toordinal() for d in future_dates_pd]).reshape(-1, 1)
        future_predictions = model.predict(future_ordinals)

        # --- 6. Determine Recommendation ---
        last_actual_price = y.iloc[-1]
        last_predicted_price = future_predictions[-1]
        recommendation = "BUY" if last_predicted_price > last_actual_price else "SELL"

        # --- 7. Format Data for JSON Response ---
        # This is what we send back to the front-end JavaScript
        response_data = {
            'ticker': ticker,
            'recommendation': recommendation,
            'historical_data': {
                'dates': stock_data['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'prices': y.tolist()
            },
            'future_predictions': {
                'dates': [d.strftime('%Y-%m-%d') for d in future_dates_pd],
                'prices': future_predictions.tolist()
            }
        }
        
        return {
            'statusCode': 200,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps(response_data)
        }

    except Exception as e:
        # If any error occurs, return an error message
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json' },
            'body': json.dumps({'error': f'An internal error occurred: {str(e)}'})
        }
