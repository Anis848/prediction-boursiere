import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Prédiction boursière", layout="wide")

st.title("📈 Application de prédiction boursière")

# Liste de tickers populaires
tickers = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "S&P 500 (^GSPC)": "^GSPC",
    "Nasdaq (^IXIC)": "^IXIC"
}

# Sélecteur déroulant
ticker_label = st.selectbox("Choisissez un actif :", list(tickers.keys()))
ticker = tickers[ticker_label]

# Sélecteur de période
period = st.selectbox("Période", ["6mo", "1y", "2y", "5y"], index=1)

data = yf.download(ticker, period=period)['Close']

st.write("Données brutes récupérées :", data.head())  # Debug

if data.empty:
    st.error(f"❌ Aucune donnée trouvée pour {ticker}")
    st.stop()

else:
    st.subheader(f"Données historiques : {ticker_label}")
    st.line_chart(data)

    # Choix du modèle
    model_choice = st.radio("Choisissez la méthode de prévision :", ["ARIMA", "Régression Linéaire (ML simple)"])
    steps = st.slider("Nombre de jours à prédire :", 5, 60, 30)

    forecast = None

    if model_choice == "ARIMA":
        try:
            model = ARIMA(data, order=(5,1,0))
            fit = model.fit()
            forecast = fit.forecast(steps=steps)
        except Exception as e:
            st.error(f"Erreur ARIMA : {e}")

    elif model_choice == "Régression Linéaire (ML simple)":
        try:
            X = np.arange(len(data)).reshape(-1,1)
            y = data.values
            model = LinearRegression()
            model.fit(X, y)
            future_X = np.arange(len(data), len(data)+steps).reshape(-1,1)
            forecast = pd.Series(model.predict(future_X),
                                 index=pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=steps))
        except Exception as e:
            st.error(f"Erreur Régression : {e}")

    if forecast is not None:
        st.subheader("📊 Prédiction")
        result = pd.concat([data, forecast.rename("Prévision")], axis=1)
        st.line_chart(result)
