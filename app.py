import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Stock Prediction App", page_icon="📈")

st.title("📈 Stock Market Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, index_col='Date', parse_dates=True)

    st.subheader("Dataset Preview")
    st.write(data.head())

    # Select features
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X = []
    y = []

    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, [0, 3]])

    X, y = np.array(X), np.array(y)

    # Train model
    if st.button("Train Model"):
        model = keras.models.Sequential([
            keras.layers.LSTM(50, return_sequences=True, input_shape=(X.shape[1], 5)),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50, return_sequences=True),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(50),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2)
        ])

        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, batch_size=32)

        st.success("Model trained successfully!")

        # Prediction
        predictions = model.predict(X)

        st.subheader("Prediction Output")
        st.write(predictions[:10])

        # Plot
        actual_open = y[:, 0]
        predicted_open = predictions[:, 0]

        df_plot = pd.DataFrame({
            "Actual Open": actual_open,
            "Predicted Open": predicted_open
        })

        st.line_chart(df_plot)
