import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter

# Carregar os dados
data = pd.read_csv('dados.csv', delimiter=';')

# Converter vírgulas para pontos e depois para float
data['A_mod'] = data['A_mod'].astype(str).str.replace(',', '.').astype(float)

# Tratar valores NaN na coluna A_mod com a média
mean_A_mod = data['A_mod'].mean()
data['A_mod'].fillna(mean_A_mod, inplace=True)

# Converter a coluna 'data' para datetime
data['data'] = pd.to_datetime(data['data'], format='%d/%m/%Y')

# Preparar os dados para a rede neural
X = data['A_mod'].values.reshape(-1, 1)
y = data['A_mod'].values.reshape(-1, 1)

# Normalizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.transform(y)

# Dividir em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Definir o modelo da rede neural com dropout
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dropout(0.2),  # Dropout layer added
    Dense(64, activation='relu'),
    Dropout(0.2),  # Dropout layer added
    Dense(1)
])

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Prever os valores para preencher A_mod
data['A_mod_filled'] = scaler.inverse_transform(model.predict(X_scaled))

# Calcular R² e MAE
r2 = r2_score(data['A_mod'], data['A_mod_filled'])
mae = mean_absolute_error(data['A_mod'], data['A_mod_filled'])

# Plotar os resultados como no exemplo de ARIMA

# Plotar os resultados
plt.figure(figsize=(12, 6))
plt.plot(data['data'], data['A_mod'], label='A_mod Original')
plt.plot(data['data'], data['A_mod_filled'], label='A_mod Preenchido', linestyle='--')
plt.xlabel('Tempo')
plt.ylabel('Precipitação Mensal Acumulada (mm)')
plt.title('Preenchimento de A_mod com Rede Neural')

# Configurar o eixo x para mostrar apenas o ano
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))

plt.legend()
plt.tight_layout()

# Inserir R² e MAE na lateral direita do gráfico
plt.text(1.01, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=12)
plt.text(1.01, 0.90, f'MAE = {mae:.4f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=12)

plt.show()
