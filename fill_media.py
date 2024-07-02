import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# 1. Ler o CSV e tratar os dados
data = pd.read_csv('dados.csv', delimiter=';')

# Convertendo vírgulas para pontos e depois para float
data['A_mod'] = data['A_mod'].astype(str).str.replace(',', '.').astype(float)
data['A_original'] = data['A_original'].astype(str).str.replace(',', '.').astype(float)

# Convertendo a coluna 'data' para datetime
data['data'] = pd.to_datetime(data['data'], format='%d/%m/%Y')

# Função para preencher os dados faltantes usando a média dos dados
def fill_gaps_with_mean(series):
    # Preencher os gaps com a média dos dados
    filled_series = series.fillna(series.mean())
    return filled_series

# Certifique-se de que o DataFrame foi carregado corretamente
if 'data' in locals():
    # 2. Aplicar preenchimento com média para os gaps na coluna "A_mod"
    data['A_mod_filled'] = fill_gaps_with_mean(data['A_mod'])

    # 3. Calcular R² e MAE entre "A_mod_filled" e "A_original"
    r2 = r2_score(data['A_original'], data['A_mod_filled'])
    mae = mean_absolute_error(data['A_original'], data['A_mod_filled'])

    print(f'R²: {r2}')
    print(f'MAE: {mae}')

    # 4. Plotar os dados
    plt.figure(figsize=(12, 6))
    plt.plot(data['data'], data['A_original'], label='A_original')
    plt.plot(data['data'], data['A_mod_filled'], label='A_mod_filled', linestyle='--')
    plt.xlabel('Tempo')
    plt.ylabel('Precipitação Mensal Acumulada (mm)')
    plt.title('Dados Originais x Dados Preenchidos - Média dos Dados')
    plt.legend()

    # Inserir R² e MAE na lateral direita do gráfico
    plt.text(1.01, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=12)
    plt.text(1.01, 0.90, f'MAE = {mae:.4f}', transform=plt.gca().transAxes, verticalalignment='top', fontsize=12)

    plt.tight_layout()
    plt.show()
else:
    print("Erro ao carregar os dados.")
