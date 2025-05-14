import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

base_dir = os.path.dirname(__file__)
path_demanda_online_insatisfecha = os.path.join(
    base_dir, '..', '..', 'Datos', 'demanda_online_insatisfecha_20250115.csv')

# Cargar los datos de demanda online insatisfecha
df_demanda_online = pd.read_csv(path_demanda_online_insatisfecha)
slope = df_demanda_online['slope'].values[0]

print(f"Pendiente de costo de externalización: {slope}")
# Graficar la pendiente de costo de externalización

x = np.linspace(0, 100, 100)
y = slope * x
plt.plot(x, y, label=f'y = {slope:.2f}x', color="blue")
plt.title("Costo de externalización de demanda online insatisfecha")
plt.xlabel("Demanda online insatisfecha")
plt.ylabel("Costo de externalización")
plt.grid(True)
plt.legend()
plt.show()
