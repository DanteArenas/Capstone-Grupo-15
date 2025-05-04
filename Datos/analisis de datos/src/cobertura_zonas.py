from matplotlib.patches import Patch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# Cargar datos
df_zonas = pd.read_csv('zonas_20250115.csv')
df_zonas['tienda_zona'] = df_zonas['tienda_zona'].fillna(-1).astype(int)

# Crear tabla de zonas
grilla_tiendas = df_zonas.pivot_table(
    index='y_zona',
    columns='x_zona',
    values='tienda_zona',
    fill_value=-1
)


zonas_unicas = sorted(grilla_tiendas.stack().unique())
print(zonas_unicas)

colormap = plt.get_cmap('tab20', len(zonas_unicas))
color_map = ListedColormap([colormap(i) for i in range(len(zonas_unicas))])
norm = BoundaryNorm(zonas_unicas + [max(zonas_unicas)+1], color_map.N)

grilla_indices = grilla_tiendas
print(grilla_tiendas)

# Dibujar grilla
plt.figure(figsize=(12, 10))
im = plt.imshow(grilla_indices,  cmap=color_map, norm=norm)

df_tiendas = pd.read_csv('tiendas_20250115.csv')

# Ajustar para que indices empiecen en cero
df_tiendas['pos_x'] -= 1
df_tiendas['pos_y'] -= 1

x_max = grilla_tiendas.columns.max()
y_max = grilla_tiendas.index.max()
x_min = grilla_tiendas.columns.min()
y_min = grilla_tiendas.index.min()
print(f"X: {x_min} - {x_max} Y: {y_min} - {y_max}")
print(grilla_tiendas.iloc[29])

for tienda in df_tiendas.itertuples():

    plt.text(tienda.pos_x, tienda.pos_y, str(tienda.id_tienda),
             ha='center', va='center', color='white', fontsize=8)
    # if tamano == 'grande':
    #     plt.scatter(x, y, s=100, c='black', marker='o', alpha=0.5)
    # elif tamano == 'pequena':
    #     plt.scatter(x, y, s=50, c='black', marker='o', alpha=0.5)
    # elif tamano == 'mediana':
    #     plt.scatter(x, y, s=75, c='black', marker='o', alpha=0.5)

plt.title('Cobertura actual de tiendas por zona')
plt.xlabel('pos_x')
plt.ylabel('pos_y')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
