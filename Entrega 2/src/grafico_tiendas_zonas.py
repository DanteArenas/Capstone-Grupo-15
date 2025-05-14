import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
# ---------- Mapa de tiendas de entrega 1 ---------- #
base_dir = os.path.dirname(__file__)
path_tiendas = os.path.join(
    base_dir, '..', '..', 'Datos', 'tiendas_20250115.csv')
df = pd.read_csv(path_tiendas)
df['pos_x'] = df['pos_x'].astype(int)
df['pos_y'] = df['pos_y'].astype(int)
df['pos_x'] -= 1
df['pos_y'] -= 1
path_zonas = os.path.join(
    base_dir, '..', '..', 'Datos', 'zonas_20250115.csv')
df_zonas = pd.read_csv(path_zonas)

largo_x = max(df_zonas['x_zona'])
largo_y = max(df_zonas['y_zona'])

tiendas_pequenas = df[df['tipo_tienda'] == 'pequena']
tiendas_medianas = df[df['tipo_tienda'] == 'mediana']
tiendas_grandes = df[df['tipo_tienda'] == 'grande']

plt.xlim(0-1, (df_zonas['x_zona'].max())+1)
plt.ylim(0-1, (df_zonas['y_zona'].max())+1)

plt.scatter(tiendas_pequenas['pos_x'], tiendas_pequenas['pos_y'],
            c='blue', marker='o', label='Tienda pequeña', zorder=3)
plt.scatter(tiendas_medianas['pos_x'], tiendas_medianas['pos_y'],
            c='green', marker='o', label='Tienda mediana', zorder=3)
plt.scatter(tiendas_grandes['pos_x'], tiendas_grandes['pos_y'],
            c='red', marker='o', label='Tienda grande', zorder=3)

plt.title('Tiendas y cobertura de zonas')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(range(largo_x+1), labels=[])
plt.yticks(range(largo_y+1), labels=[])

plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().set_aspect('equal', adjustable='box')  # Chatgpt
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

# ---------- Gráfico de las zonas (de entrega pasada, con ayuda dde Chatgpt) ---------- #

# Crear tabla de zonas
grilla_tiendas = df_zonas.pivot_table(
    index='y_zona',
    columns='x_zona',
    values='tienda_zona',
    fill_value=-1
)

zonas_unicas = sorted(grilla_tiendas.stack().unique())

colormap = plt.get_cmap('tab20', len(zonas_unicas))
color_map = ListedColormap([colormap(i) for i in range(len(zonas_unicas))])
norm = BoundaryNorm(zonas_unicas + [max(zonas_unicas)+1], color_map.N)

grilla_indices = grilla_tiendas
print(grilla_tiendas)

# Dibujar grilla
im = plt.imshow(grilla_indices,  cmap=color_map,
                norm=norm, zorder=1, alpha=0.5)
plt.tight_layout()
plt.show()
print(
    f"Distancia total recorrida por todos los camiones: {distancia_total} unidades")
