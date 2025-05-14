from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import matplotlib.pyplot as plt
import os

# Construir la ruta relativa basada en la ubicación del script
base_dir = os.path.dirname(__file__)
path_camiones = os.path.join(
    base_dir, '..', 'resultados', 'rutas_camiones_caso_base_v2.csv')

# Verificar la ruta absoluta y la existencia del archivo
absolute_path_camiones = os.path.abspath(path_camiones)
print(f"Ruta absoluta del archivo: {absolute_path_camiones}")

if not os.path.exists(absolute_path_camiones):
    raise FileNotFoundError(
        f"El archivo no existe en la ruta: {absolute_path_camiones}")

df = pd.read_csv(path_camiones)

distancia_total = 0

for tienda, df_tienda in df.groupby('tienda'):
    plt.figure()

    for camion, df_camion in df_tienda.groupby('camion'):
        xs = df_camion['x'].values
        ys = df_camion['y'].values
        zs = df_camion['id_zona'].values

        plt.plot(xs, ys, marker='o', label=f'Camión {camion}')

        for x, y, zona in zip(xs, ys, zs):
            plt.text(x, y, str(zona), fontsize=8, ha='right', va='bottom')
        distancia_camion = df_camion['distancia_total_recorrida_camion'].values[0]
        distancia_total += distancia_camion

    plt.title(f'Tienda {tienda} - Rutas de Camiones')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend(title='Camiones')
    plt.grid(True)
    plt.tight_layout()
    path_figura = os.path.join(
        base_dir, '..', 'resultados', 'graficos_caso_base_v2', f'rutas_camiones_tienda_{tienda}.png')
    plt.savefig(path_figura)
    # plt.show()
    plt.close()


# ---------- Gráfico de todas las tiendas y camiones ---------- #

plt.figure(constrained_layout=True)
for tienda, df_tienda in df.groupby('tienda'):
    for camion, df_camion in df_tienda.groupby('camion'):
        xs = df_camion['x'].values
        ys = df_camion['y'].values
        plt.plot(xs, ys, zorder=2)

# ---------- Mapa de tiendas de entrega 1 ---------- #
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

plt.title('Rutas de Camiones - Caso Base')
plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(range(largo_x + 2), labels=[])
plt.yticks(range(largo_y + 2), labels=[])

plt.xticks(range(largo_x + 2), labels=[])
plt.yticks(range(largo_y + 2), labels=[])
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

plt.savefig(os.path.join(
    base_dir, '..', 'resultados', 'graficos_caso_base_v2', 'rutas_camiones_todas_las_tiendas.png'))
plt.show()
print(
    f"Distancia total recorrida por todos los camiones: {distancia_total} unidades")
