from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(__file__)
path_resultados_CW = os.path.join(
    base_dir, '..', 'resultados', 'resultados_CW.csv')
data_resultados = pd.read_csv(path_resultados_CW)
print(data_resultados.head())

path_zonas = os.path.join(base_dir, '..', '..', "Datos", "zonas_20250115.csv")
data_zonas = pd.read_csv(path_zonas)
data_zonas = data_zonas.drop(
    columns=[col for col in data_zonas.columns if "Unnamed" in col])
print(data_zonas.head())

distancia_total = 0
plt.figure(constrained_layout=True)
for i, row in data_resultados.iterrows():

    rutas = row['rutas']
    rutas = rutas.split("], [")
    for i in range(len(rutas)):
        rutas[i] = rutas[i].strip("[]").split(", ")
        for j in range(len(rutas[i])):
            rutas[i][j] = int(rutas[i][j])

    camion = 1
    for ruta in rutas:
        xs_camion = []
        ys_camion = []
        for id_zona in ruta:
            zona = data_zonas.loc[data_zonas['id_zona'] == id_zona]
            xs_camion.append(zona['x_zona'].values[0])
            ys_camion.append(zona['y_zona'].values[0])

        plt.plot(xs_camion, ys_camion, zorder=2)
        camion += 1

    distancia = row['distancia']
    lista_distancia = distancia.strip("[]").split(",")
    for i in range(len(lista_distancia)):
        distancia_total += float(lista_distancia[i])
    print(f"Distancia recorrida para la tienda {camion}: {distancia}")


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

plt.title('Rutas de Camiones - Clarke and Wright')
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
    base_dir, '..', 'resultados', 'graficos_CW', 'rutas_camiones_todas_las_tiendas_CW.png'))
plt.show()
print(
    f"Distancia total recorrida por todos los camiones: {distancia_total} unidades")
