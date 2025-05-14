import pandas as pd
import matplotlib.pyplot as plt
import os

# Construir la ruta relativa basada en la ubicación del script
base_dir = os.path.dirname(__file__)
path_camiones = os.path.join(
    base_dir, '..', 'resultados', 'rutas_camiones_caso_base.csv')

# Verificar la ruta absoluta y la existencia del archivo
absolute_path_camiones = os.path.abspath(path_camiones)
print(f"Ruta absoluta del archivo: {absolute_path_camiones}")

if not os.path.exists(absolute_path_camiones):
    raise FileNotFoundError(
        f"El archivo no existe en la ruta: {absolute_path_camiones}")

df = pd.read_csv(path_camiones)

for tienda, df_tienda in df.groupby('tienda'):
    plt.figure()

    for camion, df_camion in df_tienda.groupby('camion'):
        xs = df_camion['x'].values
        ys = df_camion['y'].values
        zs = df_camion['id_zona'].values

        plt.plot(xs, ys, marker='o', label=f'Camión {camion}')

        for x, y, zona in zip(xs, ys, zs):
            plt.text(x, y, str(zona), fontsize=8, ha='right', va='bottom')

    plt.title(f'Tienda {tienda} - Rutas de Camiones')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.legend(title='Camiones')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
