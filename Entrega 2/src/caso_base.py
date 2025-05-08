import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
print(f"Directorio actual: {os.getcwd()}")
# Se cargan los datos de zonas y tiendas
path_zonas = os.path.join('..', '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join('..', '..', 'Datos', 'tiendas_20250115.csv')
path_zonas = r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Datos\zonas_20250115.csv'
path_tiendas = r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Datos\tiendas_20250115.csv'
zonas_data = pd.read_csv(path_zonas)
tiendas_data = pd.read_csv(path_tiendas)
# Se resta 1 a las coordenadas de las tiendas para que coincidan con el sistema de coordenadas de las zonas
tiendas_data['pos_x'] -= 1
tiendas_data['pos_y'] -= 1

coords_zonas = zonas_data[['x_zona', 'y_zona']].values
coords_tiendas = tiendas_data[['pos_x', 'pos_y']].values

print(coords_zonas[:5])
# Se calcula la matriz de distancias entre zonas y tiendas
dist_matrix = cdist(coords_zonas, coords_tiendas, metric='euclidean')

# Se convierte a df para facilitar lectura
dist_df = pd.DataFrame(
    dist_matrix,
    index=zonas_data['id_zona'],
    columns=tiendas_data['id_tienda']
)

print(dist_df.head())

path_venta_zona_1 = os.path.join(
    '..', '..', 'Datos', 'venta_zona_1_20250115.csv')
path_venta_zona_1 = r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Datos\venta_zona_1_20250115.csv'

clientes_1_data = pd.read_csv(path_venta_zona_1)
path_flota = os.path.join('..', '..', 'Datos', 'flota_20250115.csv')
path_flota = r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Datos\flota_20250115.csv'
flota_data = pd.read_csv(path_flota)
path_camiones = os.path.join('..', '..', 'Datos', 'vehiculos_20250115.csv')
path_camiones = r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Datos\vehiculos_20250115.csv'
camiones_data = pd.read_csv(path_camiones)
path_productos = os.path.join('..', '..', 'Datos', 'productos_20250115.csv')
path_productos = r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Datos\productos_20250115.csv'
productos_data = pd.read_csv(path_productos)

print("Datos de clientes:")
print(clientes_1_data.head())
print("Datos de flota:")
print(flota_data.head())
print("Datos de camiones:")
print(camiones_data.head())
print("Datos de productos:")
print(productos_data.head())

# === Calcular demanda por zona ===
# demanda_por_zona = clientes_1_data.groupby('id_zona')['venta_digital'].sum().reset_index()

print("Datos de demanda por zona con volumen:")
clientes_productos_data = pd.merge(
    clientes_1_data, productos_data, on='id_producto', how='inner')
clientes_productos_data = clientes_productos_data.drop(
    columns=[col for col in clientes_productos_data.columns if "Unnamed" in col])

print(clientes_productos_data.head())
# zonas_datos = pd.merge(demanda_por_zona, zonas_data, on='id_zona')
zonas_datos = pd.merge(clientes_productos_data,
                       zonas_data, on='id_zona', how='inner')
zonas_datos = zonas_datos.drop(
    columns=[col for col in zonas_datos.columns if "Unnamed" in col])


print("Datos de zonas con demanda:")
print(zonas_datos.head())

# print("Datos de tiendas:")
# for index, row in tiendas_data.iterrows():
#     print(f"Fila {index}:")
#     print(row)
#     print("-" * 40)  # Separador para mayor claridad

# === Agrupar por tienda física ===
tiendas = zonas_datos['tienda_zona'].unique()
rutas_totales = {}

camiones_rutas_dict = {}

demanda_insatisfecha_por_tienda = {}

for index, row in tiendas_data.iterrows():
    tienda = row['id_tienda']
    print(f"\nProcesando tienda: {tienda}")
    print(row)

    # Subconjunto de zonas asociadas a esta tienda
    sub_zonas = zonas_datos[zonas_datos['tienda_zona'] == tienda].copy()
    sub_zonas = sub_zonas.reset_index(drop=True)

    # Obtener tipo y cantidad de camiones para la tienda
    flota_info = flota_data[flota_data['id_tienda'] == tienda]
    if flota_info.empty:
        print(f"No hay datos de flota para tienda {tienda}, se omite.")
        continue

    id_camion = flota_info.iloc[0]['id_camion']
    n_camiones = flota_info.iloc[0]['N']
    capacidad = camiones_data.loc[camiones_data['tipo_camion']
                                  == id_camion, 'Q'].values[0]

    # print(sub_zonas)
    # print(demanda_por_zona)

    # Cargar el camión y definir rutas

    posicion_camion = row[['pos_x', 'pos_y']].values

    print(f"\nPosición inicial del camión: {posicion_camion}")
    print(f"\nCapacidad del camión: {capacidad}")
    camiones_rutas_dict[tienda] = {
        'posicion_tienda': posicion_camion,
        'camiones': n_camiones,
        'capacidad': capacidad,
        'rutas': [
            {'camion': i + 1, 'zonas': [], 'carga': 0, 'distancia_recorrida': 0} for i in range(n_camiones)
        ]
    }
    print('\n')
    if sub_zonas[(sub_zonas['x_zona'] == posicion_camion[0]) & (sub_zonas['y_zona'] == posicion_camion[1])].empty:
        print(
            f"No se encontró una zona para la posición de la tienda {tienda}.")
        continue
    id_zona_tienda = sub_zonas[(sub_zonas['x_zona'] == posicion_camion[0]) & (
        sub_zonas['y_zona'] == posicion_camion[1])].iloc[0]['id_zona']

    # print("Zona de la tienda:")
    # print(id_zona_tienda)

    for i in range(n_camiones):
        n_camion = i + 1
        carga_camion = 0
        contador = 0
        camiones_rutas_dict[tienda]['rutas'][i]['camion'] = n_camion

        zonas_recorrido = []

        zonas_recorrido.append({
            'id_zona': id_zona_tienda,
            'x': row['pos_x'],
            'y': row['pos_y']
        })
        pos_actual_camion = posicion_camion
        distancia_total_camion = 0

        index = 0
        while carga_camion < int(capacidad) and index < len(sub_zonas):

            fila = sub_zonas.iloc[index]

            # print(f"Fila seleccionada: {fila}")
            demanda_fila = fila['venta_digital']
            volumen_producto = int(fila['volumen'])
            volumen_fila = demanda_fila * volumen_producto

            if carga_camion + volumen_fila > capacidad:
                break

            if zonas_recorrido[-1]['id_zona'] == fila['id_zona']:
                # Si la zona ya fue visitada, se suma la demanda
                carga_camion += volumen_fila
                index += 1
                continue

            zonas_recorrido.append({
                'id_zona': fila['id_zona'],
                'x': fila['x_zona'],
                'y': fila['y_zona']
            })
            # Calcular distancia desde la posición actual del camión a la nueva zona
            distancia = np.linalg.norm(
                pos_actual_camion - np.array([fila['x_zona'], fila['y_zona']]))

            distancia_total_camion += distancia
            pos_actual_camion = np.array([fila['x_zona'], fila['y_zona']])
            carga_camion += volumen_fila
            index += 1

        sub_zonas = sub_zonas.iloc[index:].reset_index(drop=True)

        # Que el camion vuelva a la tienda
        zonas_recorrido.append({
            'id_zona': id_zona_tienda,
            'x': row['pos_x'],
            'y': row['pos_y']
        })
        distancia_total_camion += np.linalg.norm(
            pos_actual_camion - np.array([row['pos_x'], row['pos_y']]))
        camiones_rutas_dict[tienda]['rutas'][i]['zonas'] = zonas_recorrido
        camiones_rutas_dict[tienda]['rutas'][i]['carga'] = carga_camion
        camiones_rutas_dict[tienda]['rutas'][i]['distancia_recorrida'] = distancia_total_camion

    # Revisar demanda insatisfecha
    if not sub_zonas.empty:
        demanda_insatisfecha = sub_zonas.rename(
            columns={'venta_digital': 'unidades_pendientes'})
        print(f"Demanda insatisfecha para tienda {tienda}")
        demanda_insatisfecha_por_tienda[tienda] = demanda_insatisfecha

for tienda, info in camiones_rutas_dict.items():
    print(f"Tienda {tienda} tiene {info['camiones']} camiones.")
    for ruta in info['rutas']:
        print(
            f"  Camión {ruta['camion']} - Carga: {ruta['carga']:.2f}, Zonas visitadas: {len(ruta['zonas'])}")

# Guardar rutas en un archivo CSV

os.chdir(r'C:\Users\dante\Desktop\Capstone\Capstone-Grupo-15\Entrega 2\resultados')

rutas_output = []
for tienda, info in camiones_rutas_dict.items():
    for ruta in info['rutas']:
        for zona in ruta['zonas']:
            rutas_output.append({
                'tienda': tienda,
                'camion': ruta['camion'],
                'id_zona': zona['id_zona'],
                'x': zona['x'],
                'y': zona['y'],
                'carga_total_camion': ruta['carga'],
                'distancia_total_recorrida_camion': ruta['distancia_recorrida']
            })
df_rutas = pd.DataFrame(rutas_output)
df_rutas.to_csv('rutas_camiones.csv', index=False)

# Guardar demanda insatisfecha en un archivo CSV
if demanda_insatisfecha_por_tienda:
    df_demanda_insatisfecha = pd.concat(
        demanda_insatisfecha_por_tienda.values())
    df_demanda_insatisfecha.to_csv(
        'demanda_insatisfecha_por_tienda.csv', index=False)
else:
    print("No hay demanda insatisfecha para ninguna tienda.")
