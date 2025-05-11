import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os

# Se cargan los datos de zonas y tiendas
base_dir = os.path.dirname(__file__)
path_zonas = os.path.join(base_dir, '..', '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(
    base_dir, '..', '..', 'Datos', 'tiendas_20250115.csv')
zonas_data = pd.read_csv(path_zonas)
tiendas_data = pd.read_csv(path_tiendas)

# Se resta 1 a las coordenadas de las tiendas para que coincidan con el sistema de coordenadas de las zonas
tiendas_data['pos_x'] -= 1
tiendas_data['pos_y'] -= 1

coords_zonas = zonas_data[['x_zona', 'y_zona']].values
coords_tiendas = tiendas_data[['pos_x', 'pos_y']].values

path_venta_zona_1 = os.path.join(
    base_dir, '..', '..', 'Datos', 'venta_zona_1_20250115.csv')
clientes_1_data = pd.read_csv(path_venta_zona_1)
path_flota = os.path.join(base_dir, '..', '..', 'Datos', 'flota_20250115.csv')
flota_data = pd.read_csv(path_flota)
path_camiones = os.path.join(
    base_dir, '..', '..', 'Datos', 'vehiculos_20250115.csv')
camiones_data = pd.read_csv(path_camiones)
path_productos = os.path.join(
    base_dir, '..', '..', 'Datos', 'productos_20250115.csv')
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
print("Datos de demanda por zona con volumen:")
clientes_productos_data = pd.merge(
    clientes_1_data, productos_data, on='id_producto', how='inner')
clientes_productos_data = clientes_productos_data.drop(
    columns=[col for col in clientes_productos_data.columns if "Unnamed" in col])

print(clientes_productos_data.head())
# Zonas datos de la forma: id_zona  id_producto  venta_digital  volumen  x_zona  y_zona  tienda_zona
zonas_datos = pd.merge(clientes_productos_data,
                       zonas_data, on='id_zona', how='inner')
zonas_datos = zonas_datos.drop(
    columns=[col for col in zonas_datos.columns if "Unnamed" in col])


print("Datos de zonas con demanda:")
print(zonas_datos.head())

# === Agrupar por tienda física ===
tiendas = zonas_datos['tienda_zona'].unique()
rutas_totales = {}

camiones_rutas_dict = {}

demanda_insatisfecha_por_tienda = {}

for index, row in tiendas_data.iterrows():
    tienda = row['id_tienda']
    print(f"\nProcesando tienda: {tienda}")
    # row de la forma: id_tienda tipo_tienda pos_x  pos_y
    print(row)

    # Subconjunto de zonas asociadas a esta tienda
    # id_zona  id_producto  venta_digital  volumen  x_zona  y_zona  tienda_zona
    sub_zonas = zonas_datos[zonas_datos['tienda_zona'] == tienda].copy()
    sub_zonas = sub_zonas.reset_index(drop=True)
    sub_zonas[['x_zona', 'y_zona']] = sub_zonas[[
        'x_zona', 'y_zona']].apply(pd.to_numeric)

    # Obtener tipo y cantidad de camiones para la tienda
    flota_info = flota_data[flota_data['id_tienda'] == tienda]

    # flota_info es una fila: id_tienda  id_camion  N
    id_camion = flota_info.iloc[0]['id_camion']
    n_camiones = flota_info.iloc[0]['N'].astype(int)
    capacidad = camiones_data.loc[camiones_data['tipo_camion']
                                  == id_camion, 'Q'].values[0]

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

    id_zona_tienda = sub_zonas[(sub_zonas['x_zona'] == posicion_camion[0]) & (
        sub_zonas['y_zona'] == posicion_camion[1])].iloc[0]['id_zona']

    # print("Zona de la tienda:")
    # print(id_zona_tienda)

    zonas_unicas = sub_zonas[['id_zona', 'x_zona', 'y_zona', 'tienda_zona']].drop_duplicates(
        subset=['id_zona']).reset_index(drop=True)
    print("Zonas únicas:")
    print(zonas_unicas)

    # Inicializar las zonas visitadas
    visitadas = np.zeros(len(zonas_unicas), dtype=bool)

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

        if n_camion == 1:
            # ver demanda de la zona de la tienda
            info_zona_tienda = sub_zonas[(
                sub_zonas['id_zona'] == id_zona_tienda)]
            volumen_zona_tienda = 0
            for indice, rowtienda in info_zona_tienda.iterrows():
                volumen_zona_tienda += int(rowtienda['venta_digital']
                                           ) * int(rowtienda['volumen'])
            carga_camion += volumen_zona_tienda

        pos_actual_camion = posicion_camion
        distancia_total_camion = 0

        # Marcar la zona de la tienda visitada para los camiones
        visitadas[zonas_unicas['id_zona'] == id_zona_tienda] = True

        while carga_camion < int(capacidad) and not visitadas.all():
            # Calcular distancias solo para las zonas no visitadas (copilot)
            # print(
            #     f"Calculando distancias desde la posición actual del camión: {pos_actual_camion}")
            indices_no_visitadas = np.where(~visitadas)[0]  # Chatgpt
            distancias = zonas_unicas.loc[~visitadas, ['x_zona', 'y_zona']].apply(
                lambda fila: np.linalg.norm(pos_actual_camion - np.array([fila['x_zona'], fila['y_zona']])), axis=1
            )
            # print(f"Distancias calculadas")

            # Seleccionar la zona más cercana (Chatgpt)
            indice_min_dist_subset = distancias.idxmin()
            indice_real = indices_no_visitadas[list(
                distancias.index).index(indice_min_dist_subset)]
            fila = zonas_unicas.iloc[indice_real]

            # demanda_fila = fila['venta_digital']
            # volumen_producto = int(fila['volumen'])
            # volumen_fila = demanda_fila * volumen_producto

            info_zona = sub_zonas[(sub_zonas['id_zona'] == fila['id_zona'])]
            volumen_zona = 0
            for indice, row2 in info_zona.iterrows():
                volumen_zona += int(row2['venta_digital']
                                    ) * int(row2['volumen'])

            distancia = distancias[indice_min_dist_subset]

            # print(
            #     f"Zona más cercana: {fila['id_zona']}, cooord: {fila['x_zona'], fila['y_zona']}, distancia: {distancia}")

            if carga_camion + volumen_zona > capacidad:
                break

            # Agregar la zona al recorrido
            zonas_recorrido.append({
                'id_zona': fila['id_zona'],
                'x': fila['x_zona'],
                'y': fila['y_zona']
            })

            distancia_total_camion += distancia
            pos_actual_camion = np.array([fila['x_zona'], fila['y_zona']])
            carga_camion += volumen_zona

            # print(
            #     f"posición del camión: {pos_actual_camion}, carga: {carga_camion}")

            # Marcar la zona como visitada
            visitadas[indice_real] = True

        # Que el camion vuelva a la tienda después de recorrer las zonas
        print(f"Volviendo a la tienda: {tienda}")

        zonas_recorrido.append({
            'id_zona': id_zona_tienda,
            'x': row['pos_x'],
            'y': row['pos_y']
        })
        print(
            f"Zona recorrida (tienda): {id_zona_tienda}, coordenadas: {row['pos_x'], row['pos_y']}")

        distancia_total_camion += np.linalg.norm(
            pos_actual_camion - np.array([row['pos_x'], row['pos_y']]))
        camiones_rutas_dict[tienda]['rutas'][i]['zonas'] = zonas_recorrido
        camiones_rutas_dict[tienda]['rutas'][i]['carga'] = carga_camion
        camiones_rutas_dict[tienda]['rutas'][i]['distancia_recorrida'] = distancia_total_camion

    # Revisar demanda insatisfecha basada en el índice booleano `visitadas` (copilot)
    if not visitadas.all():
        demanda_insatisfecha = sub_zonas.loc[~visitadas].rename(
            columns={'venta_digital': 'unidades_pendientes'}
        )
        print(f"Demanda insatisfecha para tienda {tienda}")
        demanda_insatisfecha_por_tienda[tienda] = demanda_insatisfecha

for tienda, info in camiones_rutas_dict.items():
    print(f"Tienda {tienda} tiene {info['camiones']} camiones.")
    for ruta in info['rutas']:
        print(
            f"  Camión {ruta['camion']} - Carga: {ruta['carga']:.2f}, Zonas visitadas: {len(ruta['zonas'])}")

# Guardar rutas en un archivo CSV

resultados_dir = os.path.join(base_dir, '..', 'resultados')
os.chdir(resultados_dir)

rutas_output = []
# copilot:
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
df_rutas.to_csv('rutas_camiones_caso_base_v2.csv', index=False)

# Guardar demanda insatisfecha en un archivo CSV
if demanda_insatisfecha_por_tienda:
    df_demanda_insatisfecha = pd.concat(
        demanda_insatisfecha_por_tienda.values())
    df_demanda_insatisfecha.to_csv(
        'demanda_insatisfecha_por_tienda_caso_base_v2.csv', index=False)
else:
    print("No hay demanda insatisfecha para ninguna tienda.")
