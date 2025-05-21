import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
from itertools import combinations

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


# Se calcula la matriz de distancias entre zonas y tiendas
dist_matrix = cdist(coords_zonas, coords_tiendas, metric='euclidean')

# Se convierte a df para facilitar lectura
dist_df = pd.DataFrame(
    dist_matrix,
    index=zonas_data['id_zona'],
    columns=tiendas_data['id_tienda']
)

print(dist_df.head())

print("tiendas_data")
print(tiendas_data.head())

dist_matrix = cdist(coords_zonas, coords_zonas, metric='euclidean')


# Se calcula la matríz de distancias eucledianas entre solo zonas.
dist_df = pd.DataFrame(
    dist_matrix,
    index=zonas_data['id_zona'],
    columns=zonas_data['id_zona']
)

print(dist_df.head())


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

print("Datos de demanda por zona con volumen:")
clientes_productos_data = pd.merge(
    clientes_1_data, productos_data, on='id_producto', how='inner')
clientes_productos_data = clientes_productos_data.drop(
    columns=[col for col in clientes_productos_data.columns if "Unnamed" in col])
print("clientes_productos_data:")
print(clientes_productos_data.head())

zonas_datos = pd.merge(clientes_productos_data, zonas_data, on='id_zona')
zonas_datos = zonas_datos.drop(
    columns=[col for col in zonas_datos.columns if "Unnamed" in col])

print("Datos de zonas con demanda:")
print(zonas_datos.head())
# De la forma: id_zona  id_producto  venta_digital  volumen  x_zona  y_zona  tienda_zona

print("Datos de tiendas:")
for index, row in tiendas_data.iterrows():
    print(f"Fila {index}:")
    print(row)
    print("-" * 40)
# id_tienda  tipo_tienda  pos_x   pos_y

# === Agrupar por tienda física ===
tiendas = zonas_datos['tienda_zona'].unique()
rutas_totales = {}

demanda_insatisfecha = pd.DataFrame()
# Demanda insatisfecha por tienda

# Algoritmo Clarke-Wright

for index, row in tiendas_data.iterrows():
    tienda = row['id_tienda']
    print(f"\nProcesando tienda: {tienda}")
    # row de la forma: id_tienda tipo_tienda pos_x  pos_y
    # print(row)

    # Subconjunto de zonas asociadas a esta tienda
    sub_zonas = zonas_datos[zonas_datos['tienda_zona'] == tienda].copy()
    sub_zonas = sub_zonas.reset_index(drop=True)
    # sub_zonas de la forma: id_zona  id_producto  venta_digital  volumen  x_zona  y_zona  tienda_zona

    # Obtener tipo y cantidad de camiones para la tienda
    flota_info = flota_data[flota_data['id_tienda'] == tienda]
    if flota_info.empty:
        print(f"No hay datos de flota para tienda {tienda}, se omite.")
        continue

    # flota_info tiene solo una fila
    id_camion = flota_info.iloc[0]['id_camion']
    n_camiones = flota_info.iloc[0]['N']
    capacidad = camiones_data.loc[camiones_data['tipo_camion']
                                  == id_camion, 'Q'].values[0]

    # Establecer el depósito como primer punto
    deposito_coord = row[['pos_x', 'pos_y']].values.astype(float)

    print(f"Coordenadas del depósito: {deposito_coord}")

    deposito_filas = sub_zonas[
        (sub_zonas['x_zona'] == deposito_coord[0]) & (
            sub_zonas['y_zona'] == deposito_coord[1])
    ]
    print(f"Filas del depósito: {deposito_filas}")

    if not deposito_filas.empty:
        # Eliminar las filas correspondientes depósito de sub_zonas para evitar duplicados
        sub_zonas = sub_zonas.drop(deposito_filas.index)
    else:
        raise ValueError("El depósito no está en sub_zonas.")

    sub_zonas_agrupadas = sub_zonas.groupby('id_zona').agg({
        'venta_digital': 'sum',
        'x_zona': 'first',
        'y_zona': 'first'
    }).reset_index()

    print(f"Sub zonas agrupadas: {sub_zonas_agrupadas}")
    # sub_zonas_agrupadas de la forma: id_zona  venta_digital  x_zona  y_zona

    coords = np.vstack(
        [deposito_coord, sub_zonas_agrupadas[['x_zona', 'y_zona']].values])
    # coords de la forma: [x_zona, y_zona] del depósito y de las zonas

    volumen_total_zona_deposito = 0
    for indice_deposito_filas, fila_deposito_filas in deposito_filas.iterrows():
        # suma el volumen total de la zona de depósito, por producto
        volumen_total_zona_deposito += fila_deposito_filas['volumen'] * \
            fila_deposito_filas['venta_digital']

    #     print(f"row['volumen']: {row['volumen']}")
    #     print(f"row['venta_digital']: {row['venta_digital']}")
    #     print(f"volumen_total_zona_deposito: {volumen_total_zona_deposito}")

    sub_zonas['volumen_total'] = sub_zonas['volumen'] * \
        sub_zonas['venta_digital']  # Volumen total de cada producto (fila)
    # El volumen total de los productos a repartir en cada zona, sin la del depísito(creo)

    carga_agrupada_por_zona = sub_zonas.groupby(
        'id_zona')['volumen_total'].sum().reset_index()
    print(f"carga_agrupada_por_zona: {carga_agrupada_por_zona}")
    # carga_agrupada_por_zona de la forma: id_zona  volumen_total

    # carga_ruta = np.concatenate(np.array([carga_zona_deposito]),carga_agrupada_por_zona['volumen'].values)
    print(f"deposito_filas: {deposito_filas}")
    print(f"numero de zonas: {len(sub_zonas_agrupadas)}")
    print(f"largo carga_agrupada_por_zona: {len(carga_agrupada_por_zona)}")

    dist = cdist(coords, coords)
    # dist de la forma: matriz de distancias euclidianas entre el depósito y las zonas

    # carga ruta con volumen total de la zona de depósito primero
    # y luego el volumen total de cada zona
    carga_ruta = np.concatenate(
        ([volumen_total_zona_deposito], carga_agrupada_por_zona['volumen_total'].values))
    print(f"carga_ruta: {carga_ruta}")
    print(f"coords: {coords}")
    print(f"dist: {dist}")

    print("largo nodos (coords):", coords.shape[0])
    print("largo carga_ruta:", carga_ruta.shape[0])

    rutas = {i: [0, i, 0] for i in range(1, len(coords))}
    # rutas de la forma: {i: [0, i, 0]} donde i es el índice de la zona en coords, sin contar el depósito
    # print(f"rutas: {rutas}")

    savings = []

    for i, j in combinations(range(1, len(coords)), 2):
        # Calcular el ahorro de distancia al unir las rutas i y j
        # dist[0, i] es la distancia del depósito a la zona i
        # dist[0, j] es la distancia del depósito a la zona j
        # dist[i, j] es la distancia entre las zonas i y j
        # El ahorro es la distancia del depósito a i + la distancia del depósito a j - la distancia entre i y j
        s = dist[0, i] + dist[0, j] - dist[i, j]
        savings.append((s, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        ruta_i = next((r for r in rutas.values() if i in r[1:-1]), None)
        ruta_j = next((r for r in rutas.values() if j in r[1:-1]), None)

        if ruta_i is None or ruta_j is None or ruta_i == ruta_j:
            continue

        carga_i = sum(carga_ruta[k] for k in ruta_i if k != 0)
        carga_j = sum(carga_ruta[k] for k in ruta_j if k != 0)
        if carga_i + carga_j > capacidad:
            continue

        if ruta_i[-2] == i and ruta_j[1] == j:
            nueva_ruta = ruta_i[:-1] + ruta_j[1:]
        elif ruta_j[-2] == j and ruta_i[1] == i:
            nueva_ruta = ruta_j[:-1] + ruta_i[1:]
        else:
            continue

        rutas = {k: v for k, v in rutas.items() if v != ruta_i and v != ruta_j}
        rutas[i] = nueva_ruta
        if tienda in [7, 10]:
            print(f"rutas actuales tienda {tienda}:")
            print(rutas)

    # === Seleccionar como máximo N rutas con menor distancia total ===
    rutas_finales = []
    id_zona_tienda = deposito_filas.iloc[0]['id_zona']
    print(f"ID zona tienda: {id_zona_tienda}")
    for ruta in rutas.values():
        zonas_ruta = [sub_zonas_agrupadas.iloc[i - 1]['id_zona']
                      if i != 0 else id_zona_tienda for i in ruta]
        carga = sum(carga_ruta[i] for i in ruta if i != 0)
        distancia = sum(dist[ruta[k]][ruta[k+1]] for k in range(len(ruta)-1))
        rutas_finales.append({
            'ruta': zonas_ruta,
            'carga': carga,
            'distancia': round(distancia, 2)
        })

    rutas_finales.sort(key=lambda r: r['distancia'])  # priorizar rutas cortas
    rutas_finales = rutas_finales[:n_camiones]  # Limitar a N rutas

    rutas_totales[tienda] = rutas_finales
    # rutas_totales[tienda] = rutas_finales[:n_camiones]  # Limitar a N rutas

    # 1. Recolectar demanda total original por zona
    demanda_por_zona = sub_zonas_agrupadas[['id_zona', 'venta_digital']].copy()
    demanda_por_zona.rename(columns={'venta_digital': 'demanda'}, inplace=True)

    # 2. Listado de zonas visitadas
    zonas_visitadas = set()
    for r in rutas_finales:
        zonas_visitadas |= set(r['ruta'])

    # 3. Zonas sin cubrir (demanda insatisfecha)
    demanda_insat_tienda = demanda_por_zona[
        ~demanda_por_zona['id_zona'].isin(zonas_visitadas)
    ].copy()
    demanda_insat_tienda['tienda'] = tienda
    print("Zonas con demanda insatisfecha:")
    print(demanda_insat_tienda)

    # 4) Acumular en un DataFrame global
    demanda_insatisfecha = pd.concat(
        [demanda_insatisfecha, demanda_insat_tienda],
        ignore_index=True
    )


# === Mostrar resultados ===
resultados = []
n_camiones_disponibles_por_tienda = {}
for index, row in flota_data.iterrows():
    tienda = row['id_tienda']
    n_camiones_disponibles_por_tienda[tienda] = row['N']

for tienda, rutas in rutas_totales.items():
    print(f"\n Rutas desde tienda: {tienda}")
    for i, r in enumerate(rutas):
        print(
            f"  Ruta {i+1}: {r['ruta']}, Carga: {r['carga']}, Distancia: {r['distancia']}")
    resultados.append({
        'tienda': tienda,
        'rutas': [r['ruta'] for r in rutas],
        'carga': [r['carga'] for r in rutas],
        'distancia': [r['distancia'] for r in rutas],
        'carga_total': sum(r['carga'] for r in rutas),
        'n_camiones_utilizados': len(rutas),
        'n_camiones_disponibles': n_camiones_disponibles_por_tienda[tienda]
    })


print("\nResultados finales:")
print(resultados)

df_resultados = pd.DataFrame(resultados)
print(df_resultados)

# Guardar resultados en CSV
output_path = os.path.join(
    base_dir, '..', 'resultados', 'resultados_CW_v2.csv')
df_resultados.to_csv(output_path, index=False)

demanda_insatisfecha_path = os.path.join(
    base_dir, '..', 'resultados', 'demanda_insatisfecha_CW_v2.csv')
demanda_insatisfecha.to_csv(demanda_insatisfecha_path, index=False)
