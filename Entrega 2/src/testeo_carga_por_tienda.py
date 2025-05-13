import pandas as pd
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

# === Calcular demanda por zona ===
# demanda_por_zona = clientes_1_data.groupby('id_zona')['venta_digital'].sum().reset_index()

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

print("zonas_datos:")
print(zonas_datos.head())

# Dict tienda, volumen a cubrir, capacidad camiones, lograble
res = {}
for i, row in tiendas_data.iterrows():
    tienda = row['id_tienda']
    print(f"\nProcesando tienda: {tienda}")

    sub_zonas = zonas_datos[zonas_datos['tienda_zona'] == tienda].copy()
    sub_zonas = sub_zonas.reset_index(drop=True)
    sub_zonas[['x_zona', 'y_zona']] = sub_zonas[[
        'x_zona', 'y_zona']].apply(pd.to_numeric)

    print(f"Subconjunto de zonas para la tienda {tienda}:")
    print(sub_zonas.head())

    # Calcular volumen total por tienda
    volumen_a_cubrir = 0
    for index, fila in sub_zonas.iterrows():
        suma_volumen_producto = fila['venta_digital'] * fila['volumen']
        volumen_a_cubrir += suma_volumen_producto

    print(
        f"Volumen total a cubrir para la tienda {tienda}: {volumen_a_cubrir}")

    flota_info = flota_data[flota_data['id_tienda'] == tienda]
    print(f"Flota para la tienda {tienda}:")
    print(flota_info)
    info_camiones_tienda = camiones_data[camiones_data['tipo_camion']
                                         == flota_info['id_camion'].values[0]]
    capacidad_cada_camion_tienda = info_camiones_tienda['Q'].values[0]
    print(f"Capacidad de cada camion para la tienda {tienda}:")
    print(capacidad_cada_camion_tienda)
    print(f"Capacidad total de despacho para la tienda {tienda}:")
    capacidad_despacho_tienda = flota_info['N'].values[0] * \
        capacidad_cada_camion_tienda
    print(capacidad_despacho_tienda)

    # Se guarda el resultado en un diccionario
    res[tienda] = {
        'volumen_a_cubrir': volumen_a_cubrir,
        'capacidad_despacho_tienda': capacidad_despacho_tienda,
        'lograble': volumen_a_cubrir <= capacidad_despacho_tienda
    }


print("Resultados:")
for tienda, resultado in res.items():
    print(f"Tienda {tienda}:")
    print(f"  Volumen a cubrir: {resultado['volumen_a_cubrir']}")
    print(
        f"  Capacidad de despacho: {resultado['capacidad_despacho_tienda']}")
    print(
        f"Diferencia: {resultado['capacidad_despacho_tienda'] - resultado['volumen_a_cubrir']}")
    print(f"  Lograble: {resultado['lograble']}")

# Ordenar resultados por diferencia de menor a mayor (copilot)
resultados_ordenados = sorted(res.items(
), key=lambda x: x[1]['capacidad_despacho_tienda'] - x[1]['volumen_a_cubrir'])


print()
print("Resultados ordenados por diferencia (menor a mayor):")
for tienda, resultado in resultados_ordenados:
    print(f"Tienda {tienda}:")
    print(f"  Volumen a cubrir: {resultado['volumen_a_cubrir']}")
    print(f"  Capacidad de despacho: {resultado['capacidad_despacho_tienda']}")
    print(
        f"  Diferencia: {resultado['capacidad_despacho_tienda'] - resultado['volumen_a_cubrir']}")
    print(f"  Lograble: {resultado['lograble']}")
