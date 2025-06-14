from ruteo import generar_rutas
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo, resolver_precio_optimo_inv
import os

# path_zonas, path_tiendas, path_venta_zona, path_flota, path_camiones, path_producto, dia
base_dir = os.path.dirname(__file__)
print()
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_venta_zona_1 = os.path.join(
    base_dir, '..', 'Datos', 'venta_zona_1_20250115.csv')
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_camiones = os.path.join(base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')

dia = 1

data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona_1,
                                path_flota, path_camiones, path_productos, dia)
print(data_resultados.head())

df_distancias_ordenada, kmeans = procesar_datos_de_distancia(data_resultados)

matriz_ck = generar_matriz_ck(df_distancias_ordenada, data_resultados)

df_precios_optimos = resolver_precio_optimo(matriz_ck)

print(df_precios_optimos)

import pandas as pd


df_distancias_ordenado = pd.DataFrame({
    'tienda': ['A', 'A', 'B', 'B'],
    'vehiculo': [0, 1, 0, 1],
    'distancia': [15.0, 25.0, 30.0, 35.0],
    'cluster_ordenado': [0, 0, 1, 1],
    'n_k (total clientes)': [100, 150, 200, 250],
    'centroide': [15.5, 25.5, 30.5, 35.5],
    'c_k': [20.0, 22.0, 18.0, 21.0]
})

print(df_distancias_ordenado)
print()
print()
print()
print()
print()
print()
print()
print()

print("NUEVOS PRECIOS OPTIMOS")
df_stock = pd.DataFrame({
    'id_tienda': ['A', 'B'],
    'id_producto': [101, 102],
    'reorden': [500, 400],
    'stock_actual': [350, 300]
})


df_demanda = pd.DataFrame({
    'id_zona': [1, 2, 3, 4],
    'id_producto': [101, 102, 101, 102],
    'venta_digital': [1000, 1500, 2000, 2500]
})



df_zonas = pd.DataFrame({
    'id_zona': [1, 2, 3, 4],
    'x_zona': [10, 12, 15, 20],
    'y_zona': [20, 22, 25, 30],
    'tienda_zona': ['A', 'A', 'B', 'B']
})

df_precios_tienda_ruta = resolver_precio_optimo_inv(df_distancias_ordenado, df_stock, df_demanda, df_zonas)
print(df_precios_tienda_ruta)