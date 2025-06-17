from ruteo import generar_rutas
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo, resolver_precio_optimo_inv, resolver_precio_optimo_zona, zona_vehiculo, caso_base_1, caso_base_2
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

print()
print()
print()
print()
print()
print()
print()
print()

base_dir = os.path.dirname(__file__)
df_distancias_ordenada, kmeans = procesar_datos_de_distancia(data_resultados)
path_stock = "Datos/analisis de datos/stock_diario.csv"
df_stock = pd.read_csv(path_stock)
path_demanda = "Entrega 3/ventas_digitales_estocasticas/venta_zona_estocastica_dia_1.csv"
df_demanda = pd.read_csv(path_demanda)
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
df_zonas = pd.read_csv(path_zonas)
df_zona_vehiculo = zona_vehiculo(data_resultados, df_zonas)
print(df_zona_vehiculo.head())

print(matriz_ck.head())
print(df_stock.tail())
print(df_demanda.head())
print(df_zonas.head())

df_precios_por_zona = resolver_precio_optimo_zona(matriz_ck, df_zona_vehiculo, df_stock, df_demanda, df_zonas, dia=1)
print(df_precios_por_zona)


# Obtener listas de valores únicos por columna
unique_values = {col: df_precios_por_zona[col].unique().tolist() for col in df_precios_por_zona.columns}

# Mostrar los valores únicos por columna
for col, values in unique_values.items():
    print(f"{col}: {values}")

df_base_0 = caso_base_1(matriz_ck, df_zonas, dia=1)
df_base_t = caso_base_2(df_demanda, dia=1)
