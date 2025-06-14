from ruteo import generar_rutas
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo
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
