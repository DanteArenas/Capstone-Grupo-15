# este archivo juntará todo, se llamarán a las funciones de cada archivo y se generarán las simulaciones
from ruteo import generar_rutas, graficar_rutas
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo
import os

base_dir = os.path.dirname(__file__)
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_camiones = os.path.join(base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')

for dia in range(1, 4):
    path_venta_zona = os.path.join(
        base_dir, 'ventas_zona_estocasticas', f'venta_zona_dia_{dia}.csv')

    # Generar rutas
    data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                    path_flota, path_camiones, path_productos, dia)

    # Graficar rutas
    graficar_rutas(data_resultados, path_zonas, path_tiendas, dia)

    # Cargar datos de distancia
    df_distancias_ordenada = procesar_datos_de_distancia(data_resultados)

    # Generar matriz Ck
    matriz_ck = generar_matriz_ck(df_distancias_ordenada, data_resultados)

    # Resolver el precio óptimo
    df_precios_optimos = resolver_precio_optimo(matriz_ck)

    path_resultados = os.path.join(
        base_dir, 'resultados', f'dia_{dia}', f'resultados_dia_{dia}.csv')
