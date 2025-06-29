# este archivo juntará todo, se llamarán a las funciones de cada archivo y se generarán las simulaciones
from ruteo import generar_rutas, graficar_rutas, mejorar_rutas_2_opt, caso_base_ruteo
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo, caso_base_1, zona_vehiculo, caso_base_2
from kpis import utilidad_total, distancia_promedio, utilidad_de_entregas, ventas_perdidas, demanda_online_insatisfecha, creacion_df_40_dias_ruteo
import pandas as pd
import os

base_dir = os.path.dirname(__file__)
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_camiones = os.path.join(base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')
path_stock_diario = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'stock_diario.csv')
path_stock_diario_caso_base = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'stock_diario_caso_base.csv')

df_stock_diario = pd.read_csv(path_stock_diario)
df_stock_diario_caso_base = pd.read_csv(path_stock_diario_caso_base)
df_zonas = pd.read_csv(path_zonas)

n_dias = 40
for i in range(0, 6):

    for dia in range(1, n_dias + 1):
        path_venta_zona = os.path.join(
            base_dir, f'ventas_digitales_realizacion_{i}', f'venta_zona_estocastica_dia_{dia}.csv')

        # Caso base
        data_resultados_caso_base = caso_base_ruteo(path_zonas, path_tiendas, path_venta_zona,
                                                    path_flota, path_camiones, path_productos, dia, id_realizacion=i)
        # Graficar rutas de caso base

        resultados_caso_base = os.path.join(
            base_dir, f'realizacion_{i}', 'resultados', f'dia_{dia}', 'caso_base', f'resultados_caso_base_dia_{dia}.csv')
        graficar_rutas(data_resultados_caso_base, path_zonas, path_tiendas,
                       dia, caso_base=True, id_realizacion=i)

        # Cargar datos de distancia
        df_distancias_ordenada, kmeans = procesar_datos_de_distancia(
            data_resultados_caso_base)

        # Generar matriz ck
        matriz_ck = generar_matriz_ck(
            df_distancias_ordenada, data_resultados_caso_base)

        # Resolver precio óptimo

        df_demanda_digital = pd.read_csv(path_venta_zona)

        df_precios_optimos_cb_1 = caso_base_1(
            matriz_ck, df_zonas, dia, id_realizacion=i)
        df_precios_optimos_cb_2 = caso_base_2(
            df_demanda_digital, dia, id_realizacion=i)
