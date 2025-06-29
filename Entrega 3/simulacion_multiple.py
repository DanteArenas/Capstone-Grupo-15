# este archivo juntará todo, se llamarán a las funciones de cada archivo y se generarán las simulaciones
from ruteo import generar_rutas, graficar_rutas, mejorar_rutas_2_opt, caso_base_ruteo
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo, resolver_precio_optimo_zona, zona_vehiculo
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

path_stock_diario_caso_base = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'stock_diario_caso_base.csv')

df_stock_diario_caso_base = pd.read_csv(path_stock_diario_caso_base)
df_zonas = pd.read_csv(path_zonas)


for i in range(0, 6):
    n_dias = 40

    path_base_stock_diario = os.path.join(
        base_dir, '..', 'Datos', 'analisis de datos', 'inventario_escenarios')
    if i == 0:
        path_stock_diario = os.path.join(
            path_base_stock_diario, 'semilla_42', 'stock_diario.csv')
    if i == 1:
        path_stock_diario = os.path.join(
            path_base_stock_diario, 'semilla_101', 'stock_diario.csv')
    if i == 2:
        path_stock_diario = os.path.join(
            path_base_stock_diario, 'semilla_202', 'stock_diario.csv')
    if i == 3:
        path_stock_diario = os.path.join(
            path_base_stock_diario, 'semilla_303', 'stock_diario.csv')
    if i == 4:
        path_stock_diario = os.path.join(
            path_base_stock_diario, 'semilla_404', 'stock_diario.csv')
    if i == 5:
        path_stock_diario = os.path.join(
            path_base_stock_diario, 'semilla_505', 'stock_diario.csv')

    df_stock_diario = pd.read_csv(path_stock_diario)

    for dia in range(1, n_dias + 1):
        path_venta_zona = os.path.join(
            base_dir, f'ventas_digitales_realizacion_{i}', f'venta_zona_estocastica_dia_{dia}.csv')

        # Generar rutas
        data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                        path_flota, path_camiones, path_productos, dia, id_realizacion=i)

        # Graficar rutas
        graficar_rutas(data_resultados, path_zonas,
                       path_tiendas, dia, id_realizacion=i)

        # Mejorar rutas con 2-opt
        data_resultados_mejorada = mejorar_rutas_2_opt(
            data_resultados, path_zonas, path_tiendas, dia, id_realizacion=i)

        # Graficar rutas mejoradas
        graficar_rutas(data_resultados_mejorada, path_zonas, path_tiendas,
                       dia, mejora_2_opt=True, id_realizacion=i)

        # Cargar datos de distancia
        df_distancias_ordenada, kmeans = procesar_datos_de_distancia(
            data_resultados_mejorada)

        # Generar matriz ck
        matriz_ck = generar_matriz_ck(
            df_distancias_ordenada, data_resultados_mejorada)

        # Obtener df_zona_vehiculo
        df_zona_vehiculo = zona_vehiculo(data_resultados_mejorada, df_zonas)

        # Resolver precio óptimo
        path_demanda_digital = os.path.join(
            base_dir, f'ventas_digitales_realizacion_{i}', f'venta_zona_estocastica_dia_{dia}.csv')
        df_demanda_digital = pd.read_csv(path_demanda_digital)

        df_precios_optimos = resolver_precio_optimo_zona(
            matriz_ck, df_zona_vehiculo, df_stock_diario, df_demanda_digital, df_zonas, dia)

        # Guardar resultados
        path_resultados = os.path.join(
            base_dir, f'realizacion_{i}', 'resultados', f'dia_{dia}', f'resultados_pricing_dia_{dia}.csv')

        if not os.path.exists(os.path.dirname(path_resultados)):
            os.makedirs(os.path.dirname(path_resultados))

        df_precios_optimos.to_csv(path_resultados, index=False)

    # KPIs
    path_distancias_dias_cw_solo = creacion_df_40_dias_ruteo(
        n_dias, caso_base=False, mejorados=False, cw_solo=True, id_realizacion=i)
    distancias_promedio_cw_solo = distancia_promedio(
        path_distancias_dias_cw_solo)
    print(f"Distancia promedio CW solo: {distancias_promedio_cw_solo}")

    path_distancias_dias_mejorados = creacion_df_40_dias_ruteo(
        n_dias, caso_base=False, mejorados=True, cw_solo=False, id_realizacion=i)
    distancias_promedio_mejorados = distancia_promedio(
        path_distancias_dias_mejorados)
    print(f"Distancia promedio mejorados: {distancias_promedio_mejorados}")
