# este archivo juntará todo, se llamarán a las funciones de cada archivo y se generarán las simulaciones
from ruteo import generar_rutas, graficar_rutas, mejorar_rutas_2_opt, caso_base_ruteo
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo
import os

base_dir = os.path.dirname(__file__)
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_camiones = os.path.join(base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')

for dia in range(1, 2):
    path_venta_zona = os.path.join(
        base_dir, 'ventas_digitales_estocasticas', f'venta_zona_estocastica_dia_{dia}.csv')

    # Generar rutas
    data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                    path_flota, path_camiones, path_productos, dia)

    # Graficar rutas
    graficar_rutas(data_resultados, path_zonas, path_tiendas, dia)

    # Mejorar rutas con 2-opt
    data_resultados_mejorada = mejorar_rutas_2_opt(
        data_resultados, path_zonas, path_tiendas, dia)

    # Graficar rutas mejoradas
    graficar_rutas(data_resultados_mejorada, path_zonas, path_tiendas,
                   dia, mejora_2_opt=True)

    # Cargar datos de distancia
    df_distancias_ordenada, kmeans = procesar_datos_de_distancia(
        data_resultados_mejorada)

    # Generar matriz ck
    matriz_ck = generar_matriz_ck(
        df_distancias_ordenada, data_resultados_mejorada)

    # Resolver precio óptimo
    df_precios_optimos = resolver_precio_optimo(matriz_ck)

    # Guardar resultados
    path_resultados = os.path.join(
        base_dir, 'resultados', f'dia_{dia}', f'resultados_pricing_dia_{dia}.csv')

    if not os.path.exists(os.path.dirname(path_resultados)):
        os.makedirs(os.path.dirname(path_resultados))

    df_precios_optimos.to_csv(path_resultados, index=False)

    # Caso base
    data_resultados_caso_base = caso_base_ruteo(path_zonas, path_tiendas, path_venta_zona,
                                                path_flota, path_camiones, path_productos, dia)
    # Graficar rutas de caso base

    resultados_caso_base = os.path.join(
        base_dir, 'resultados', f'dia_{dia}', 'caso_base', f'resultados_caso_base_dia_{dia}.csv')
    graficar_rutas(data_resultados_caso_base, path_zonas, path_tiendas,
                   dia, caso_base=True)

