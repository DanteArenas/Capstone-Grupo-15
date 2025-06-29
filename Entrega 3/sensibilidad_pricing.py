from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo_zona, zona_vehiculo, agrupa_archivos
from ruteo import generar_rutas, graficar_rutas, mejorar_rutas_2_opt, caso_base_ruteo
from kpis import utilidad_de_entregas, utilidad_total
import os
import pandas as pd

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

for dia in range(1, n_dias + 1):
    path_venta_zona = os.path.join(
        base_dir, 'ventas_digitales_estocasticas', f'venta_zona_estocastica_dia_{dia}.csv')

    # Generar rutas
    data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                    path_flota, path_camiones, path_productos, dia)

    # Mejorar rutas con 2-opt
    data_resultados_mejorada = mejorar_rutas_2_opt(
        data_resultados, path_zonas, path_tiendas, dia)

    # Graficar rutas mejoradas
    graficar_rutas(data_resultados_mejorada, path_zonas, path_tiendas,
                   dia, mejora_2_opt=True)

    # Cargar datos de distancia
    df_distancias_ordenada, kmeans = procesar_datos_de_distancia(
        data_resultados_mejorada, n_clusters=3)

    # Generar matriz ck
    matriz_ck = generar_matriz_ck(
        df_distancias_ordenada, data_resultados_mejorada)

    # Obtener df_zona_vehiculo
    df_zona_vehiculo = zona_vehiculo(data_resultados_mejorada, df_zonas)

    # Resolver precio óptimo
    path_demanda_digital = os.path.join(
        base_dir, 'ventas_digitales_estocasticas', f'venta_zona_estocastica_dia_{dia}.csv')
    df_demanda_digital = pd.read_csv(path_demanda_digital)

    df_precios_optimos = resolver_precio_optimo_zona(
        matriz_ck, df_zona_vehiculo, df_stock_diario, df_demanda_digital, df_zonas, dia, P_LB=20, P_UB=4+0)

    # Guardar resultados
    path_resultados = os.path.join(
        base_dir, 'resultados', 'sensibilidad_pricing_LB20_UB40', f'dia_{dia}', f'resultados_pricing_dia_{dia}.csv')

    if not os.path.exists(os.path.dirname(path_resultados)):
        os.makedirs(os.path.dirname(path_resultados))

    df_precios_optimos.to_csv(path_resultados, index=False)

base_dir = os.path.dirname(__file__)
rutas = []
for dia in range(1, n_dias + 1):
    print(f"Procesando día {dia}")
    base_dir = os.path.dirname(__file__)
    path_distancia_total_dia = os.path.join(base_dir,
                                            'resultados', 'sensibilidad_pricing_LB20_UB40', f'dia_{dia}', f'resultados_pricing_dia_{dia}.csv')
    rutas.append(path_distancia_total_dia)
df_agrupado = agrupa_archivos(rutas, 'sensibilidad_pricing_LB20_UB40')

path_tarifas = os.path.join(
    base_dir, 'resultados', 'totales', 'sensibilidad_pricing_LB20_UB40')
if not os.path.exists(os.path.dirname(path_tarifas)):
    os.makedirs(os.path.dirname(path_tarifas))
df_agrupado.to_csv(path_tarifas, index=False)
utilidad_de_entregas_total = utilidad_de_entregas(path_tarifas)
path_costo_diario_inventario = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'costos_diarios.csv')
path_distancias_ruteo = os.path.join(
    base_dir, 'resultados', 'distancias_totales_mejoradas_2opt_CW_40_dias.csv')
utilidad_total_total = utilidad_total(
    path_costo_diario_inventario, path_distancias_ruteo, path_tarifas)
print(f"Utilidad de entregas con 3 clusters: {utilidad_de_entregas_total}")
print(f"Utilidad total con 3 clusters: {utilidad_total_total}")


