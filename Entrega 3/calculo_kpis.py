from kpis import creacion_df_40_dias_ruteo, distancia_promedio, utilidad_total, utilidad_de_entregas, ventas_perdidas, demanda_online_insatisfecha, creacion_df_40_dias_ruteo, creacion_df_n_dias_demanda_insatisfecha
import os
from pricing import agrupa_archivos
import pandas as pd
# KPIs

base_dir = os.path.dirname(__file__)
n_dias = 40
df_resultados_kpis = pd.DataFrame()

for i in range(0, 6):

    # ----- RUTEO -----
    path_distancias_dias_cw_solo = creacion_df_40_dias_ruteo(
        n_dias, caso_base=False, mejorados=False, cw_solo=True, id_realizacion=i)
    distancias_promedio_cw_solo = distancia_promedio(
        path_distancias_dias_cw_solo)

    print(f"Distancia promedio CW solo: {distancias_promedio_cw_solo}")
    df_resultados_kpis['distancia_promedio_cw_solo'] = [
        distancias_promedio_cw_solo]

    path_distancias_dias_mejorados = creacion_df_40_dias_ruteo(
        n_dias, caso_base=False, mejorados=True, cw_solo=False, id_realizacion=i)

    distancias_promedio_mejorados = distancia_promedio(
        path_distancias_dias_mejorados)
    print(f"Distancia promedio mejorada: {distancias_promedio_mejorados}")
    df_resultados_kpis['distancia_promedio_mejorada'] = [
        distancias_promedio_mejorados]

    path_distancias_dias_caso_base = creacion_df_40_dias_ruteo(
        n_dias, caso_base=True, mejorados=False, cw_solo=False, id_realizacion=i)
    distancias_promedio_caso_base = distancia_promedio(
        path_distancias_dias_caso_base)
    print(f"Distancia promedio caso base: {distancias_promedio_caso_base}")
    df_resultados_kpis['distancia_promedio_caso_base'] = [
        distancias_promedio_caso_base]

    path_demanda_online_insatisfecha_cw = creacion_df_n_dias_demanda_insatisfecha(
        40, cw=True, id_realizacion=i)
    demanda_online_insatisfecha_cw = demanda_online_insatisfecha(
        path_demanda_online_insatisfecha_cw)
    print(f"Demanda online insatisfecha CW: {demanda_online_insatisfecha_cw}")
    df_resultados_kpis['demanda_online_insatisfecha_cw'] = [
        demanda_online_insatisfecha_cw]

    path_demanda_online_insatisfecha_cb = creacion_df_n_dias_demanda_insatisfecha(
        40, caso_base=True, id_realizacion=i)
    demanda_online_insatisfecha_cb = demanda_online_insatisfecha(
        path_demanda_online_insatisfecha_cb)
    print(f"Demanda online insatisfecha CB: {demanda_online_insatisfecha_cb}")
    df_resultados_kpis['demanda_online_insatisfecha_cb'] = [
        demanda_online_insatisfecha_cb]

    # ----- PRICING -----
    rutas = []
    for dia in range(1, 41):
        print(f"Procesando ruta día {dia}")
        path_distancia_total_dia = os.path.join(base_dir, f'realizacion_{i}',
                                                'resultados', f'dia_{dia}', f'resultados_pricing_dia_{dia}.csv')
        rutas.append(path_distancia_total_dia)
    df_agrupado = agrupa_archivos(
        rutas, f'total_pricing_r{i}', id_realizacion=i)

    path_tarifas = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'totales', f'total_pricing_r{i}.csv')
    utilidad_de_entregas_total = utilidad_de_entregas(path_tarifas)
    print(f"Utilidad de entregas: {utilidad_de_entregas_total}")
    df_resultados_kpis['utilidad_de_entregas'] = [
        utilidad_de_entregas_total]

    # path_costo_diario_inventario = os.path.join(
    #     base_dir, '..', 'Datos', 'analisis de datos', 'costos_diarios.csv')

    path_base_costos_diarios = os.path.join(
        base_dir, '..', 'Datos', 'analisis de datos', 'inventario_escenarios')
    if i == 0:
        path_costos_diarios = os.path.join(
            path_base_costos_diarios, 'semilla_42', 'costos_diarios.csv')
    if i == 1:
        path_costos_diarios = os.path.join(
            path_base_costos_diarios, 'semilla_101', 'costos_diarios.csv')
    if i == 2:
        path_costos_diarios = os.path.join(
            path_base_costos_diarios, 'semilla_202', 'costos_diarios.csv')
    if i == 3:
        path_costos_diarios = os.path.join(
            path_base_costos_diarios, 'semilla_303', 'costos_diarios.csv')
    if i == 4:
        path_costos_diarios = os.path.join(
            path_base_costos_diarios, 'semilla_404', 'costos_diarios.csv')
    if i == 5:
        path_costos_diarios = os.path.join(
            path_base_costos_diarios, 'semilla_505', 'costos_diarios.csv')

    path_distancias_ruteo = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'distancias_totales_mejoradas_2opt_CW_40_dias.csv')

    utilidad_total_total = utilidad_total(
        path_costos_diarios, path_distancias_ruteo, path_tarifas)
    print(f"Utilidad total: {utilidad_total_total}")
    df_resultados_kpis['utilidad_total'] = [utilidad_total_total]

    rutas_cb1 = []
    rutas_cb2 = []
    for dia in range(1, 41):
        path_cb1_pricing = os.path.join(
            base_dir, f'realizacion_{i}', 'resultados', f'dia_{dia}', 'caso_base_1_pricing', f'resultados_caso_base_dia_{dia}.csv')

        path_cb2_pricing = os.path.join(
            base_dir, f'realizacion_{i}', 'resultados', f'dia_{dia}', 'caso_base_2_pricing', f'resultados_caso_base_dia_{dia}.csv')

        rutas_cb1.append(path_cb1_pricing)
        rutas_cb2.append(path_cb2_pricing)
    df_agrupado_cb1 = agrupa_archivos(
        rutas_cb1, f'total_cb1_pricing_r{i}', id_realizacion=i)
    df_agrupado_cb2 = agrupa_archivos(
        rutas_cb2, f'total_cb2_pricing_r{i}', id_realizacion=i)

    path_tarifas_cb1 = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'totales', f'total_cb1_pricing_r{i}.csv')
    path_tarifas_cb2 = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'totales', f'total_cb2_pricing_r{i}.csv')
    utilidad_de_entregas_cb1 = utilidad_de_entregas(path_tarifas_cb1)
    utilidad_de_entregas_cb2 = utilidad_de_entregas(path_tarifas_cb2)

    print(f"Utilidad de entregas caso base 1: {utilidad_de_entregas_cb1}")
    df_resultados_kpis['utilidad_de_entregas_cb1'] = [
        utilidad_de_entregas_cb1]
    print(f"Utilidad de entregas caso base 2: {utilidad_de_entregas_cb2}")
    df_resultados_kpis['utilidad_de_entregas_cb2'] = [
        utilidad_de_entregas_cb2]

    # path_costo_diario_caso_base = os.path.join(
    #     base_dir, '..', 'Datos', 'analisis de datos', 'costos_diarios_caso_base.csv')

    path_base_caso_base_costos_diarios = os.path.join(
        base_dir, '..', 'Datos', 'analisis de datos', 'cbase_inventario_escenarios')
    if i == 0:
        path_costos_diarios_cb = os.path.join(
            path_base_caso_base_costos_diarios, 'semilla_42', 'costos_diarios.csv')
    if i == 1:
        path_costos_diarios_cb = os.path.join(
            path_base_caso_base_costos_diarios, 'semilla_101', 'costos_diarios.csv')
    if i == 2:
        path_costos_diarios_cb = os.path.join(
            path_base_caso_base_costos_diarios, 'semilla_202', 'costos_diarios.csv')
    if i == 3:
        path_costos_diarios_cb = os.path.join(
            path_base_caso_base_costos_diarios, 'semilla_303', 'costos_diarios.csv')
    if i == 4:
        path_costos_diarios_cb = os.path.join(
            path_base_caso_base_costos_diarios, 'semilla_404', 'costos_diarios.csv')
    if i == 5:
        path_costos_diarios_cb = os.path.join(
            path_base_caso_base_costos_diarios, 'semilla_505', 'costos_diarios.csv')

    path_distancias_ruteo = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'distancias_totales_mejoradas_2opt_CW_40_dias.csv')

    path_distancias_ruteo_cb = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'distancias_totales_caso_base_40_dias.csv')

    utilidad_total_cb1 = utilidad_total(
        path_costos_diarios_cb, path_distancias_ruteo_cb, path_tarifas_cb1)
    utilidad_total_cb2 = utilidad_total(
        path_costos_diarios_cb, path_distancias_ruteo_cb, path_tarifas_cb2)
    print(f"Utilidad total caso base 1: {utilidad_total_cb1}")
    df_resultados_kpis['utilidad_total_cb1'] = [utilidad_total_cb1]
    print(f"Utilidad total caso base 2: {utilidad_total_cb2}")
    df_resultados_kpis['utilidad_total_cb2'] = [utilidad_total_cb2]

    # ----- INVENTARIO ----- ***
    # ventas_perdidas_caso_base = ventas_perdidas(base=True)
    # print(
    #     f"Número de ventas perdidas por falta de inventario caso base: {ventas_perdidas_caso_base[0]}")
    # df_resultados_kpis['ventas_perdidas_caso_base'] = [
    #     ventas_perdidas_caso_base[0]]
    # print(
    #     f"Porcentaje de ventas perdidas por falta de inventario caso base: {ventas_perdidas_caso_base[1]*100:.2f}%")
    # df_resultados_kpis['porcentaje_ventas_perdidas_caso_base'] = [
    #     ventas_perdidas_caso_base[1]]

    # ventas_perdidas_no_caso_base = ventas_perdidas()
    # print(
    #     f"Número de ventas perdidas por falta de inventario: {ventas_perdidas_no_caso_base[0]}")
    # df_resultados_kpis['ventas_perdidas_no_caso_base'] = [
    #     ventas_perdidas_no_caso_base[0]]
    # print(
    #     f"Porcentaje de ventas perdidas por falta de inventario: {ventas_perdidas_no_caso_base[1]*100:.2f}%")
    # df_resultados_kpis['porcentaje_ventas_perdidas_no_caso_base'] = [
    #     ventas_perdidas_no_caso_base[1]]

    # Guardar resultados en un CSV
    output_path = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'kpis', 'resultados_kpis.csv')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df_resultados_kpis.to_csv(output_path, index=False)
    print(f"Resultados de KPIs guardados en: {output_path}")
