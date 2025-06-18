from kpis import creacion_df_40_dias_ruteo, distancia_promedio, utilidad_total, distancia_promedio, utilidad_de_entregas, ventas_perdidas, demanda_online_insatisfecha, creacion_df_40_dias_ruteo, creacion_df_n_dias_demanda_insatisfecha
import os
import pandas as pd
# KPIs

base_dir = os.path.dirname(__file__)
n_dias = 40
df_resultados_kpis = pd.DataFrame()

# ----- RUTEO -----
path_distancias_dias_cw_solo = creacion_df_40_dias_ruteo(
    n_dias, caso_base=False, mejorados=False, cw_solo=True)
distancias_promedio_cw_solo = distancia_promedio(path_distancias_dias_cw_solo)

print(f"Distancia promedio CW solo: {distancias_promedio_cw_solo}")
df_resultados_kpis['distancia_promedio_cw_solo'] = [
    distancias_promedio_cw_solo]

path_distancias_dias_mejorados = creacion_df_40_dias_ruteo(
    n_dias, caso_base=False, mejorados=True, cw_solo=False)

distancias_promedio_mejorados = distancia_promedio(
    path_distancias_dias_mejorados)
print(f"Distancia promedio mejorada: {distancias_promedio_mejorados}")
df_resultados_kpis['distancia_promedio_mejorada'] = [
    distancias_promedio_mejorados]

path_distancias_dias_caso_base = creacion_df_40_dias_ruteo(
    n_dias, caso_base=True, mejorados=False, cw_solo=False)
distancias_promedio_caso_base = distancia_promedio(
    path_distancias_dias_caso_base)
print(f"Distancia promedio caso base: {distancias_promedio_caso_base}")
df_resultados_kpis['distancia_promedio_caso_base'] = [
    distancias_promedio_caso_base]

path_demanda_online_insatisfecha_cw = creacion_df_n_dias_demanda_insatisfecha(
    40, cw=True)
demanda_online_insatisfecha_cw = demanda_online_insatisfecha(
    path_demanda_online_insatisfecha_cw)
print(f"Demanda online insatisfecha CW: {demanda_online_insatisfecha_cw}")
df_resultados_kpis['demanda_online_insatisfecha_cw'] = [
    demanda_online_insatisfecha_cw]

path_demanda_online_insatisfecha_cb = creacion_df_n_dias_demanda_insatisfecha(
    40, caso_base=True)
demanda_online_insatisfecha_cb = demanda_online_insatisfecha(
    path_demanda_online_insatisfecha_cb)
print(f"Demanda online insatisfecha CB: {demanda_online_insatisfecha_cb}")
df_resultados_kpis['demanda_online_insatisfecha_cb'] = [
    demanda_online_insatisfecha_cb]

# ----- PRICING -----
path_tarifas = os.path.join(
    base_dir, 'resultados', 'totales', 'total_pricing.csv')
utilidad_de_entregas_total = utilidad_de_entregas(path_tarifas)
print(f"Utilidad de entregas: {utilidad_de_entregas_total}")
df_resultados_kpis['utilidad_de_entregas'] = [
    utilidad_de_entregas_total]

path_costo_diario_inventario = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'costos_diarios.csv')
path_distancias_ruteo = os.path.join(
    base_dir, 'resultados', 'distancias_totales_mejoradas_2opt_CW_40_dias.csv')

utilidad_total_total = utilidad_total(
    path_costo_diario_inventario, path_distancias_ruteo, path_tarifas)
print(f"Utilidad total: {utilidad_total_total}")
df_resultados_kpis['utilidad_total'] = [utilidad_total_total]

path_tarifas_cb1 = os.path.join(
    base_dir, 'resultados', 'totales', 'total_caso_base_1.csv')
path_tarifas_cb2 = os.path.join(
    base_dir, 'resultados', 'totales', 'total_caso_base_2.csv')
utilidad_de_entregas_cb1 = utilidad_de_entregas(path_tarifas_cb1)
utilidad_de_entregas_cb2 = utilidad_de_entregas(path_tarifas_cb2)

print(f"Utilidad de entregas caso base 1: {utilidad_de_entregas_cb1}")
df_resultados_kpis['utilidad_de_entregas_cb1'] = [
    utilidad_de_entregas_cb1]
print(f"Utilidad de entregas caso base 2: {utilidad_de_entregas_cb2}")
df_resultados_kpis['utilidad_de_entregas_cb2'] = [
    utilidad_de_entregas_cb2]

path_costo_diario_caso_base = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'costos_diarios_caso_base.csv')
path_distancias_ruteo_cb = os.path.join(
    base_dir, 'resultados', 'distancias_totales_caso_base_40_dias.csv')

utilidad_total_cb1 = utilidad_total(
    path_costo_diario_caso_base, path_distancias_ruteo_cb, path_tarifas_cb1)
utilidad_total_cb2 = utilidad_total(
    path_costo_diario_caso_base, path_distancias_ruteo_cb, path_tarifas_cb2)
print(f"Utilidad total caso base 1: {utilidad_total_cb1}")
df_resultados_kpis['utilidad_total_cb1'] = [utilidad_total_cb1]
print(f"Utilidad total caso base 2: {utilidad_total_cb2}")
df_resultados_kpis['utilidad_total_cb2'] = [utilidad_total_cb2]

# ----- INVENTARIO -----
ventas_perdidas_caso_base = ventas_perdidas(base=True)
print(
    f"Número de ventas perdidas por falta de inventario caso base: {ventas_perdidas_caso_base[0]}")
df_resultados_kpis['ventas_perdidas_caso_base'] = [
    ventas_perdidas_caso_base[0]]
print(
    f"Porcentaje de ventas perdidas por falta de inventario caso base: {ventas_perdidas_caso_base[1]*100:.2f}%")
df_resultados_kpis['porcentaje_ventas_perdidas_caso_base'] = [
    ventas_perdidas_caso_base[1]]

ventas_perdidas_no_caso_base = ventas_perdidas()
print(
    f"Número de ventas perdidas por falta de inventario: {ventas_perdidas_no_caso_base[0]}")
df_resultados_kpis['ventas_perdidas_no_caso_base'] = [
    ventas_perdidas_no_caso_base[0]]
print(
    f"Porcentaje de ventas perdidas por falta de inventario: {ventas_perdidas_no_caso_base[1]*100:.2f}%")
df_resultados_kpis['porcentaje_ventas_perdidas_no_caso_base'] = [
    ventas_perdidas_no_caso_base[1]]

# Guardar resultados en un CSV
output_path = os.path.join(
    base_dir, 'resultados', 'kpis', 'resultados_kpis.csv')
if not os.path.exists(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
df_resultados_kpis.to_csv(output_path, index=False)
print(f"Resultados de KPIs guardados en: {output_path}")
