from kpis_sensibilidad_ruteo import creacion_df_40_dias_ruteo, distancia_promedio, demanda_online_insatisfecha, creacion_df_n_dias_demanda_insatisfecha
import os
import pandas as pd
# KPIs

base_dir = os.path.dirname(__file__)
n_dias = 5
df_resultados_kpis = pd.DataFrame()

path_ids_sensibilidad = os.path.join(
    base_dir, 'resultados_sensibilidad', 'ids_sensibilidad_ruteo.csv')
ids_sensibilidad = pd.read_csv(path_ids_sensibilidad)
if isinstance(ids_sensibilidad, pd.DataFrame):
    ids_sensibilidad = ids_sensibilidad.squeeze().tolist()

# ----- RUTEO -----
for id_sensibilidad in ids_sensibilidad:
    path_distancias_dias_cw_solo = creacion_df_40_dias_ruteo(
        n_dias, caso_base=False, mejorados=False, cw_solo=True, id_sensibilidad=id_sensibilidad)
    distancias_promedio_cw_solo = distancia_promedio(
        path_distancias_dias_cw_solo)

    print(f"Distancia promedio CW solo: {distancias_promedio_cw_solo}")
    df_resultados_kpis['distancia_promedio_cw_solo'] = [
        distancias_promedio_cw_solo]

    path_distancias_dias_mejorados = creacion_df_40_dias_ruteo(
        n_dias, caso_base=False, mejorados=True, cw_solo=False, id_sensibilidad=id_sensibilidad)

    distancias_promedio_mejorados = distancia_promedio(
        path_distancias_dias_mejorados)
    print(f"Distancia promedio mejorada: {distancias_promedio_mejorados}")
    df_resultados_kpis['distancia_promedio_mejorada'] = [
        distancias_promedio_mejorados]

    path_demanda_online_insatisfecha_cw = creacion_df_n_dias_demanda_insatisfecha(
        n_dias, cw=True, id_sensibilidad=id_sensibilidad)
    demanda_online_insatisfecha_cw = demanda_online_insatisfecha(
        path_demanda_online_insatisfecha_cw)
    print(f"Demanda online insatisfecha CW: {demanda_online_insatisfecha_cw}")
    df_resultados_kpis['demanda_online_insatisfecha_cw'] = [
        demanda_online_insatisfecha_cw]

    # Guardar resultados en un CSV
    output_path = os.path.join(
        base_dir, 'resultados_sensibilidad', 'kpis', f'resultados_kpis_{id_sensibilidad}.csv')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df_resultados_kpis.to_csv(output_path, index=False)
    print(f"Resultados de KPIs guardados en: {output_path}")
