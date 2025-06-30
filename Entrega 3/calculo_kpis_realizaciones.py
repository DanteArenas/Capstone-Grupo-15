import os
import pandas as pd
import numpy as np
from scipy import stats

distancias_promedio_mejoradas = []
distancias_promedio_cb = []
demanda_online_insatisfecha_cw = []
demanda_online_insatisfecha_cb = []
utilidad_de_entregas = []
utilidad_total = []
utilidad_de_entregas_cb1 = []
utilidad_de_entregas_cb2 = []
utilidad_total_cb1 = []
utilidad_total_cb2 = []
base_dir = os.path.dirname(__file__)
# distancia_promedio_cw_solo,distancia_promedio_mejorada,distancia_promedio_caso_base,demanda_online_insatisfecha_cw,
# demanda_online_insatisfecha_cb,utilidad_de_entregas,utilidad_total,utilidad_de_entregas_cb1,utilidad_de_entregas_cb2,
# utilidad_total_cb1,utilidad_total_cb2
for i in range(0, 6):
    path_kpis_realizacion = os.path.join(
        base_dir, f'realizacion_{i}', 'resultados', 'kpis', 'resultados_kpis.csv')
    data_kpis = pd.read_csv(path_kpis_realizacion)
    print(data_kpis)
    distancias_promedio_mejoradas.append(
        data_kpis['distancia_promedio_mejorada'].iloc[0])
    distancias_promedio_cb.append(
        data_kpis['distancia_promedio_caso_base']. iloc[0])
    demanda_online_insatisfecha_cw.append(
        data_kpis['demanda_online_insatisfecha_cw'].iloc[0])
    demanda_online_insatisfecha_cb.append(
        data_kpis['demanda_online_insatisfecha_cb'].iloc[0])
    utilidad_de_entregas.append(data_kpis['utilidad_de_entregas'].iloc[0])
    utilidad_total.append(data_kpis['utilidad_total'].iloc[0])
    utilidad_de_entregas_cb1.append(
        data_kpis['utilidad_de_entregas_cb1'].iloc[0])
    utilidad_de_entregas_cb2.append(
        data_kpis['utilidad_de_entregas_cb2'].iloc[0])
    utilidad_total_cb1.append(data_kpis['utilidad_total_cb1'].iloc[0])
    utilidad_total_cb2.append(data_kpis['utilidad_total_cb2'].iloc[0])


def calcular_promedio_intervalo(lista):
    # Copilot
    arr = np.array(lista, dtype=float)
    promedio = np.mean(arr)
    sem = stats.sem(arr)
    intervalo = stats.t.interval(0.95, len(
        arr)-1, loc=promedio, scale=sem) if len(arr) > 1 else (promedio, promedio)
    return promedio, intervalo


listas_kpis = [
    ('distancias_promedio_mejoradas', distancias_promedio_mejoradas),
    ('distancias_promedio_cb', distancias_promedio_cb),
    ('demanda_online_insatisfecha_cw', demanda_online_insatisfecha_cw),
    ('demanda_online_insatisfecha_cb', demanda_online_insatisfecha_cb),
    ('utilidad_de_entregas', utilidad_de_entregas),
    ('utilidad_total', utilidad_total),
    ('utilidad_de_entregas_cb1', utilidad_de_entregas_cb1),
    ('utilidad_de_entregas_cb2', utilidad_de_entregas_cb2),
    ('utilidad_total_cb1', utilidad_total_cb1),
    ('utilidad_total_cb2', utilidad_total_cb2),
]

# Calcular y mostrar promedios e intervalos de confianza
resultados = []
for nombre, lista in listas_kpis:
    promedio, intervalo = calcular_promedio_intervalo(lista)
    resultados.append({
        'kpi': nombre,
        'promedio': promedio,
        'ic_95_inf': intervalo[0],
        'ic_95_sup': intervalo[1]
    })
    print(
        f"{nombre}: Promedio = {promedio:.2f}, IC 95% = ({intervalo[0]:.2f}, {intervalo[1]:.2f})")

# Guardar resultados en un CSV
df_resultados = pd.DataFrame(resultados)
output_csv = os.path.join(base_dir, 'resultados_kpis_resumen.csv')
df_resultados.to_csv(output_csv, index=False)
print(f"Resumen de KPIs guardado en: {output_csv}")
