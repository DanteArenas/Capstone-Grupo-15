from pricing import agrupa_archivos
import os
import pandas as pd

rutas = []
for dia in range(1, 41):
    print(f"Procesando día {dia}")
    base_dir = os.path.dirname(__file__)
    path_distancia_total_dia = os.path.join(base_dir,
                                            'resultados', f'dia_{dia}', f'resultados_pricing_dia_{dia}.csv')
    rutas.append(path_distancia_total_dia)
df_agrupado = agrupa_archivos(rutas, 'total_pricing')


rutas_cb1 = []
for dia in range(1, 41):
    print(f"Procesando día {dia} caso base 1")
    base_dir = os.path.dirname(__file__)
    path_distancia_total_dia = os.path.join(base_dir,
                                            'resultados', f'dia_{dia}', 'caso_base_1_pricing', f'resultados_caso_base_dia_{dia}.csv')
    rutas_cb1.append(path_distancia_total_dia)
df_agrupado = agrupa_archivos(rutas_cb1, 'total_caso_base_1')

rutas_cb2 = []
for dia in range(1, 41):
    print(f"Procesando día {dia} caso base 2")
    base_dir = os.path.dirname(__file__)
    path_distancia_total_dia = os.path.join(base_dir,
                                            'resultados', f'dia_{dia}', 'caso_base_2_pricing', f'resultados_caso_base_dia_{dia}.csv')
    rutas_cb2.append(path_distancia_total_dia)
df_agrupado = agrupa_archivos(rutas_cb2, 'total_caso_base_2')
