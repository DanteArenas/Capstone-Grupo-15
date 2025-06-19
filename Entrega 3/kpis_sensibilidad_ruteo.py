import pandas as pd
import os


def distancia_promedio(path_distancias_ruteo):
    data_distancias_ruteo = pd.read_csv(path_distancias_ruteo)
    return data_distancias_ruteo['distancia_total'].mean()


def demanda_online_insatisfecha(path_demandas_online_insatisfechas):
    data_demandas_online = pd.read_csv(path_demandas_online_insatisfechas)
    return data_demandas_online['demanda_total_insatisfecha'].mean()


def creacion_df_n_dias_demanda_insatisfecha(n_dias, caso_base=False, cw=False, id_sensibilidad=None):
    df_output = pd.DataFrame(columns=['dia', 'demanda_total_insatisfecha'])
    base_dir = os.path.dirname(__file__)
    for i in range(1, n_dias + 1):
        if caso_base:
            path_demanda_total_dia = os.path.join(base_dir,
                                                  'resultados_sensibilidad', f'{id_sensibilidad}', f'dia_{i}', 'caso_base_ruteo', f'demanda_insatisfecha_caso_base_dia_{i}.csv')
        elif cw:
            path_demanda_total_dia = os.path.join(base_dir,
                                                  'resultados_sensibilidad', f'{id_sensibilidad}', f'dia_{i}', f'demanda_insatisfecha_CW_dia_{i}.csv')
        else:
            raise ValueError(
                "Debe especificar un caso: caso_base, cw_solo o mejorados.")
        df = pd.read_csv(path_demanda_total_dia)
        demanda_total = df['demanda'].sum()
        df_output = pd.concat([df_output, pd.DataFrame({'dia': [i], 'demanda_total_insatisfecha': [demanda_total]})],
                              ignore_index=True)
    # guardar el DataFrame en un archivo CSV
    if caso_base:
        output_path = os.path.join(base_dir,
                                   'resultados_sensibilidad', f'{id_sensibilidad}', f'demanda_total_insatisfecha_caso_base_{n_dias}_dias.csv')
    elif cw:
        output_path = os.path.join(base_dir,
                                   'resultados_sensibilidad', f'{id_sensibilidad}', f'demanda_total_insatisfecha_CW_{n_dias}_dias.csv')
    df_output.to_csv(output_path, index=False)
    return output_path


def creacion_df_40_dias_ruteo(n_dias, caso_base=False, cw_solo=False, mejorados=False, id_sensibilidad=None):
    df_output = pd.DataFrame(columns=['dia', 'distancia_total'])
    base_dir = os.path.dirname(__file__)
    for i in range(1, n_dias + 1):
        if caso_base:
            path_distancia_total_dia = os.path.join(base_dir,
                                                    'resultados_sensibilidad', f'{id_sensibilidad}', f'dia_{i}', 'caso_base_ruteo', f'distancia_total_caso_base_dia_{i}.csv')
        elif cw_solo:
            path_distancia_total_dia = os.path.join(base_dir,
                                                    'resultados_sensibilidad', f'{id_sensibilidad}', f'dia_{i}', f'distancia_total_CW_dia_{i}.csv')
        elif mejorados:
            path_distancia_total_dia = os.path.join(base_dir,
                                                    'resultados_sensibilidad', f'{id_sensibilidad}', f'dia_{i}', f'distancia_total_mejorada_2opt_CW_dia_{i}.csv')
        else:
            raise ValueError(
                "Debe especificar un caso: caso_base, cw_solo o mejorados.")
        df = pd.read_csv(path_distancia_total_dia)
        distancia_total = df['distancia_total']
        distancia_total = distancia_total.iloc[0]
        df_output = pd.concat([df_output, pd.DataFrame({'dia': [i], 'distancia_total': [distancia_total]})],
                              ignore_index=True)
    # guardar el DataFrame en un archivo CSV
    if caso_base:
        output_path = os.path.join(base_dir,
                                   'resultados_sensibilidad', f'{id_sensibilidad}', f'distancias_totales_caso_base_{n_dias}_dias.csv')
    elif cw_solo:
        output_path = os.path.join(base_dir,
                                   'resultados_sensibilidad', f'{id_sensibilidad}', f'distancias_totales_CW_{n_dias}_dias.csv')
    elif mejorados:
        output_path = os.path.join(base_dir,
                                   'resultados_sensibilidad', f'{id_sensibilidad}', f'distancias_totales_mejoradas_2opt_CW_{n_dias}_dias.csv')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df_output.to_csv(output_path, index=False)
    return output_path
