import pandas as pd
import os


def utilidad_total(path_costos_diarios_inventario, path_distancias_ruteo, path_tarifas):
    data_costos_diarios_inventario = pd.read_csv(
        path_costos_diarios_inventario)
    data_distancias_ruteo = pd.read_csv(path_distancias_ruteo)
    data_tarifas = pd.read_csv(path_tarifas)
    promedio_costos_inventario = data_costos_diarios_inventario['costo_inventario'].mean(
    )
    promedio_distancia_ruteo = data_distancias_ruteo['distancia_total'].mean()
    promedio_utilidad_entregas = data_tarifas['U_k (entero óptimo)'].mean()
    return promedio_utilidad_entregas - (promedio_costos_inventario + promedio_distancia_ruteo)


def distancia_promedio(path_distancias_ruteo):
    data_distancias_ruteo = pd.read_csv(path_distancias_ruteo)
    return data_distancias_ruteo['distancia_total'].mean()


def utilidad_de_entregas(path_tarifas):
    data_tarifas = pd.read_csv(path_tarifas)
    return data_tarifas['U_k (entero óptimo)'].mean()


def ventas_perdidas(base=False):
    """
    Retorna las ventas perdidas absolutas y porcentuales.
    Por defecto retorna el caso óptimo. Si base=True, retorna el caso base.
    """
    base_dir = os.path.dirname(__file__)
    ruta_csv = os.path.join(base_dir,
                            '..', 'Datos', 'analisis de datos', 'kpi_ventas_perdidas_inventario.csv')
    df = pd.read_csv(ruta_csv)
    caso = 'base' if base else 'optima'
    row = df[df['caso'] == caso]
    if row.empty:
        raise ValueError(f"No hay resultados para el caso '{caso}'")
    ventas_perdidas = float(row['ventas_perdidas'].values[0])
    nivel_servicio = float(row['nivel_servicio'].values[0])
    ventas_perdidas_pct = 1 - nivel_servicio
    return ventas_perdidas, ventas_perdidas_pct


def demanda_online_insatisfecha(path_demandas_online_insatisfechas):
    data_demandas_online = pd.read_csv(path_demandas_online_insatisfechas)
    return data_demandas_online['demanda_total_insatisfecha'].mean()


def creacion_df_n_dias_demanda_insatisfecha(n_dias, caso_base=False, cw=False, id_realizacion=None):
    df_output = pd.DataFrame(columns=['dia', 'demanda_total_insatisfecha'])
    base_dir = os.path.dirname(__file__)
    for i in range(1, n_dias + 1):
        if caso_base:
            path_demanda_total_dia = os.path.join(base_dir,
                                                  'resultados', f'dia_{i}', 'caso_base_ruteo', f'demanda_insatisfecha_caso_base_dia_{i}.csv')
        elif cw:
            path_demanda_total_dia = os.path.join(base_dir,
                                                  'resultados', f'dia_{i}', f'demanda_insatisfecha_CW_dia_{i}.csv')
        else:
            raise ValueError(
                "Debe especificar un caso: caso_base, cw_solo o mejorados.")
        df = pd.read_csv(path_demanda_total_dia)
        demanda_total = df['demanda'].sum()
        df_output = pd.concat([df_output, pd.DataFrame({'dia': [i], 'demanda_total_insatisfecha': [demanda_total]})],
                              ignore_index=True)
    # guardar el DataFrame en un archivo CSV
    if caso_base:
        output_path = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                   'resultados', f'demanda_total_insatisfecha_caso_base_{n_dias}_dias.csv')
    elif cw:
        output_path = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                   'resultados', f'demanda_total_insatisfecha_CW_{n_dias}_dias.csv')
    df_output.to_csv(output_path, index=False)
    return output_path


def creacion_df_40_dias_ruteo(n_dias, caso_base=False, cw_solo=False, mejorados=False, id_realizacion=None):
    df_output = pd.DataFrame(columns=['dia', 'distancia_total'])
    base_dir = os.path.dirname(__file__)
    for i in range(1, n_dias + 1):
        if caso_base:
            path_distancia_total_dia = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                                    'resultados', f'dia_{i}', 'caso_base_ruteo', f'distancia_total_caso_base_dia_{i}.csv')
        elif cw_solo:
            path_distancia_total_dia = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                                    'resultados', f'dia_{i}', f'distancia_total_CW_dia_{i}.csv')
        elif mejorados:
            path_distancia_total_dia = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                                    'resultados', f'dia_{i}', f'distancia_total_mejorada_2opt_CW_dia_{i}.csv')
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
        output_path = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                   'resultados', f'distancias_totales_caso_base_{n_dias}_dias.csv')
    elif cw_solo:
        output_path = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                   'resultados', f'distancias_totales_CW_{n_dias}_dias.csv')
    elif mejorados:
        output_path = os.path.join(base_dir, f'realizacion_{id_realizacion}' if id_realizacion is not None else '',
                                   'resultados', f'distancias_totales_mejoradas_2opt_CW_{n_dias}_dias.csv')
    df_output.to_csv(output_path, index=False)
    return output_path
