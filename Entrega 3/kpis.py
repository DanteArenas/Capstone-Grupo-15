import pandas as pd
import os


def utilidad_total(path_costos_diarios_inventario, path_distancias_ruteo, path_tarifas):
    pass


def distancia_promedio(path_distancias_ruteo):
    data_distancias_ruteo = pd.read_csv(path_distancias_ruteo)
    return data_distancias_ruteo['distancia_total'].mean()


def utilidad_de_entregas(path_tarifas):
    data_tarifas = pd.read_csv(path_tarifas)
    return data_tarifas['U_k (entero óptimo)'].mean()


def ventas_perdidas():
    """
    Retorna las ventas perdidas absolutas y porcentuales del caso óptimo.
    """
    ruta_csv = os.path.join('..', 'Datos', 'analisis de datos', 'kpi_ventas_perdidas_inventario.csv')
    df = pd.read_csv(ruta_csv)
    row = df[df['caso'] == 'optima']
    if row.empty:
        raise ValueError("No hay resultados para el caso 'optima'")
    ventas_perdidas = float(row['ventas_perdidas'].values[0])
    demanda_total = float(row['demanda_total'].values[0])
    ventas_perdidas_pct = ventas_perdidas / demanda_total if demanda_total > 0 else 0.0
    return ventas_perdidas, ventas_perdidas_pct

    # Para Caso Base:
    # row_base = df[df['caso'] == 'base']
    # if not row_base.empty:
    #     ventas_perdidas_base = float(row_base['ventas_perdidas'].values[0])
    #     demanda_total_base = float(row_base['demanda_total'].values[0])
    #     ventas_perdidas_pct_base = ventas_perdidas_base / demanda_total_base if demanda_total_base > 0 else 0.0
    #     return {
    #         'optima': (ventas_perdidas, ventas_perdidas_pct),
    #         'base': (ventas_perdidas_base, ventas_perdidas_pct_base)
    #     }


def demanda_online_insatisfecha(path_demandas_online_insatisfechas):
    data_demandas_online = pd.read_csv(path_demandas_online_insatisfechas)
    return data_demandas_online['demanda_online_insatisfecha'].mean()


def creacion_df_40_dias_ruteo(n_dias, caso_base=False, cw_solo=False, mejorados=False):
    df_output = pd.DataFrame(columns=['dia', 'distancia_total'])
    base_dir = os.path.dirname(__file__)
    for i in range(1, n_dias + 1):
        if caso_base:
            path_distancia_total_dia = os.path.join(base_dir,
                                                    'resultados', f'dia_{i}', 'caso_base_ruteo', f'distancia_total_caso_base_dia_{i}.csv')
        elif cw_solo:
            path_distancia_total_dia = os.path.join(base_dir,
                                                    'resultados', f'dia_{i}', f'distancia_total_CW_dia_{i}.csv')
        elif mejorados:
            path_distancia_total_dia = os.path.join(base_dir,
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
        output_path = os.path.join(base_dir,
                                   'resultados', f'distancias_totales_caso_base_{n_dias}_dias.csv')
    elif cw_solo:
        output_path = os.path.join(base_dir,
                                   'resultados', f'distancias_totales_CW_{n_dias}_dias.csv')
    elif mejorados:
        output_path = os.path.join(base_dir,
                                   'resultados', f'distancias_totales_mejoradas_2opt_CW_{n_dias}_dias.csv')
    df_output.to_csv(output_path, index=False)
    return output_path
