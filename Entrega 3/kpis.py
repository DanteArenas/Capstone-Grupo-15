def utilidad_total(path_costos_diarios_inventario, path_distancias_ruteo, path_tarifas):
    pass


def distancia_promedio(path_distancias_ruteo):
    data_distancias_ruteo = pd.read_csv(path_distancias_ruteo)
    return data_distancias_ruteo['distancia_total_recorrida_camion'].mean()


def utilidad_de_entregas(path_tarifas):
    data_tarifas = pd.read_csv(path_tarifas)
    return data_tarifas['U_k (entero óptimo)'].mean()


def ventas_perdidas():
    pass


def demanda_online_insatisfecha(path_demandas_online_insatisfechas):
    data_demandas_online = pd.read_csv(path_demandas_online_insatisfechas)
    return data_demandas_online['demanda_online_insatisfecha'].mean()
