# en este archivo se crearan las simulaciones de pricing y sumatoria de ingresos
# se necesita:
# - función que defina la tarifa según las rutas diarias
# - función que calcule el ingreso diario
# - función que calcule el ingreso total
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpContinuous

def procesar_datos_de_distancia(df_resultados, n_clusters=3):
    """
    Recibe:
    - df_resultados: DataFrame con columnas ['tienda', 'rutas', 'carga', 'distancia', 'carga_total', 'n_camiones_utilizados', 'n_camiones_disponibles']
    
    Retorna:
    - df_ordenado: DataFrame con columnas ['tienda', 'vehiculo', 'distancia', 'cluster_ordenado']
    - kmeans: modelo KMeans entrenado
    """
    df = df_resultados.copy()

    # Inicializar listas para las distancias, tiendas y vehículos
    distancias = []
    tiendas = []
    vehiculos = []

    # Recorremos cada fila y extraemos los elementos de la lista de distancias
    for i, row in df.iterrows():
        tienda = row['tienda']
        dist_list = row['distancia']  # Ya es una lista de distancias
        for j, dist in enumerate(dist_list):
            distancias.append(float(dist))  # Convertimos a float si es necesario
            tiendas.append(tienda)
            vehiculos.append(j)  # El índice de la ruta como identificador del vehículo

    # Creamos el DataFrame con distancias, tiendas y vehículos
    df_distancias = pd.DataFrame({
        'tienda': tiendas,
        'vehiculo': vehiculos,
        'distancia': distancias
    })

    # Ordenamos las distancias para hacer el clustering
    df_distancias_ordenado = df_distancias.sort_values(by='distancia').reset_index(drop=True)

    # Utilizamos la columna 'distancia' del DataFrame ordenado para aplicar KMeans
    dist_array = df_distancias_ordenado['distancia'].values.reshape(-1, 1)

    # Clustering con KMeans sobre las distancias
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(dist_array)

    # Reordenamos los centroides y reasignamos etiquetas para que el cluster 0 tenga las distancias más bajas
    centroids = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centroids)
    label_map = {original: new for new, original in enumerate(sorted_indices)}

    # Aplicamos el nuevo etiquetado
    df_distancias_ordenado['cluster'] = cluster_labels
    df_distancias_ordenado['cluster_ordenado'] = df_distancias_ordenado['cluster'].map(label_map)

    return df_distancias_ordenado, kmeans




def generar_matriz_ck(df_distancias_ordenado, df_cw):
    """
    Recibe:
    - df_distancias_ordenado: DataFrame con columnas ['tienda', 'vehiculo', 'distancia', 'cluster_ordenado']
    - df_cw: DataFrame con columnas ['tienda', 'rutas', 'carga', 'distancia', 'carga_total', 'n_camiones_utilizados', 'n_camiones_disponibles']
    Retorna:
    - df_ordenado: DataFrame con columnas ['tienda', 'vehiculo', 'distancia', 'cluster_ordenado', 'n_k (total clientes)', 'centroide', 'c_k']
    """
    df_cw = df_cw.copy()

    n_clientes_ruta = []
    # Calcular el número de clientes por ruta en cada tienda y vehículo
    for _, row in df_distancias_ordenado.iterrows():
        tienda = row['tienda']
        vehiculo_idx = int(row['vehiculo'])
        rutas_lista = df_cw.loc[df_cw['tienda'] == tienda, 'rutas'].iloc[0]
        n_clientes = len(rutas_lista[vehiculo_idx])
        n_clientes_ruta.append(n_clientes)

    # Agregar la columna n_clientes_ruta al df
    df_distancias_ordenado['n_clientes_ruta'] = n_clientes_ruta

    # Calcular los centroides (promedio de distancia por cluster)
    centroides_ordenados = df_distancias_ordenado.groupby(
        'cluster_ordenado')['distancia'].mean().reset_index()
    centroides_ordenados.columns = ['cluster_ordenado', 'centroide']

    # Calcular el número total de clientes por cluster (nk)
    nk_por_cluster = df_distancias_ordenado.groupby(
        'cluster_ordenado')['n_clientes_ruta'].sum().reset_index()
    nk_por_cluster.columns = ['cluster_ordenado', 'n_k (total clientes)']

    # Unir los cálculos de n_k y centroide por cluster
    nk_ck_cluster = pd.merge(
        nk_por_cluster, centroides_ordenados, on='cluster_ordenado')

    # Calcular c_k
    nk_ck_cluster['c_k'] = nk_ck_cluster['centroide'] / \
        nk_ck_cluster['n_k (total clientes)']

    # Agregar las columnas de c_k y n_k al df_ordenado original
    df_distancias_ordenado = pd.merge(df_distancias_ordenado, nk_ck_cluster[[
                                      'cluster_ordenado', 'n_k (total clientes)', 'centroide', 'c_k']], on='cluster_ordenado')

    return df_distancias_ordenado


def resolver_precio_optimo(df_distancias_ordenado, beta=0.0152, theta=0.9, max_price=55.9, num_precios=100):
    """
    Recibe:
    - df_distancias_ordenado: DataFrame con las columnas ['tienda', 'vehiculo', 'distancia', 'cluster_ordenado', 'n_k (total clientes)', 'centroide', 'c_k']

    Retorna:
    - df_precios_tienda_ruta: DataFrame con las mismas columnas de entrada más las columnas 'p_k (entero óptimo)' y 'U_k (entero óptimo)'
    """
    price_candidates = np.linspace(0, max_price, num_precios)
    price_indices = list(range(len(price_candidates)))

    # Crear un modelo de optimización
    model = LpProblem("Maximizar_Utilidad_Por_Cluster", LpMaximize)

    # Definir las variables binarias para las rutas y los precios
    x = {
        (k, i): LpVariable(f"x_{k}_{i}", cat='Binary')
        for k in df_distancias_ordenado.index
        for i in price_indices
    }

    # Variables para los precios óptimos por cluster
    p_vars = {
        k: LpVariable(f"p_{k}", lowBound=0, cat=LpContinuous)
        for k in df_distancias_ordenado.index
    }

    # Definir la función objetivo
    model += lpSum([
        row['n_k (total clientes)'] *
        ((-beta * price_candidates[i] + theta) *
         (price_candidates[i] - row['c_k']) * x[(k, i)])
        for k, row in df_distancias_ordenado.iterrows()
        for i in price_indices
    ])

    # Restricciones para asegurar que cada ruta tenga exactamente un precio
    for k in df_distancias_ordenado.index:
        model += lpSum([x[(k, i)] for i in price_indices]) == 1

    # Definir las restricciones para los precios óptimos
    for k in df_distancias_ordenado.index:
        model += p_vars[k] == lpSum([price_candidates[i] * x[(k, i)]
                                    for i in price_indices])

    # Asegurarse de que el precio óptimo sea mayor o igual al c_k
    for k, row in df_distancias_ordenado.iterrows():
        model += p_vars[k] >= row['c_k']

    # Restringir los precios para que sean no decrecientes entre clusters
    cluster_indices = list(df_distancias_ordenado.index)
    for k1, k2 in zip(cluster_indices, cluster_indices[1:]):
        model += p_vars[k2] >= p_vars[k1]

    # Resolver el modelo
    model.solve()

    # Recopilar los precios óptimos y las utilidades para cada cluster
    optimal_prices = []
    optimal_utils = []

    for k in df_distancias_ordenado.index:
        # Identificar el índice del precio óptimo
        selected_i = [i for i in price_indices if x[(k, i)].value() == 1][0]
        p_opt = price_candidates[selected_i]
        n_k = df_distancias_ordenado.loc[k, 'n_k (total clientes)']
        c_k = df_distancias_ordenado.loc[k, 'c_k']
        A_k = -beta * p_opt + theta
        U_k = n_k * A_k * (p_opt - c_k)
        optimal_prices.append(p_opt)
        optimal_utils.append(U_k)

    # Crear un nuevo DataFrame con los precios y utilidades óptimos
    df_precios_tienda_ruta = df_distancias_ordenado.copy()
    df_precios_tienda_ruta['p_k (entero óptimo)'] = optimal_prices
    df_precios_tienda_ruta['U_k (entero óptimo)'] = optimal_utils

    return df_precios_tienda_ruta

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pulp import LpProblem, LpVariable, lpSum, LpContinuous

def resolver_precio_optimo_inv(df_distancias_ordenado, df_stock, df_demanda, df_zonas, beta=0.0152, theta=0.9, max_price=55.9, num_precios=100, P_LB=7, P_UB=46, M=1900000):
    """
    Recibe:
    - df_distancias_ordenado: DataFrame con las columnas ['tienda', 'vehiculo', 'distancia', 'cluster_ordenado', 'n_k (total clientes)', 'centroide', 'c_k']
    - df_stock: DataFrame con las columnas ['id_tienda', 'id_producto', 'reorden', 'stock_actual']
    - Otros parámetros de la optimización como 'beta', 'theta', 'max_price', 'num_precios', 'P_LB', 'P_UB', 'M'
    
    Retorna:
    - df_precios_tienda_ruta: DataFrame con las mismas columnas de entrada más las columnas 'p_k (entero óptimo)' y 'U_k (entero óptimo)'
    """
    # Verificar el nivel de inventario para cada tienda
    df_distancias_ordenado['stock_actual'] = df_distancias_ordenado['tienda'].map(df_stock.set_index('id_tienda')['stock_actual'])
    # Unimos df_zonas con df_demanda para agregar la tienda correspondiente a cada zona
    df_merged = pd.merge(df_demanda, df_zonas[['id_zona', 'tienda_zona']], on='id_zona', how='left')

    # Agrupar por 'tienda_zona' y sumar la 'venta_digital' para obtener la demanda total por tienda
    df_demanda_por_tienda = df_merged.groupby('tienda_zona')['venta_digital'].sum().reset_index()

    # Calcular I_UB como el 1.3% de la demanda total por tienda
    df_demanda_por_tienda['I_UB'] = df_demanda_por_tienda['venta_digital'] * 1.3  # 1.3% de la demanda total por tienda

    
    # Crear los candidatos de precios
    price_candidates = np.linspace(0, max_price, num_precios)
    price_indices = list(range(len(price_candidates)))

    # Crear el modelo de optimización
    model = LpProblem("Maximizar_Utilidad_Por_Cluster", LpMaximize)

    # Definir las variables binarias para las rutas y los precios
    x = {
        (k, i): LpVariable(f"x_{k}_{i}", cat='Binary')
        for k in df_distancias_ordenado.index
        for i in price_indices
    }

    # Variables para los precios óptimos por cluster
    p_vars = {
        k: LpVariable(f"p_{k}", lowBound=0, cat=LpContinuous)
        for k in df_distancias_ordenado.index
    }

    # Definir la función objetivo
    model += lpSum([
        row['n_k (total clientes)'] * 
        ((-beta * price_candidates[i] + theta) * 
        (price_candidates[i] - row['c_k']) * x[(k, i)])
        for k, row in df_distancias_ordenado.iterrows()
        for i in price_indices
    ])

    # Restricciones para asegurar que cada ruta tenga exactamente un precio
    for k in df_distancias_ordenado.index:
        model += lpSum([x[(k, i)] for i in price_indices]) == 1

    # Definir las restricciones para los precios óptimos
    for k in df_distancias_ordenado.index:
        model += p_vars[k] == lpSum([price_candidates[i] * x[(k, i)] for i in price_indices])

    # Asegurarse de que el precio óptimo sea mayor o igual al c_k
    for k, row in df_distancias_ordenado.iterrows():
        model += p_vars[k] >= row['c_k']

    # Restringir los precios para que sean no decrecientes entre clusters
    cluster_indices = list(df_distancias_ordenado.index)
    for k1, k2 in zip(cluster_indices, cluster_indices[1:]):
        model += p_vars[k2] >= p_vars[k1]

    # Verificar si el nivel de inventario es bajo
    for k, row in df_distancias_ordenado.iterrows():
        tienda = row['tienda']
        stock_actual = row['stock_actual']

        I_UB_tienda = df_demanda_por_tienda.loc[df_demanda_por_tienda['tienda_zona'] == tienda, 'I_UB'].values[0]

        if stock_actual < I_UB_tienda:
            # Añadir las restricciones si el nivel de inventario es bajo
            if row['n_k (total clientes)'] > 500:  # Ajuste según el número de clientes en el cluster (basado en simulación)
                W_i = 0
                Z_i = 1  # Precio ajustado a P_UB
            else:
                # Si la demanda es baja, ajustamos el precio a P_LB
                W_i = 1
                Z_i = 0 

            # Restricciones de precios
            model += p_vars[k] <= P_LB - (1 - W_i) * M
            model += p_vars[k] >= P_UB - (1 - Z_i) * M
            model += W_i + Z_i == 1

    # Resolver el modelo
    model.solve()

    # Recopilar los precios óptimos y las utilidades para cada cluster
    optimal_prices = []
    optimal_utils = []

    for k in df_distancias_ordenado.index:
        selected_i = [i for i in price_indices if x[(k, i)].value() == 1][0]
        p_opt = price_candidates[selected_i]
        n_k = df_distancias_ordenado.loc[k, 'n_k (total clientes)']
        c_k = df_distancias_ordenado.loc[k, 'c_k']
        A_k = -beta * p_opt + theta
        U_k = n_k * A_k * (p_opt - c_k)
        optimal_prices.append(p_opt)
        optimal_utils.append(U_k)

    # Crear un nuevo DataFrame con los precios y utilidades óptimos
    df_precios_tienda_ruta = df_distancias_ordenado.copy()
    df_precios_tienda_ruta['p_k (entero óptimo)'] = optimal_prices
    df_precios_tienda_ruta['U_k (entero óptimo)'] = optimal_utils

    return df_precios_tienda_ruta


def resolver_precio_optimo_zona(df_distancias_ordenado, df_stock, df_demanda, df_zonas, beta=0.0152, theta=0.9, max_price=55.9, num_precios=100, P_LB=7, P_UB=46, M=1900000):
    """
    Recibe:
    - df_distancias_ordenado: DataFrame con las columnas ['tienda', 'vehiculo', 'distancia', 'cluster_ordenado', 'n_k (total clientes)', 'centroide', 'c_k']
    - df_stock: DataFrame con las columnas ['id_tienda', 'id_producto', 'reorden', 'stock_actual']
    - df_demanda: DataFrame con las columnas ['id_zona', 'id_producto', 'venta_digital']
    - df_zonas: DataFrame con las columnas ['id_zona', 'tienda_zona', 'x_zona', 'y_zona']
    - Otros parámetros de optimización como 'beta', 'theta', 'max_price', 'num_precios', 'P_LB', 'P_UB', 'M'
    
    Retorna:
    - df_precios_tienda_zona: DataFrame con ['id_zona', 'distancia', 'cluster_ordenado', 'n_clientes_ruta', 'n_k (total clientes)', 'centroide', 'c_k', 'p_k (entero óptimo)', 'U_k (entero óptimo)']
    """
    
    # Paso 1: Unir df_demanda con df_zonas para obtener la tienda correspondiente a cada zona
    df_merged = pd.merge(df_demanda, df_zonas[['id_zona', 'tienda_zona']], on='id_zona', how='left')

    # Paso 2: Unir df_merged con df_stock para agregar stock_actual por tienda y producto
    df_merged = pd.merge(df_merged, df_stock[['id_tienda', 'id_producto', 'stock_actual']], 
                         left_on=['tienda_zona', 'id_producto'], right_on=['id_tienda', 'id_producto'], how='left')

    # Paso 3: Calcular I_UB como el 1.3% de la demanda total por producto y tienda
    df_merged['I_UB'] = df_merged['venta_digital'] * 1.3  # 1.3% de la demanda total por producto y tienda

    # Paso 4: Unir df_merged con df_distancias_ordenado usando 'tienda_zona' para obtener la información de la tienda
    df_final = pd.merge(df_distancias_ordenado, df_merged[['id_zona', 'tienda_zona', 'venta_digital', 'stock_actual', 'I_UB']], 
                        left_on='tienda', right_on='tienda_zona', how='left')

    # Crear los candidatos de precios
    price_candidates = np.linspace(0, max_price, num_precios)
    price_indices = list(range(len(price_candidates)))

    # Paso 5: Crear el modelo de optimización
    model = LpProblem("Maximizar_Utilidad_Por_Cluster", LpMaximize)

    # Definir las variables binarias para las rutas y los precios
    x = {
        (k, i): LpVariable(f"x_{k}_{i}", cat='Binary')
        for k in df_final.index
        for i in price_indices
    }

    # Variables para los precios óptimos por cluster
    p_vars = {
        k: LpVariable(f"p_{k}", lowBound=0, cat=LpContinuous)
        for k in df_final.index
    }

    # Definir la función objetivo
    model += lpSum([
        row['n_k (total clientes)'] * 
        ((-beta * price_candidates[i] + theta) * 
        (price_candidates[i] - row['c_k']) * x[(k, i)])
        for k, row in df_final.iterrows()
        for i in price_indices
    ])

    # Restricciones para asegurar que cada ruta tenga exactamente un precio
    for k in df_final.index:
        model += lpSum([x[(k, i)] for i in price_indices]) == 1

    # Definir las restricciones para los precios óptimos
    for k in df_final.index:
        model += p_vars[k] == lpSum([price_candidates[i] * x[(k, i)] for i in price_indices])

    # Asegurarse de que el precio óptimo sea mayor o igual al c_k
    for k, row in df_final.iterrows():
        model += p_vars[k] >= row['c_k']

    # Restringir los precios para que sean no decrecientes entre clusters
    cluster_indices = list(df_final.index)
    for k1, k2 in zip(cluster_indices, cluster_indices[1:]):
        model += p_vars[k2] >= p_vars[k1]

    # Verificar si el nivel de inventario por producto y tienda es bajo
    for k, row in df_final.iterrows():
        tienda = row['tienda']
        stock_actual = row['stock_actual']
        I_UB_tienda_producto = row['I_UB']

        if stock_actual < I_UB_tienda_producto:
            # Añadir las restricciones si el nivel de inventario es bajo
            if row['n_k (total clientes)'] > 500:  # Ajuste según el número de clientes en el cluster
                W_i = 0
                Z_i = 1  # Precio ajustado a P_UB
            else:
                # Si la demanda es baja, ajustamos el precio a P_LB
                W_i = 1
                Z_i = 0 

            # Restricciones de precios
            model += p_vars[k] <= P_LB - (1 - W_i) * M
            model += p_vars[k] >= P_UB - (1 - Z_i) * M
            model += W_i + Z_i == 1

    # Resolver el modelo
    model.solve()

    # Recopilar los precios óptimos y las utilidades para cada cluster
    optimal_prices = []
    optimal_utils = []

    for k in df_final.index:
        selected_i = [i for i in price_indices if x[(k, i)].value() == 1][0]
        p_opt = price_candidates[selected_i]
        n_k = df_final.loc[k, 'n_k (total clientes)']
        c_k = df_final.loc[k, 'c_k']
        A_k = -beta * p_opt + theta
        U_k = n_k * A_k * (p_opt - c_k)
        optimal_prices.append(p_opt)
        optimal_utils.append(U_k)

    # Crear un nuevo DataFrame con los precios y utilidades óptimos
    df_precios_tienda_ruta = df_final.copy()
    df_precios_tienda_ruta['p_k (entero óptimo)'] = optimal_prices
    df_precios_tienda_ruta['U_k (entero óptimo)'] = optimal_utils

    # Paso 6: Agrupar por `id_zona` para obtener el precio promedio por zona
    df_precios_tienda_zona = df_precios_tienda_ruta.groupby('id_zona').agg({
        'distancia': 'mean',  # Promedio de las distancias
        'cluster_ordenado': 'first',  # Primer cluster para la zona
        'n_clientes_ruta': 'sum',  # Número total de clientes por ruta
        'n_k (total clientes)': 'sum',  # Total de clientes en la zona
        'centroide': 'mean',  # Promedio de los centroides
        'c_k': 'mean',  # Promedio de los costos
        'p_k (entero óptimo)': 'mean',  # Promedio de los precios óptimos
        'U_k (entero óptimo)': 'mean'  # Promedio de las utilidades
    }).reset_index()
    return df_precios_tienda_zona


def caso_base(df_demanda):
    """
    Recibe:
    - df_demanda: DataFrame con las columnas ['id_zona', 'id_producto', 'venta_digital']
    Retorna:
    - df_tarifas_zona: DataFrame con las columnas ['id_zona', 'venta_digital', 'P_i', 'A_k(P_i)']
    """
    suma_por_zona = df_demanda.groupby('id_zona')['venta_digital'].sum().reset_index()
    suma_por_zona = suma_por_zona.sort_values(by='venta_digital', ascending=False)
    suma_por_zona['P_i'] = 32.8 * suma_por_zona['venta_digital']
    suma_por_zona['A_k(P_i)'] = -0.0152 * suma_por_zona['P_i'] + 0.9
    suma_por_zona['U_k(P_i)'] = 0 * suma_por_zona['P_i'] 

    df_tarifa_zona = suma_por_zona[['id_zona', 'venta_digital', 'P_i', 'A_k(P_i)', 'U_k(P_i)']].copy()
    return df_tarifa_zona
