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
