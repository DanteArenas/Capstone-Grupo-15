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

def cargar_datos_csv(path):
    return pd.read_csv(path)

def procesar_datos_de_distancia(df, n_clusters=3):
    df['distancia'] = df['distancia'].apply(ast.literal_eval)

    distancias = []
    tiendas = []
    vehiculos = []

    for _, row in df.iterrows():
        tienda = row['tienda']
        for j, dist in enumerate(row['distancia']):
            distancias.append(float(dist))
            tiendas.append(tienda)
            vehiculos.append(j)

    df_dist = pd.DataFrame({
        'tienda': tiendas,
        'vehiculo': vehiculos,
        'distancia': distancias
    })

    df_ordenado = df_dist.sort_values(by='distancia').reset_index(drop=True)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans_labels = kmeans.fit_predict(df_ordenado[['distancia']])
    centroids = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centroids)
    label_map = {original: new for new, original in enumerate(sorted_indices)}

    df_ordenado['cluster'] = kmeans_labels
    df_ordenado['cluster_ordenado'] = df_ordenado['cluster'].map(label_map)

    return df_ordenado, kmeans

def generar_matriz_ck(df_distancias_ordenado, df_cw):
    df_cw = df_cw.copy()
    df_cw['rutas'] = df_cw['rutas'].apply(ast.literal_eval)

    n_clientes_ruta = []
    for _, row in df_distancias_ordenado.iterrows():
        tienda = row['tienda']
        vehiculo_idx = int(row['vehiculo'])
        rutas_lista = df_cw.loc[df_cw['tienda'] == tienda, 'rutas'].iloc[0]
        n_clientes = len(rutas_lista[vehiculo_idx])
        n_clientes_ruta.append(n_clientes)

    df_distancias_ordenado = df_distancias_ordenado.copy()
    df_distancias_ordenado['n_clientes_ruta'] = n_clientes_ruta

    centroides_ordenados = df_distancias_ordenado.groupby('cluster_ordenado')['distancia'].mean().reset_index()
    centroides_ordenados.columns = ['cluster_ordenado', 'centroide']

    nk_por_cluster = df_distancias_ordenado.groupby('cluster_ordenado')['n_clientes_ruta'].sum().reset_index()
    nk_por_cluster.columns = ['cluster_ordenado', 'n_k (total clientes)']

    nk_ck_cluster = pd.merge(nk_por_cluster, centroides_ordenados, on='cluster_ordenado')
    nk_ck_cluster['c_k'] = nk_ck_cluster['centroide'] / nk_ck_cluster['n_k (total clientes)']

    return nk_ck_cluster

def resolver_precio_optimo(nk_ck_cluster, beta=0.0152, theta=0.9, max_price=55.9, num_precios=100):
    price_candidates = np.linspace(0, max_price, num_precios)
    price_indices = list(range(len(price_candidates)))

    model = LpProblem("Maximizar_Utilidad_Por_Cluster", LpMaximize)

    x = {
        (k, i): LpVariable(f"x_{k}_{i}", cat='Binary')
        for k in nk_ck_cluster.index
        for i in price_indices
    }

    p_vars = {
        k: LpVariable(f"p_{k}", lowBound=0, cat=LpContinuous)
        for k in nk_ck_cluster.index
    }

    model += lpSum([
        row['n_k (total clientes)'] *
        ((-beta * price_candidates[i] + theta) *
         (price_candidates[i] - row['c_k']) * x[(k, i)])
        for k, row in nk_ck_cluster.iterrows()
        for i in price_indices
    ])

    for k in nk_ck_cluster.index:
        model += lpSum([x[(k, i)] for i in price_indices]) == 1

    for k in nk_ck_cluster.index:
        model += p_vars[k] == lpSum([price_candidates[i] * x[(k, i)] for i in price_indices])

    for k, row in nk_ck_cluster.iterrows():
        model += p_vars[k] >= row['c_k']

    cluster_indices = list(nk_ck_cluster.index)
    for k1, k2 in zip(cluster_indices, cluster_indices[1:]):
        model += p_vars[k2] >= p_vars[k1]

    model.solve()

    optimal_prices = []
    optimal_utils = []

    for k in nk_ck_cluster.index:
        selected_i = [i for i in price_indices if x[(k, i)].value() == 1][0]
        p_opt = price_candidates[selected_i]
        n_k = nk_ck_cluster.loc[k, 'n_k (total clientes)']
        c_k = nk_ck_cluster.loc[k, 'c_k']
        A_k = -beta * p_opt + theta
        U_k = n_k * A_k * (p_opt - c_k)
        optimal_prices.append(p_opt)
        optimal_utils.append(U_k)

    nk_ck_cluster = nk_ck_cluster.copy()
    nk_ck_cluster['p_k (entero óptimo)'] = optimal_prices
    nk_ck_cluster['U_k (entero óptimo)'] = optimal_utils

    return nk_ck_cluster
