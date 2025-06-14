# En este archivo se crearan las simulaciones de ruteo DIARIO y la sumatoria de costos
# se necesita:
# - función que genera rutas de entrega, dando un conjunto de demandas digitales
# - función que calcule el costo diario
# - función que calcule el costo total
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import os
from itertools import combinations
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.pyplot as plt


def generar_rutas(path_zonas, path_tiendas, path_venta_zona, path_flota, path_camiones, path_productos, dia):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    zonas_data = pd.read_csv(path_zonas)
    tiendas_data = pd.read_csv(path_tiendas)
    tiendas_data['pos_x'] -= 1
    tiendas_data['pos_y'] -= 1
    coords_zonas = zonas_data[['x_zona', 'y_zona']].values
    coords_tiendas = tiendas_data[['pos_x', 'pos_y']].values

    dist_matrix = cdist(coords_zonas, coords_zonas, metric='euclidean')

    # Se calcula la matríz de distancias eucledianas entre solo zonas.
    dist_df = pd.DataFrame(
        dist_matrix,
        index=zonas_data['id_zona'],
        columns=zonas_data['id_zona']
    )

    clientes_1_data = pd.read_csv(path_venta_zona)
    flota_data = pd.read_csv(path_flota)
    camiones_data = pd.read_csv(path_camiones)
    productos_data = pd.read_csv(path_productos)

    clientes_productos_data = pd.merge(
        clientes_1_data, productos_data, on='id_producto', how='inner')
    clientes_productos_data = clientes_productos_data.drop(
        columns=[col for col in clientes_productos_data.columns if "Unnamed" in col])

    zonas_datos = pd.merge(clientes_productos_data, zonas_data, on='id_zona')
    zonas_datos = zonas_datos.drop(
        columns=[col for col in zonas_datos.columns if "Unnamed" in col])

    print("Datos de zonas con demanda:")
    print(zonas_datos.head())
    # De la forma: id_zona  id_producto  venta_digital  volumen  x_zona  y_zona  tienda_zona

    print("Datos de tiendas:")
    for index, row in tiendas_data.iterrows():
        print(f"Fila {index}:")
        print(row)
        print("-" * 40)
    # id_tienda  tipo_tienda  pos_x   pos_y

    # Agrupar por tienda física
    tiendas = zonas_datos['tienda_zona'].unique()
    rutas_totales = {}

    demanda_insatisfecha = pd.DataFrame(
        columns=['id_zona', 'demanda', 'tienda'])
    # Demanda insatisfecha por tienda

    # Algoritmo Clarke-Wright

    for index, row in tiendas_data.iterrows():
        tienda = row['id_tienda']
        print(f"\nProcesando tienda: {tienda}")
        # row de la forma: id_tienda tipo_tienda pos_x  pos_y

        # Subconjunto de zonas asociadas a esta tienda
        sub_zonas = zonas_datos[zonas_datos['tienda_zona'] == tienda].copy()
        sub_zonas = sub_zonas.reset_index(drop=True)
        # sub_zonas de la forma: id_zona  id_producto  venta_digital  volumen  x_zona  y_zona  tienda_zona

        # Obtener tipo y cantidad de camiones para la tienda
        flota_info = flota_data[flota_data['id_tienda'] == tienda]
        if flota_info.empty:
            print(f"No hay datos de flota para tienda {tienda}, se omite.")
            continue

        id_camion = flota_info.iloc[0]['id_camion']
        n_camiones = flota_info.iloc[0]['N']
        capacidad = camiones_data.loc[camiones_data['tipo_camion']
                                      == id_camion, 'Q'].values[0]

        # Establecer el depósito como primer punto
        deposito_coord = row[['pos_x', 'pos_y']].values.astype(float)

        print(f"Coordenadas del depósito: {deposito_coord}")

        deposito_filas = sub_zonas[
            (sub_zonas['x_zona'] == deposito_coord[0]) & (
                sub_zonas['y_zona'] == deposito_coord[1])
        ]
        print(f"Filas del depósito: {deposito_filas}")

        if not deposito_filas.empty:
            # Eliminar las filas correspondientes depósito de sub_zonas para evitar duplicados
            sub_zonas = sub_zonas.drop(deposito_filas.index)
        else:
            raise ValueError("El depósito no está en sub_zonas.")

        sub_zonas_agrupadas = sub_zonas.groupby('id_zona').agg({
            'venta_digital': 'sum',
            'x_zona': 'first',
            'y_zona': 'first'
        }).reset_index()

        print(f"Sub zonas agrupadas: {sub_zonas_agrupadas}")
        # sub_zonas_agrupadas de la forma: id_zona  venta_digital  x_zona  y_zona

        coords = np.vstack(
            [deposito_coord, sub_zonas_agrupadas[['x_zona', 'y_zona']].values])
        # coords de la forma: [x_zona, y_zona] del depósito y de las zonas

        volumen_total_zona_deposito = 0
        for indice_deposito_filas, fila_deposito_filas in deposito_filas.iterrows():
            # suma el volumen total de la zona de depósito, por producto
            volumen_total_zona_deposito += fila_deposito_filas['volumen'] * \
                fila_deposito_filas['venta_digital']

        sub_zonas['volumen_total'] = sub_zonas['volumen'] * \
            sub_zonas['venta_digital']  # Volumen total de cada producto (fila)
        # El volumen total de los productos a repartir en cada zona, sin la del depísito(creo)

        carga_agrupada_por_zona = sub_zonas.groupby(
            'id_zona')['volumen_total'].sum().reset_index()
        print(f"carga_agrupada_por_zona: {carga_agrupada_por_zona}")
        # carga_agrupada_por_zona de la forma: id_zona  volumen_total

        # carga_ruta = np.concatenate(np.array([carga_zona_deposito]),carga_agrupada_por_zona['volumen'].values)
        print(f"deposito_filas: {deposito_filas}")
        print(f"numero de zonas: {len(sub_zonas_agrupadas)}")
        print(f"largo carga_agrupada_por_zona: {len(carga_agrupada_por_zona)}")

        dist = cdist(coords, coords)
        # dist de la forma: matriz de distancias euclidianas entre el depósito y las zonas

        # carga ruta con volumen total de la zona de depósito primero
        # y luego el volumen total de cada zona
        carga_ruta = np.concatenate(
            ([volumen_total_zona_deposito], carga_agrupada_por_zona['volumen_total'].values))
        print(f"carga_ruta: {carga_ruta}")
        print(f"coords: {coords}")
        print(f"dist: {dist}")

        print("largo nodos (coords):", coords.shape[0])
        print("largo carga_ruta:", carga_ruta.shape[0])

        rutas = {i: [0, i, 0] for i in range(1, len(coords))}
        # rutas de la forma: {i: [0, i, 0]} donde i es el índice de la zona en coords, sin contar el depósito

        savings = []

        for i, j in combinations(range(1, len(coords)), 2):
            # Calcular el ahorro de distancia al unir las rutas i y j
            # dist[0, i] es la distancia del depósito a la zona i
            # dist[0, j] es la distancia del depósito a la zona j
            # dist[i, j] es la distancia entre las zonas i y j
            # El ahorro es la distancia del depósito a i + la distancia del depósito a j - la distancia entre i y j
            s = dist[0, i] + dist[0, j] - dist[i, j]
            savings.append((s, i, j))
        savings.sort(reverse=True)

        for s, i, j in savings:
            ruta_i = next((r for r in rutas.values() if i in r[1:-1]), None)
            ruta_j = next((r for r in rutas.values() if j in r[1:-1]), None)

            if ruta_i is None or ruta_j is None or ruta_i == ruta_j:
                continue

            carga_i = sum(carga_ruta[k] for k in ruta_i if k != 0)
            carga_j = sum(carga_ruta[k] for k in ruta_j if k != 0)
            if carga_i + carga_j > capacidad:
                continue

            if ruta_i[-2] == i and ruta_j[1] == j:
                nueva_ruta = ruta_i[:-1] + ruta_j[1:]
            elif ruta_j[-2] == j and ruta_i[1] == i:
                nueva_ruta = ruta_j[:-1] + ruta_i[1:]
            else:
                continue

            rutas = {k: v for k, v in rutas.items() if v !=
                     ruta_i and v != ruta_j}
            rutas[i] = nueva_ruta
            if tienda in [7, 10]:
                print(f"rutas actuales tienda {tienda}:")
                print(rutas)

        # === Seleccionar como máximo N rutas con menor distancia total ===
        rutas_finales = []
        id_zona_tienda = deposito_filas.iloc[0]['id_zona']
        print(f"ID zona tienda: {id_zona_tienda}")
        for ruta in rutas.values():
            zonas_ruta = [sub_zonas_agrupadas.iloc[i - 1]['id_zona']
                          if i != 0 else id_zona_tienda for i in ruta]
            carga = sum(carga_ruta[i] for i in ruta if i != 0)
            distancia = sum(dist[ruta[k]][ruta[k+1]]
                            for k in range(len(ruta)-1))
            rutas_finales.append({
                'ruta': zonas_ruta,
                'carga': carga,
                'distancia': round(distancia, 2)
            })

        # priorizar rutas cortas
        rutas_finales.sort(key=lambda r: r['distancia'])
        rutas_finales = rutas_finales[:n_camiones]  # Limitar a N rutas
        # cada fila de rutas_finales es de la forma: {'ruta': [id_zona1, id_zona2, ...], 'carga': carga, 'distancia': distancia}

        # rutas_totales[tienda] = rutas_finales

        # 1. Recolectar demanda total original por zona
        demanda_por_zona = sub_zonas_agrupadas[[
            'id_zona', 'venta_digital']].copy()
        demanda_por_zona.rename(
            columns={'venta_digital': 'demanda'}, inplace=True)

        # 2. Listado de zonas visitadas
        zonas_visitadas = set()
        for r in rutas_finales:
            zonas_visitadas |= set(r['ruta'])

        # 3. Zonas sin cubrir (demanda insatisfecha)
        demanda_insat_tienda = demanda_por_zona[
            ~demanda_por_zona['id_zona'].isin(zonas_visitadas)
        ].copy()
        demanda_insat_tienda['tienda'] = tienda
        print("Zonas con demanda insatisfecha:")
        print(demanda_insat_tienda)

        # --- Cheapest insertion para la demanda insatisfecha (ChatGPT) ---
        zonas_demanda_insat = demanda_insat_tienda[[
            'id_zona', 'demanda']].to_dict('records')  # lisa de dicts
        # zonas_demanda_insat de la forma: [{'id_zona': id_zona, 'demanda': demanda}, ...]
        print("zonas_demanda_insat:")
        print(zonas_demanda_insat)

        residuales = []
        # Calcular la capacidad residual de cada camión y guardar en residuales
        for r in rutas_finales:
            carga_usada = r['carga']
            residuales.append(capacidad - carga_usada)

        # residuales de la forma: [capacidad_residual_camion_1, capacidad_residual_camion_2, ...]

        idx_por_zona = {id_zona: idx for idx, id_zona in enumerate(
            [id_zona_tienda] + list(sub_zonas_agrupadas['id_zona']))}

        volumen_zona = dict(zip(
            carga_agrupada_por_zona['id_zona'],
            carga_agrupada_por_zona['volumen_total']
        ))

        id_zona_deposito = deposito_filas.iloc[0]['id_zona']
        volumen_zona[id_zona_deposito] = volumen_total_zona_deposito
        for zona in zonas_demanda_insat[:]:
            # mejor inserción de la zona encoontrada
            best = {'delta': float('inf')}
            for idx_r, r in enumerate(rutas_finales):  # para cada ruta (camión)
                # ver si el camión tiene capacidad para agregar la zona
                if residuales[idx_r] < zona['demanda']:
                    continue
                # recorre pares consecutivos en r['ruta']
                ruta = r['ruta']
                for k in range(len(ruta)-1):
                    i, j = ruta[k], ruta[k+1]
                    # utilizar los indices de la matriz de distancias
                    i_idx = idx_por_zona[i]
                    z_idx = idx_por_zona[zona['id_zona']]
                    j_idx = idx_por_zona[j]
                    delta = dist[i_idx, z_idx] + \
                        dist[z_idx, j_idx] - dist[i_idx, j_idx]
                    # delta es la distancia de insertar la zona a la ruta
                    if delta < best['delta']:
                        best = {'delta': delta, 'r_idx': idx_r, 'pos': k+1}
            if 'r_idx' in best:
                # inserto la zona
                ruta_obj = rutas_finales[best['r_idx']]
                ruta_obj['ruta'].insert(best['pos'], zona['id_zona'])
                # Recalcula usando la ruta correcta:
                nueva_carga = sum(volumen_zona[z] for z in ruta_obj['ruta'])
                ruta_obj['carga'] = nueva_carga
                residuales[best['r_idx']] = capacidad - nueva_carga
                ruta_obj['distancia'] = sum(
                    dist[idx_por_zona[ruta_obj['ruta'][k]],
                         idx_por_zona[ruta_obj['ruta'][k+1]]]
                    for k in range(len(ruta_obj['ruta'])-1)
                )
                zonas_demanda_insat.remove(zona)

            # si no hay best válido, esa zona permanece sobrante

        # 4) Acumular en un DataFrame global
        for z in zonas_demanda_insat:
            demanda_insatisfecha = pd.concat([
                demanda_insatisfecha,
                pd.DataFrame(
                    [{'id_zona': z['id_zona'], 'demanda': z['demanda'], 'tienda': tienda}])
            ], ignore_index=True)

        # 5) Guardar rutas finales
        rutas_totales[tienda] = rutas_finales

    # === Mostrar resultados ===
    resultados = []
    n_camiones_disponibles_por_tienda = {}
    for index, row in flota_data.iterrows():
        tienda = row['id_tienda']
        n_camiones_disponibles_por_tienda[tienda] = row['N']

    for tienda, rutas in rutas_totales.items():
        print(f"\n Rutas desde tienda: {tienda}")
        for i, r in enumerate(rutas):
            print(
                f"  Ruta {i+1}: {r['ruta']}, Carga: {r['carga']}, Distancia: {r['distancia']}")
        resultados.append({
            'tienda': tienda,
            'rutas': [r['ruta'] for r in rutas],
            'carga': [r['carga'] for r in rutas],
            'distancia': [r['distancia'] for r in rutas],
            'carga_total': sum(r['carga'] for r in rutas),
            'n_camiones_utilizados': len(rutas),
            'n_camiones_disponibles': n_camiones_disponibles_por_tienda[tienda]
        })

    df_resultados = pd.DataFrame(resultados)

    # Guardar resultados en CSV
    output_path = os.path.join(
        base_dir, 'resultados', f'dia_{dia}', f'resultados_CW_dia_{dia}.csv')

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    df_resultados.to_csv(output_path, index=False)

    demanda_insatisfecha_path = os.path.join(
        base_dir, 'resultados', f'dia_{dia}', f'demanda_insatisfecha_CW_dia_{dia}.csv')
    if not os.path.exists(os.path.dirname(demanda_insatisfecha_path)):
        os.makedirs(os.path.dirname(demanda_insatisfecha_path))
    demanda_insatisfecha.to_csv(demanda_insatisfecha_path, index=False)

    return df_resultados


def graficar_rutas(data_resultados, path_zonas, path_tiendas, dia, mejora_2_opt=False):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # data_resultados = pd.read_csv(path_resultados)
    data_zonas = pd.read_csv(path_zonas)
    data_zonas = data_zonas.drop(
        columns=[col for col in data_zonas.columns if "Unnamed" in col])

    df = pd.read_csv(path_tiendas)
    df['pos_x'] = df['pos_x'].astype(int)
    df['pos_y'] = df['pos_y'].astype(int)
    df['pos_x'] -= 1
    df['pos_y'] -= 1

    df_zonas = pd.read_csv(path_zonas)

    distancia_total = 0
    plt.figure(constrained_layout=True)
    for i, row in data_resultados.iterrows():

        rutas = row['rutas']
        # rutas = rutas.split("], [")
        for i in range(len(rutas)):
            # rutas[i] = rutas[i].strip("[]").split(", ")
            for j in range(len(rutas[i])):
                rutas[i][j] = int(rutas[i][j])

        camion = 1
        for ruta in rutas:
            xs_camion = []
            ys_camion = []
            for id_zona in ruta:
                zona = data_zonas.loc[data_zonas['id_zona'] == id_zona]
                xs_camion.append(zona['x_zona'].values[0])
                ys_camion.append(zona['y_zona'].values[0])

            plt.plot(xs_camion, ys_camion, zorder=2)
            camion += 1

        distancia = row['distancia']
        # lista_distancia = distancia.strip("[]").split(",")
        for i in range(len(distancia)):
            distancia_total += float(distancia[i])
        print(f"Distancia recorrida para la tienda {camion}: {distancia}")

    # ---------- Mapa de tiendas de entrega 1 ---------- #

    largo_x = max(df_zonas['x_zona'])
    largo_y = max(df_zonas['y_zona'])

    tiendas_pequenas = df[df['tipo_tienda'] == 'pequena']
    tiendas_medianas = df[df['tipo_tienda'] == 'mediana']
    tiendas_grandes = df[df['tipo_tienda'] == 'grande']

    plt.xlim(0-1, (df_zonas['x_zona'].max())+1)
    plt.ylim(0-1, (df_zonas['y_zona'].max())+1)

    plt.scatter(tiendas_pequenas['pos_x'], tiendas_pequenas['pos_y'],
                c='blue', marker='o', label='Tienda pequeña', zorder=3)
    plt.scatter(tiendas_medianas['pos_x'], tiendas_medianas['pos_y'],
                c='green', marker='o', label='Tienda mediana', zorder=3)
    plt.scatter(tiendas_grandes['pos_x'], tiendas_grandes['pos_y'],
                c='red', marker='o', label='Tienda grande', zorder=3)

    if mejora_2_opt:
        plt.title('Rutas de Camiones - Clarke and Wright mejoradas con 2-opt')
    else:
        plt.title('Rutas de Camiones - Clarke and Wright')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(range(largo_x + 2), labels=[])
    plt.yticks(range(largo_y + 2), labels=[])

    plt.xticks(range(largo_x + 2), labels=[])
    plt.yticks(range(largo_y + 2), labels=[])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')  # Chatgpt
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

    # ---------- Gráfico de las zonas (de entrega pasada, con ayuda de Chatgpt) ---------- #

    # Crear tabla de zonas
    grilla_tiendas = df_zonas.pivot_table(
        index='y_zona',
        columns='x_zona',
        values='tienda_zona',
        fill_value=-1
    )

    zonas_unicas = sorted(grilla_tiendas.stack().unique())

    colormap = plt.get_cmap('tab20', len(zonas_unicas))
    color_map = ListedColormap([colormap(i) for i in range(len(zonas_unicas))])
    norm = BoundaryNorm(zonas_unicas + [max(zonas_unicas)+1], color_map.N)

    grilla_indices = grilla_tiendas
    print(grilla_tiendas)

    # Dibujar grilla
    im = plt.imshow(grilla_indices,  cmap=color_map,
                    norm=norm, zorder=1, alpha=0.5)

    output_dir = os.path.join(base_dir, 'resultados',
                              f'dia_{dia}', 'graficos_CW')
    os.makedirs(output_dir, exist_ok=True)

    nombre_archivo = f'rutas_camiones_todas_las_tiendas_CW_mejoradas_2opt_dia_{dia}.png' if mejora_2_opt \
        else f'rutas_camiones_todas_las_tiendas_CW_dia_{dia}.png'

    plt.savefig(os.path.join(output_dir, nombre_archivo))
    # plt.show()
    print(
        f"Distancia total recorrida por todos los camiones en el día {dia}: {distancia_total} unidades")
    df_distancia_total = pd.DataFrame(
        {'distancia_total': [distancia_total]}
    )
    if not os.path.exists(os.path.join(base_dir, 'resultados', f'dia_{dia}')):
        os.makedirs(os.path.join(base_dir, 'resultados', f'dia_{dia}'))

    if not mejora_2_opt:
        df_distancia_total.to_csv(os.path.join(
            base_dir, 'resultados', f'dia_{dia}', f'distancia_total_CW_dia_{dia}.csv'), index=False)
    else:
        df_distancia_total.to_csv(os.path.join(
            base_dir, 'resultados', f'dia_{dia}', f'distancia_total_mejorada_2opt_CW_dia_{dia}.csv'), index=False)

    return True


def dos_opt_swap(lista_ruta, first, second):
    """
    Realiza un swap de dos nodos en la ruta.
    """
    nueva_ruta = lista_ruta.copy()
    nueva_ruta = lista_ruta[:first] + \
        lista_ruta[first:second+1][::-1] + lista_ruta[second+1:]
    return nueva_ruta


def calcular_distancia_ruta(ruta, dist_df):
    """
    Calcula la distancia total de una ruta dada.
    """
    distancia_total = 0
    for i in range(len(ruta) - 1):
        distancia_total += dist_df.loc[ruta[i], ruta[i + 1]]
    return distancia_total


def mejora_2_opt_ruta(ruta, dist_df):
    # adaptado de https://slowandsteadybrain.medium.com/traveling-salesman-problem-ce78187cf1f3
    nueva_distancia = float('inf')
    mejor_distancia = calcular_distancia_ruta(ruta, dist_df)
    ruta_actual = ruta.copy()
    mejorado = True
    while mejorado:
        mejorado = False
        for i in range(1, len(ruta) - 2):    # Para no hacer swap con el depósito
            for j in range(i + 1, len(ruta)-1):
                if j - i == 1:
                    continue
                nueva_ruta = dos_opt_swap(ruta_actual, i, j)
                nueva_distancia = calcular_distancia_ruta(nueva_ruta, dist_df)
                if nueva_distancia < mejor_distancia:
                    print(
                        f'nueva mejor distancia: {nueva_distancia}, mejor distancia anterior: {mejor_distancia}')
                    mejor_distancia = nueva_distancia
                    ruta_actual = nueva_ruta
                    mejorado = True
                    break
            if mejorado:
                break
    return {'ruta_mejorada': ruta_actual, 'distancia': mejor_distancia}


def mejorar_rutas_2_opt(data_resultados, path_zonas, path_tiendas, dia):
    print("Mejorando rutas con 2-opt...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_zonas = pd.read_csv(path_zonas)
    data_tiendas = pd.read_csv(path_tiendas)

    coords_zonas = data_zonas[['x_zona', 'y_zona']].values
    dist_matrix = cdist(coords_zonas, coords_zonas, metric='euclidean')

    # Se calcula la matríz de distancias euclidianas entre solo zonas.
    dist_df = pd.DataFrame(
        dist_matrix,
        index=data_zonas['id_zona'],
        columns=data_zonas['id_zona']
    )

    data_resultados_mejorados = data_resultados.copy()

    for i, row in data_resultados_mejorados.iterrows():
        rutas = row['rutas']
        print(f"Rutas originales para tienda {row['tienda']}: {rutas}")
        nuevas_rutas = []
        nuevas_distancias = []
        for j in range(len(rutas)):
            ruta_mejorada = mejora_2_opt_ruta(rutas[j], dist_df)
            if ruta_mejorada['ruta_mejorada'] != rutas[j]:
                print(
                    f"Ruta mejorada para tienda {row['tienda']}: {ruta_mejorada['ruta_mejorada']}")
                # Calcular distancia total de la ruta mejorada
                print(
                    f"Distancia total de la ruta mejorada: {ruta_mejorada['distancia']}")
                print(f'Distancia anterior: {row["distancia"][j]}')
            else:
                print(
                    f"No se mejoró la ruta para tienda {row['tienda']}: {ruta_mejorada}")
            nuevas_rutas.append(ruta_mejorada['ruta_mejorada'])
            nuevas_distancias.append(ruta_mejorada['distancia'])
        data_resultados_mejorados.at[i, 'rutas'] = nuevas_rutas
        data_resultados_mejorados.at[i, 'distancia'] = nuevas_distancias
    output_path = os.path.join(
        base_dir, 'resultados', f'dia_{dia}', f'resultados_mejorados_CW_dia_{dia}.csv')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    data_resultados_mejorados.to_csv(output_path, index=False)

    return data_resultados_mejorados
