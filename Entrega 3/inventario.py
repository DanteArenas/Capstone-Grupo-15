import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import norm, lognorm, gamma, poisson
import json


# Función para cargar ventas
def cargar_ventas(ruta_base):
    ventas = []
    for dia in range(1, 41):
        archivo = f"{ruta_base}_{dia}_20250115.csv"
        if os.path.exists(archivo):
            df = pd.read_csv(archivo)
            df['dia'] = dia
            ventas.append(df)
    return pd.concat(ventas, ignore_index=True)

# Carga de datos únicos
flota = pd.read_csv('flota_20250115.csv')
productos = pd.read_csv('productos_20250115.csv')
proporcion_eleccion = pd.read_csv('proporcion_eleccion_de_usuarios_20250115.csv')
#demanda_insatisfecha = pd.read_csv('demanda_online_insatisfecha_20250115.csv')
reorden = pd.read_csv('reorden_20250115.csv')
tiendas = pd.read_csv('tiendas_20250115.csv')
vehiculos = pd.read_csv('vehiculos_20250115.csv')
zonas = pd.read_csv('zonas_20250115.csv')

# Carga de ventas físicas y digitales
ventas_tienda = cargar_ventas('venta_tienda')
ventas_tienda.rename(columns={'venta_tienda': 'cantidad'}, inplace=True)

ventas_zona = cargar_ventas('venta_zona')
ventas_zona.rename(columns={'venta_digital': 'cantidad'}, inplace=True)

costo_inventario_unitario = 3.733

# Generar resumen tiendas con parámetros:

# === 1. Usar SOLO ventas físicas ===
ventas_fisicas = ventas_tienda[['id_producto', 'cantidad']]

# === 2. Ajustar distribuciones por producto ===
parametros_tienda = []

for id_producto, grupo in ventas_fisicas.groupby("id_producto"):
    cantidades = grupo["cantidad"].values
    positivas = cantidades[cantidades > 0]

    # Normal
    mu, sigma = norm.fit(cantidades)
    parametros_tienda.append({
        'id_producto': id_producto,
        'distribucion': 'Normal',
        'parametro1': mu,
        'parametro2': sigma
    })

    # Log-Normal
    if len(positivas) > 0:
        try:
            shape, loc, scale = lognorm.fit(positivas, floc=0)
            mu_ln = np.log(scale)
            sigma_ln = shape
            parametros_tienda.append({
                'id_producto': id_producto,
                'distribucion': 'Log-Normal',
                'parametro1': mu_ln,
                'parametro2': sigma_ln
            })
        except:
            pass

    # Gamma
    if len(positivas) > 0:
        try:
            shape_g, loc_g, scale_g = gamma.fit(positivas, floc=0)
            parametros_tienda.append({
                'id_producto': id_producto,
                'distribucion': 'Gamma',
                'parametro1': shape_g,
                'parametro2': scale_g
            })
        except:
            pass

    # Poisson
    lambda_p = cantidades.mean()
    parametros_tienda.append({
        'id_producto': id_producto,
        'distribucion': 'Poisson',
        'parametro1': lambda_p,
        'parametro2': None
    })

df_param_tienda = pd.DataFrame(parametros_tienda)

# === 3. Fusionar con archivo resumen original (para tomar el mejor ajuste) ===
resumen_tienda = pd.read_excel('resumen_tiendas.xlsx')

df_tienda_completo = resumen_tienda.merge(
    df_param_tienda,
    left_on=['id_producto', 'mejor_ajuste'],
    right_on=['id_producto', 'distribucion'],
    how='left'
)

df_tienda_completo.drop(columns=['distribucion'], inplace=True)

# === 4. Guardar Excel final ===
df_tienda_completo.to_excel('resumen_tiendas_con_parametros.xlsx', index=False)

# Generar resumen zonas con parámetros:

# === 1. Ajuste sobre ventas digitales (zonas) ===
parametros_zona = []

for id_producto, grupo in ventas_zona.groupby("id_producto"):
    cantidades = grupo["cantidad"].values
    positivas = cantidades[cantidades > 0]

    # Normal
    mu, sigma = norm.fit(cantidades)
    parametros_zona.append({'id_producto': id_producto, 'distribucion': 'Normal', 'parametro1': mu, 'parametro2': sigma})

    # Log-Normal
    if len(positivas) > 0:
        try:
            shape, loc, scale = lognorm.fit(positivas, floc=0)
            mu_ln = np.log(scale)
            sigma_ln = shape
            parametros_zona.append({
                'id_producto': id_producto,
                'distribucion': 'Log-Normal',
                'parametro1': mu_ln,
                'parametro2': sigma_ln
            })
        except:
            pass

    # Gamma
    if len(positivas) > 0:
        try:
            shape_g, loc_g, scale_g = gamma.fit(positivas, floc=0)
            parametros_zona.append({
                'id_producto': id_producto,
                'distribucion': 'Gamma',
                'parametro1': shape_g,
                'parametro2': scale_g
            })
        except:
            pass

    # Poisson
    lambda_p = cantidades.mean()
    parametros_zona.append({'id_producto': id_producto, 'distribucion': 'Poisson', 'parametro1': lambda_p, 'parametro2': None})

df_param_zona = pd.DataFrame(parametros_zona)

# === 2. Fusionar con el resumen original de zonas
resumen_zona = pd.read_excel('resumen_zonas.xlsx')

df_zona_completo = resumen_zona.merge(
    df_param_zona,
    left_on=['id_producto', 'mejor_ajuste'],
    right_on=['id_producto', 'distribucion'],
    how='left'
)

df_zona_completo.drop(columns=['distribucion'], inplace=True)

# === 3. Guardar
df_zona_completo.to_excel('resumen_zonas_con_parametros.xlsx', index=False)

import pandas as pd
import numpy as np
import os

# Leer los parámetros por producto
resumen_zonas_con_param = pd.read_excel('resumen_zonas_con_parametros.xlsx')
resumen_zonas_con_param.set_index('id_producto', inplace=True)

def generar_variacion_estocastica(distribucion, param1, param2=None):
    if pd.isna(param1) or (param2 is not None and pd.isna(param2)):
        return 1.0

    if distribucion == "Poisson":
        variacion = np.random.poisson(lam=param1) / param1 if param1 > 0 else 1.0
    elif distribucion == "Normal":
        if param2 is None or param2 <= 0:
            return 1.0
        variacion = np.random.normal(loc=param1, scale=param2) / param1 if param1 > 0 else 1.0
    elif distribucion == "Uniforme":
        if param2 is None:
            return 1.0
        variacion = np.random.uniform(low=param1, high=param2) / ((param1 + param2) / 2)
    elif distribucion in ["Log-Normal", "LogNormal"]:
        if param2 is None or param2 <= 0:
            return 1.0
        variacion = np.random.lognormal(mean=param1, sigma=param2) / np.exp(param1)
    elif distribucion == "Gamma":
        if param2 is None or param2 <= 0:
            return 1.0
        variacion = np.random.gamma(shape=param1, scale=param2) / (param1 * param2)
    else:
        return 1.0

    return max(0.7, min(variacion, 1.3))


# === FUNCIÓN PRINCIPAL PARA UN DÍA ===
def generar_demanda_digital_estocastica(ventas_zona_dia):
    demanda_estocastica = []

    for _, row in ventas_zona_dia.iterrows():
        id_zona = row['id_zona']
        id_producto = row['id_producto']
        base = row['cantidad']

        if id_producto not in resumen_zonas_con_param.index:
            total = base
        else:
            fila = resumen_zonas_con_param.loc[id_producto]
            dist = fila['mejor_ajuste']
            p1 = fila['parametro1']
            p2 = fila['parametro2'] if not pd.isna(fila['parametro2']) else None
            variacion = generar_variacion_estocastica(dist, p1, p2)
            total = max(0, int(base + variacion))

        demanda_estocastica.append({
            'id_zona': id_zona,
            'id_producto': id_producto,
            'venta_digital': total
        })

    return pd.DataFrame(demanda_estocastica)


def generar_ventas_zona_estocasticas(ruta_base_original='venta_zona', carpeta_salida='ventas_zona_estocasticas', dias=40):
    """
    Genera archivos estocásticos tipo venta_zona_dia_X.csv en base a archivos de ventas digitales originales.
    
    - ruta_base_original: nombre base de los archivos originales (sin _n_20250115.csv)
    - carpeta_salida: carpeta donde guardar los nuevos archivos
    - dias: cantidad de días a simular (por defecto: 40)
    """
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    for dia in range(1, dias + 1):
        archivo_original = f"{ruta_base_original}_{dia}_20250115.csv"
        if not os.path.exists(archivo_original):
            print(f"⚠️ Archivo no encontrado: {archivo_original}")
            continue

        ventas_zona_dia = pd.read_csv(archivo_original)
        ventas_zona_dia.rename(columns={'venta_digital': 'cantidad'}, inplace=True)

        # Aplicar variación estocástica
        ventas_estocasticas = generar_demanda_digital_estocastica(ventas_zona_dia)

        # Guardar archivo con el nuevo nombre
        nombre_salida = os.path.join(carpeta_salida, f"venta_zona_dia_{dia}.csv")
        ventas_estocasticas.to_csv(nombre_salida, index=False)

        print(f"✅ Generado: {nombre_salida}")


# Mapear tiendas a zonas (correr 1 vez)
zonas = pd.read_csv('zonas_20250115.csv')  # Contiene columna tienda_zona
ventas_zona = ventas_zona.merge(zonas[['id_zona', 'tienda_zona']], on='id_zona')
ventas_zona.rename(columns={'tienda_zona': 'id_tienda'}, inplace=True)



distribuciones_tienda = pd.read_excel('resumen_tiendas_con_parametros.xlsx')
distribuciones_zona = pd.read_excel('resumen_zonas_con_parametros.xlsx')


def simulacion_unificada(distribuciones_tienda, distribuciones_zona,
                         ventas_tienda, ventas_zona, reorden,
                         dias_simulados=10, frecuencia_reabastecimiento=5,
                         costo_inventario_unitario=3.733, reorden_multiplicador=1.0, semilla=42):

    np.random.seed(semilla)
    # Ajustar reorden
    reorden_mod = reorden.copy()
    reorden_mod['reorden'] = (reorden_mod['reorden'] * reorden_multiplicador).clip(lower=1).astype(int)

    # Diccionarios rápidos
    dist_tienda_dict = distribuciones_tienda.set_index('id_producto')[['mejor_ajuste', 'parametro1', 'parametro2']].to_dict('index')
    dist_zona_dict = distribuciones_zona.set_index('id_producto')[['mejor_ajuste', 'parametro1', 'parametro2']].to_dict('index')

    # Inicializar stock
    stock = reorden_mod[['id_tienda', 'id_producto', 'reorden']].copy()
    stock['stock_actual'] = stock['reorden']
    stock.set_index(['id_tienda', 'id_producto'], inplace=True)
    stock_dia = []

    registro_dia_a_dia = []
    costos_diarios = []

    for dia in range(1, dias_simulados + 1):
        vt = ventas_tienda[ventas_tienda['dia'] == dia][['id_tienda', 'id_producto', 'cantidad']]
        vz = ventas_zona[ventas_zona['dia'] == dia][['id_tienda', 'id_producto', 'cantidad']]
        demanda = pd.concat([vt, vz], ignore_index=True)

        demanda_total = demanda.groupby(['id_tienda', 'id_producto'])['cantidad'].sum().reset_index()

        # Generar variación estocástica
        demanda_total['variacion'] = demanda_total.apply(
            lambda row: generar_variacion_estocastica(
                *(dist_tienda_dict.get(row['id_producto']) or dist_zona_dict.get(row['id_producto']) or ('Constante', 1.0, None))
            ),
            axis=1
        )
        demanda_total['cantidad_ajustada'] = (demanda_total['cantidad'] * demanda_total['variacion']).round().astype(int)

        for _, row in demanda_total.iterrows():
            clave = (row['id_tienda'], row['id_producto'])
            if clave in stock.index:
                disponible = stock.at[clave, 'stock_actual']
                atendido = min(disponible, row['cantidad_ajustada'])
                no_atendido = row['cantidad_ajustada'] - atendido
                stock.at[clave, 'stock_actual'] -= atendido

                registro_dia_a_dia.append({
                    'dia': dia,
                    'id_tienda': row['id_tienda'],
                    'id_producto': row['id_producto'],
                    'demanda': row['cantidad_ajustada'],
                    'atendido': atendido,
                    'no_atendido': no_atendido,
                    'stock_restante': stock.at[clave, 'stock_actual']
                })

        stock_total = stock['stock_actual'].sum()
        costos_diarios.append(stock_total * costo_inventario_unitario)

        df_stock_dia = stock.reset_index().copy()
        df_stock_dia['dia'] = dia
        df_stock_dia['reorden'] = reorden_mod.set_index(['id_tienda', 'id_producto']).loc[stock.index, 'reorden'].values
        stock_dia.append(df_stock_dia)
        

        if dia % frecuencia_reabastecimiento == 0:
            stock['stock_actual'] += stock['reorden'] - stock['stock_actual']

    resultados_df = pd.DataFrame(registro_dia_a_dia)
    df_stock_diario = pd.concat(stock_dia, ignore_index=True)
    
    demanda_total = resultados_df['demanda'].sum()
    atendido_total = resultados_df['atendido'].sum()
    no_atendido_total = resultados_df['no_atendido'].sum()
    costo_total = sum(costos_diarios)
    

    return {
        'nivel_servicio': atendido_total / demanda_total if demanda_total > 0 else 0,
        'costo_total': costo_total,
        'costo_diario_promedio': costo_total / dias_simulados,
        'demanda_no_atendida_total': no_atendido_total,
        'costos_diarios': costos_diarios,
        'registro_dia_a_dia': resultados_df,
        'stock_diario': df_stock_diario
    }

## CASO BASE
resultados = simulacion_unificada(
    distribuciones_tienda=distribuciones_tienda,
    distribuciones_zona=distribuciones_zona,
    ventas_tienda=ventas_tienda,
    ventas_zona=ventas_zona,
    reorden=reorden,
    dias_simulados=40,
    frecuencia_reabastecimiento=5,
    costo_inventario_unitario=3.733,
    reorden_multiplicador=1.0
)

print(f"Nivel de servicio: {resultados['nivel_servicio']:.2%}")
print(f"Costo total: ${resultados['costo_total']:,.0f}")
print(f"Costo diario promedio: ${resultados['costo_diario_promedio']:,.0f}")
print(f"Demanda no atendida total: {resultados['demanda_no_atendida_total']:,}")


# Simulación con reorden multiplicador 0.3925 (mejor política)
resultados_finales_4 = simulacion_unificada(
    distribuciones_tienda=distribuciones_tienda,
    distribuciones_zona=distribuciones_zona,
    ventas_tienda=ventas_tienda,
    ventas_zona=ventas_zona,
    reorden_multiplicador=0.3925,
    reorden=reorden,
    dias_simulados=40,
    frecuencia_reabastecimiento=2,
    costo_inventario_unitario=3.733
)

print("\n📊 Resultados finales de la mejor política:")
print(f"📦 Nivel de servicio: {resultados_finales_4['nivel_servicio']:.2%}")
print(f"💰 Costo total inventario: ${resultados_finales_4['costo_total']:,.0f}")
print(f"Costo diario promedio: ${resultados_finales_4['costo_diario_promedio']:,.0f}")
print(f"🚫 Demanda no atendida total: {resultados_finales_4['demanda_no_atendida_total']:,}")


# kpi ventas perdidas por inventario
kpis_dict = {
    'caso': ['base', 'optima'],
    'ventas_perdidas': [
        resultados['demanda_no_atendida_total'],
        resultados_finales_4['demanda_no_atendida_total']
    ],
    'nivel_servicio': [
        resultados['nivel_servicio'],
        resultados_finales_4['nivel_servicio']
    ],
    'demanda_total': [
        resultados['demanda_total'],
        resultados_finales_4['demanda_total']
    ]
}
df_kpis = pd.DataFrame(kpis_dict)
df_kpis.to_csv('kpi_ventas_perdidas_inventario.csv', index=False)
