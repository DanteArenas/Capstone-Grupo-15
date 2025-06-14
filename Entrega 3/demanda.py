import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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

#Generar resumen tiendas con parametros

from scipy.stats import norm, lognorm, gamma, poisson
import numpy as np
import pandas as pd

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

#Generar resumen zonas con parametros:

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


#GENERAR DEMANDA

import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gamma, poisson
import os

# Leer los parámetros por producto
resumen_zonas_con_param = pd.read_excel('resumen_zonas_con_parametros.xlsx')
resumen_zonas_con_param.set_index('id_producto', inplace=True)

def generar_variacion_estocastica(distribucion, param1, param2=None):
    """
    Genera una variación estocástica según la distribución especificada.
    """
    if pd.isna(param1) or (param2 is not None and pd.isna(param2)):
        return 0

    if distribucion == 'Poisson':
        return np.random.poisson(lam=param1)

    elif distribucion == 'Normal':
        if param2 is None:
            return 0
        return max(0, int(np.random.normal(loc=param1, scale=param2)))

    elif distribucion == 'Uniforme':
        if param2 is None:
            return 0
        return np.random.randint(int(param1), int(param2) + 1)

    elif distribucion in ['Log-Normal', 'LogNormal']:
        if param2 is None:
            return 0
        return max(0, int(np.random.lognormal(mean=param1, sigma=param2)))

    elif distribucion == 'Gamma':
        if param2 is None:
            return 0
        return max(0, int(np.random.gamma(shape=param1, scale=param2)))

    elif distribucion == 'Constante':
        return int(param1)

    else:
        return 0

#DEMANDA DIGITAL

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

#Generar demanda digital para los 40 días

import os
import pandas as pd

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



