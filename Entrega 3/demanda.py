from scipy.stats import norm, lognorm, gamma, poisson
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
base_dir = os.path.dirname(os.path.abspath(__file__))
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')
path_proporcion_eleccion = os.path.join(
    base_dir, '..', 'Datos', 'proporcion_eleccion_de_usuarios_20250115.csv')
path_reorden = os.path.join(base_dir, '..', 'Datos', 'reorden_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_vehiculos = os.path.join(
    base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')

flota = pd.read_csv(path_flota)
productos = pd.read_csv(path_productos)
# Proporción de elección de usuarios
proporcion_eleccion = pd.read_csv(path_proporcion_eleccion)
# demanda_insatisfecha = pd.read_csv('demanda_online_insatisfecha_20250115.csv')
reorden = pd.read_csv(path_reorden)
tiendas = pd.read_csv(path_tiendas)
vehiculos = pd.read_csv(path_vehiculos)
zonas = pd.read_csv(path_zonas)

# Carga de ventas físicas y digitales
ventas_tienda = cargar_ventas(os.path.join(
    base_dir, '..', 'Datos', 'venta_tienda'))
ventas_tienda.rename(columns={'venta_tienda': 'cantidad'}, inplace=True)

ventas_zona = cargar_ventas(os.path.join(
    base_dir, '..', 'Datos', 'venta_zona'))
ventas_zona.rename(columns={'venta_digital': 'cantidad'}, inplace=True)

costo_inventario_unitario = 3.733

# Generar resumen tiendas con parametros


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
ruta_resumen_tiendas = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'resumen_tiendas.xlsx')
resumen_tienda = pd.read_excel(ruta_resumen_tiendas)

df_tienda_completo = resumen_tienda.merge(
    df_param_tienda,
    left_on=['id_producto', 'mejor_ajuste'],
    right_on=['id_producto', 'distribucion'],
    how='left'
)

df_tienda_completo.drop(columns=['distribucion'], inplace=True)

# === 4. Guardar Excel final ===
df_tienda_completo.to_excel('resumen_tiendas_con_parametros.xlsx', index=False)

# Generar resumen zonas con parametros:

# === 1. Ajuste sobre ventas digitales (zonas) ===
parametros_zona = []

for id_producto, grupo in ventas_zona.groupby("id_producto"):
    cantidades = grupo["cantidad"].values
    positivas = cantidades[cantidades > 0]

    # Normal
    mu, sigma = norm.fit(cantidades)
    parametros_zona.append({'id_producto': id_producto,
                           'distribucion': 'Normal', 'parametro1': mu, 'parametro2': sigma})

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
    parametros_zona.append({'id_producto': id_producto, 'distribucion': 'Poisson',
                           'parametro1': lambda_p, 'parametro2': None})

df_param_zona = pd.DataFrame(parametros_zona)

# === 2. Fusionar con el resumen original de zonas
ruta_resumen_zonas = os.path.join(
    base_dir, '..', 'Datos', 'analisis de datos', 'resumen_zonas.xlsx')
resumen_zona = pd.read_excel(ruta_resumen_zonas)

df_zona_completo = resumen_zona.merge(
    df_param_zona,
    left_on=['id_producto', 'mejor_ajuste'],
    right_on=['id_producto', 'distribucion'],
    how='left'
)

df_zona_completo.drop(columns=['distribucion'], inplace=True)

# === 3. Guardar
df_zona_completo.to_excel('resumen_zonas_con_parametros.xlsx', index=False)


# GENERAR DEMANDA


# Función actualizada: generación de variación estocástica acotada como factor multiplicador
def generar_variacion_estocastica_2(distribucion, param1, param2=None):
    """
    Genera un factor multiplicador estocástico entre ~0.7 y ~1.3
    según la distribución ajustada del producto.
    """
    # Si los parámetros son inválidos o faltantes, retornar 1.0 (sin variación)
    if pd.isna(param1) or (param2 is not None and pd.isna(param2)):
        return 1.0

    if distribucion == "Poisson":
        variacion = np.random.poisson(
            lam=param1) / param1 if param1 > 0 else 1.0
    elif distribucion == "Normal":
        if param2 is None or param2 <= 0:
            return 1.0
        variacion = np.random.normal(
            loc=param1, scale=param2) / param1 if param1 > 0 else 1.0
    elif distribucion == "Uniforme":
        if param2 is None:
            return 1.0
        variacion = np.random.uniform(
            low=param1, high=param2) / ((param1 + param2) / 2)
    elif distribucion in ["Log-Normal", "LogNormal"]:
        if param2 is None or param2 <= 0:
            return 1.0
        variacion = np.random.lognormal(
            mean=param1, sigma=param2) / np.exp(param1)
    elif distribucion == "Gamma":
        if param2 is None or param2 <= 0:
            return 1.0
        variacion = np.random.gamma(
            shape=param1, scale=param2) / (param1 * param2)
    else:
        return 1.0  # Distribución desconocida

    # Acotar la variación para evitar valores extremos
    variacion = max(0.7, min(variacion, 1.3))

    return variacion


def generar_demanda_digital_estocastica_2(ventas_zona_dia, resumen_parametros, seed=None):
    """
    Genera demanda digital estocástica para un DataFrame diario de ventas por zona,
    aplicando un factor multiplicador acotado según la distribución ajustada.
    """
    if seed is not None:
        np.random.seed(seed)

    demanda_estocastica = []

    for _, row in ventas_zona_dia.iterrows():
        id_zona = row['id_zona']
        id_producto = row['id_producto']
        base = row['venta_digital']

        if id_producto not in resumen_parametros.index:
            total = base
        else:
            fila = resumen_parametros.loc[id_producto]
            dist = fila['mejor_ajuste']
            p1 = fila['parametro1']
            p2 = fila['parametro2'] if not pd.isna(
                fila['parametro2']) else None
            variacion = generar_variacion_estocastica_2(
                dist, p1, p2)  # <- usa la nueva
            total = max(0, int(base * variacion))

        demanda_estocastica.append({
            'id_zona': id_zona,
            'id_producto': id_producto,
            'venta_digital': total
        })

    return pd.DataFrame(demanda_estocastica)


def generar_csvs_demanda_digital_estocastica_2_multiple_realizaciones(
    ruta_base=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), '..', 'Datos', 'venta_zona'),
    resumen_parametros_path='resumen_zonas_con_parametros.xlsx',
    seeds=[101, 202, 303, 404, 505]
):
    resumen_parametros = pd.read_excel(
        resumen_parametros_path).set_index('id_producto')

    for i, seed in enumerate(seeds, start=1):

        carpeta_salida = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), f'ventas_digitales_realizacion_{i}')
        os.makedirs(carpeta_salida, exist_ok=True)
        print(f"📁 Generando realización {i} con seed {seed}...")

        for dia in range(1, 41):
            archivo_original = f"{ruta_base}_{dia}_20250115.csv"
            if not os.path.exists(archivo_original):
                print(f"⚠️ No encontrado: {archivo_original}")
                continue

            ventas_zona_dia = pd.read_csv(archivo_original)
            df_estocastico = generar_demanda_digital_estocastica_2(
                ventas_zona_dia, resumen_parametros, seed=seed +
                dia
            )
            nombre_salida = os.path.join(
                carpeta_salida, f"venta_zona_estocastica_dia_{dia}.csv")
            df_estocastico.to_csv(nombre_salida, index=False)

        print(f"✅ Finalizada realización {i} en carpeta: {carpeta_salida}")


generar_csvs_demanda_digital_estocastica_2_multiple_realizaciones()
