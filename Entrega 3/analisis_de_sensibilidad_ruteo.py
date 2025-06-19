# este archivo juntará todo, se llamarán a las funciones de cada archivo y se generarán las simulaciones
from ruteo import generar_rutas, graficar_rutas, mejorar_rutas_2_opt, caso_base_ruteo
from pricing import procesar_datos_de_distancia, generar_matriz_ck, resolver_precio_optimo
import os
import pandas as pd

base_dir = os.path.dirname(__file__)
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_camiones = os.path.join(base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_flotaAS1 = os.path.join(base_dir, '..', 'Datos', 'flota_AS1.csv')
path_flotaAS2 = os.path.join(base_dir, '..', 'Datos', 'flota_AS2.csv')
path_camionesAS3 = os.path.join(base_dir, '..', 'Datos', 'vehiculos_AS3.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')


# Analisis de sensibilidad:
ids_sensibilidad = [
    'aumento_camiones_tiendas_2_13', 'disminucion_camiones_tiendas_7_8', 'disminucion_capacidad_camion_3']
# guardar ids en c
df_ids_sensibilidad = pd.DataFrame(
    ids_sensibilidad, columns=['id_sensibilidad'])
path_ids_sensibilidad = os.path.join(
    base_dir, 'resultados_sensibilidad', 'ids_sensibilidad_ruteo.csv')
os.makedirs(os.path.dirname(path_ids_sensibilidad), exist_ok=True)
df_ids_sensibilidad.to_csv(path_ids_sensibilidad, index=False)

# Caso 1: aumento de 1 camion en las tiendas 2 y 13 (por tener pocos camiones)
n_dias = 5
for dia in range(1, n_dias + 1):
    path_venta_zona = os.path.join(
        base_dir, 'ventas_digitales_estocasticas', f'venta_zona_estocastica_dia_{dia}.csv')

    # Generar rutas
    data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                    path_flotaAS1, path_camiones, path_productos, dia, sensibilidad=True, id_sensibilidad='aumento_camiones_tiendas_2_13')

    # Graficar rutas
    graficar_rutas(data_resultados, path_zonas, path_tiendas, dia,
                   sensibilidad=True, id_sensibilidad='aumento_camiones_tiendas_2_13')

    # Mejorar rutas con 2-opt
    data_resultados_mejorada = mejorar_rutas_2_opt(
        data_resultados, path_zonas, path_tiendas, dia, sensibilidad=True, id_sensibilidad='aumento_camiones_tiendas_2_13')

    # Graficar rutas mejoradas
    graficar_rutas(data_resultados_mejorada, path_zonas, path_tiendas,
                   dia, mejora_2_opt=True, sensibilidad=True, id_sensibilidad='aumento_camiones_tiendas_2_13')


# Caso 2: disminuir 1 camión en las tiendas 7 y 8 (por ser las que más tienen)
for dia in range(1, n_dias + 1):
    path_venta_zona = os.path.join(
        base_dir, 'ventas_digitales_estocasticas', f'venta_zona_estocastica_dia_{dia}.csv')

    # Generar rutas
    data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                    path_flotaAS2, path_camiones, path_productos, dia, sensibilidad=True, id_sensibilidad='disminucion_camiones_tiendas_7_8')

    # Graficar rutas
    graficar_rutas(data_resultados, path_zonas, path_tiendas, dia, sensibilidad=True,
                   id_sensibilidad='disminucion_camiones_tiendas_7_8')

    # Mejorar rutas con 2-opt
    data_resultados_mejorada = mejorar_rutas_2_opt(
        data_resultados, path_zonas, path_tiendas, dia, sensibilidad=True, id_sensibilidad='disminucion_camiones_tiendas_7_8')

    # Graficar rutas mejoradas
    graficar_rutas(data_resultados_mejorada, path_zonas, path_tiendas,
                   dia, mejora_2_opt=True, sensibilidad=True, id_sensibilidad='disminucion_camiones_tiendas_7_8')

# Caso 3: Disminuir capacidad del camión tipo 3 de 80.000.000 a 60.000.000
for dia in range(1, n_dias + 1):
    path_venta_zona = os.path.join(
        base_dir, 'ventas_digitales_estocasticas', f'venta_zona_estocastica_dia_{dia}.csv')

    # Generar rutas
    data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona,
                                    path_flota, path_camionesAS3, path_productos, dia, sensibilidad=True, id_sensibilidad='disminucion_capacidad_camion_3')

    # Graficar rutas
    graficar_rutas(data_resultados, path_zonas, path_tiendas, dia, sensibilidad=True,
                   id_sensibilidad='disminucion_capacidad_camion_3')

    # Mejorar rutas con 2-opt
    data_resultados_mejorada = mejorar_rutas_2_opt(
        data_resultados, path_zonas, path_tiendas, dia, sensibilidad=True, id_sensibilidad='disminucion_capacidad_camion_3')

    # Graficar rutas mejoradas
    graficar_rutas(data_resultados_mejorada, path_zonas, path_tiendas,
                   dia, mejora_2_opt=True, sensibilidad=True, id_sensibilidad='disminucion_capacidad_camion_3')
