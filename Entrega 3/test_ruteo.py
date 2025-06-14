from ruteo import generar_rutas, graficar_rutas, mejorar_rutas_2_opt
import os

# path_zonas, path_tiendas, path_venta_zona, path_flota, path_camiones, path_producto, dia
base_dir = os.path.dirname(__file__)
print()
path_zonas = os.path.join(base_dir, '..', 'Datos', 'zonas_20250115.csv')
path_tiendas = os.path.join(base_dir, '..', 'Datos', 'tiendas_20250115.csv')
path_venta_zona_1 = os.path.join(
    base_dir, '..', 'Datos', 'venta_zona_1_20250115.csv')
path_flota = os.path.join(base_dir, '..', 'Datos', 'flota_20250115.csv')
path_camiones = os.path.join(base_dir, '..', 'Datos', 'vehiculos_20250115.csv')
path_productos = os.path.join(
    base_dir, '..', 'Datos', 'productos_20250115.csv')

dia = 1

data_resultados = generar_rutas(path_zonas, path_tiendas, path_venta_zona_1,
                                path_flota, path_camiones, path_productos, dia)
graficar_rutas(data_resultados, path_zonas, path_tiendas, dia)

data_resultados = mejorar_rutas_2_opt(
    data_resultados, path_zonas, path_tiendas, dia)
graficar_rutas(data_resultados, path_zonas,
               path_tiendas, dia, mejora_2_opt=True)
