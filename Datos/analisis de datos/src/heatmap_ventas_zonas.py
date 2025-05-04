import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_zonas = pd.read_csv('zonas_20250115.csv')
df_ventas_1 = pd.read_csv('venta_zona_1_20250115.csv')

df_ventas_digitales_totales = df_ventas_1.drop(columns=['id_producto'])
df_ventas_digitales_totales = df_ventas_1.groupby('id_zona', as_index=False)[
    'venta_digital'].sum()

df_ventas_digitales_totales = pd.merge(
    df_zonas, df_ventas_digitales_totales, on='id_zona', how='left')

heatmap_data = df_ventas_digitales_totales.pivot_table(
    index='y_zona',
    columns='x_zona',
    values='venta_digital',
    fill_value=0
)

# # Crear heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, linewidths=1, linecolor='black',
            square=True, cmap='Reds', cbar_kws={'label': 'Ventas digitales', 'shrink': 0.5})
plt.title('Ventas digitales totales por zona día 1')
plt.xlabel('x_zona')
plt.ylabel('y_zona')
plt.gca().invert_yaxis()  # Para que el eje y coincida con orientación de mapa
plt.tight_layout()
plt.show()

ventas_cero = df_ventas_digitales_totales[df_ventas_digitales_totales['venta_digital'] == 0]
print("Zonas con venta_digital igual a 0:")
print(ventas_cero)

print('Mínimo de venta_digital:')
print(min(df_ventas_digitales_totales['venta_digital']))
