import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('tiendas_20250115.csv')
df['pos_x'] = df['pos_x'].astype(int)
df['pos_y'] = df['pos_y'].astype(int)
df['pos_x'] -= 1
df['pos_y'] -= 1
df_zonas = pd.read_csv('zonas_20250115.csv')

print(df)

print(df.describe())

largo_x = max(df_zonas['x_zona'])
largo_y = max(df_zonas['y_zona'])

tiendas_pequenas = df[df['tipo_tienda'] == 'pequena']
tiendas_medianas = df[df['tipo_tienda'] == 'mediana']
tiendas_grandes = df[df['tipo_tienda'] == 'grande']

plt.figure(figsize=(largo_x, largo_y))
plt.xlim(0, (df_zonas['x_zona'].max()))
plt.ylim(0, (df_zonas['y_zona'].max()))


# plt.scatter(df_zonas['x_zona'] + 0.5, df_zonas['y_zona'] + 0.5,
#             c='gray', marker='s', alpha=0.3, label='Zonas')

plt.scatter(tiendas_pequenas['pos_x'] + 0.5, tiendas_pequenas['pos_y'] + 0.5,
            c='blue', marker='o', label='Tienda pequeña')
plt.scatter(tiendas_medianas['pos_x'] + 0.5, tiendas_medianas['pos_y'] + 0.5,
            c='green', marker='o', label='Tienda mediana')
plt.scatter(tiendas_grandes['pos_x'] + 0.5, tiendas_grandes['pos_y'] + 0.5,
            c='red', marker='o', label='Tienda grande')

plt.title('Mapa de las tiendas')
plt.xlabel('pos_x')
plt.ylabel('pos_y')

plt.xticks(range(largo_x + 2), labels=[])
plt.yticks(range(largo_y + 2), labels=[])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

print(max(df_zonas['id_zona']))

plt.show()
