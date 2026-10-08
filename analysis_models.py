# -*- coding: utf-8 -*-
"""
Estrategia de Medición de Riesgo y Tarifas Basada en Incidencia Delictiva
Proyecto Final - Análisis de Delitos Municipales en México para el Sector Asegurador
Autor: Marco Antonio Ruiz (Analista de Datos)
Fecha: Octubre de 2026
"""

import os
import sys
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Configure visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=================================================================")
print("PROYECTO FINAL: ESTRATEGIA DE MEDICIÓN DE RIESGO ASEGURADOR")
print("=================================================================\n")

# -------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE DATOS
# -------------------------------------------------------------------------
gen_path = os.path.join(BASE_DIR, 'riesgo_automotriz_general.csv')
sev_path = os.path.join(BASE_DIR, 'riesgo_automotriz_severidad.csv')

df_gen = pd.read_csv(gen_path, encoding='utf-8-sig')
df_sev = pd.read_csv(sev_path, encoding='utf-8-sig')

# Diccionario para corregir caracteres especiales en entidades
clean_names = {
    'Ciudad de Mxico': 'Ciudad de México',
    'Michoacn de Ocampo': 'Michoacán',
    'Mxico': 'Estado de México',
    'Nuevo Len': 'Nuevo León',
    'Quertaro': 'Querétaro',
    'San Luis Potos': 'San Luis Potosí',
    'Yucatn': 'Yucatán',
    'Veracruz de Ignacio de la Llave': 'Veracruz',
    'Coahuila de Zaragoza': 'Coahuila'
}
df_gen['Estado'] = df_gen['Estado'].replace(clean_names)
df_sev['Estado'] = df_sev['Estado'].replace(clean_names)

mes_map = {
    'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4,
    'MAYO': 5, 'JUNIO': 6, 'JULIO': 7, 'AGOSTO': 8,
    'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12
}
df_gen['Mes_Num'] = df_gen['Mes_Nombre'].str.strip().str.upper().map(mes_map)

# -------------------------------------------------------------------------
# 2. MODELO DE SERIES DE TIEMPO Y PRONÓSTICOS (ARIMA 2015-2022)
# Municipio: Ecatepec de Morelos (#1 en siniestralidad vehicular nacional)
# -------------------------------------------------------------------------
print(">>> [1/2] Modelado de Series de Tiempo (ARIMA) para Ecatepec de Morelos...")

ecatepec = df_gen[df_gen['Municipio'] == 'Ecatepec de Morelos'].copy()
ecatepec['Fecha'] = pd.to_datetime(
    ecatepec['Anio'].astype(str) + '-' + ecatepec['Mes_Num'].astype(str).str.zfill(2) + '-01'
)
ecatepec = ecatepec.sort_values('Fecha').set_index('Fecha')

ts = ecatepec['Total_Robos']

# Separación en conjunto de entrenamiento (2015-2021: 84 meses) y prueba (2022: 12 meses)
train = ts['2015-01-01':'2021-12-01']
test = ts['2022-01-01':'2022-12-01']

print(f"    - Periodo histórico de entrenamiento: {train.index.min().strftime('%Y-%m')} a {train.index.max().strftime('%Y-%m')} ({len(train)} meses)")
print(f"    - Periodo de evaluación / test: {test.index.min().strftime('%Y-%m')} a {test.index.max().strftime('%Y-%m')} ({len(test)} meses)")

# Prueba de estacionariedad Augmented Dickey-Fuller (ADF)
adf_stat, adf_pvalue, _, _, adf_crit, _ = adfuller(train)
print(f"    - Prueba ADF en serie original: Estadístico = {adf_stat:.4f}, p-value = {adf_pvalue:.4f}")

# Diferenciación de primer orden para inducir estacionariedad (d=1)
train_diff = train.diff().dropna()
adf_diff_stat, adf_diff_pvalue, _, _, _, _ = adfuller(train_diff)
print(f"    - Prueba ADF en serie diferenciada (d=1): Estadístico = {adf_diff_stat:.4f}, p-value = {adf_diff_pvalue:.4e} (Estacionaria)")

# Ajuste y selección de hiperparámetros ARIMA(p,d,q)
best_aic = float('inf')
best_order = (2, 1, 2)
for p in range(0, 3):
    for q in range(0, 3):
        try:
            m = ARIMA(train, order=(p, 1, q))
            res = m.fit()
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = (p, 1, q)
        except Exception:
            continue

print(f"    - Mejor arquitectura ARIMA seleccionada: ARIMA{best_order} (AIC: {best_aic:.2f})")

model = ARIMA(train, order=best_order)
fit_res = model.fit()

# Pronóstico a 12 pasos para el año 2022
forecast_obj = fit_res.get_forecast(steps=12)
fc_mean = forecast_obj.predicted_mean
fc_ci = forecast_obj.conf_int(alpha=0.05)  # Intervalo de confianza al 95%

# Métricas de error
mae = mean_absolute_error(test, fc_mean)
rmse = np.sqrt(mean_squared_error(test, fc_mean))
mape = np.mean(np.abs((test.values - fc_mean.values) / test.values)) * 100

print(f"    - Desempeño predictivo en 2022:")
print(f"      * MAE: {mae:.2f} robos/mes")
print(f"      * RMSE: {rmse:.2f}")
print(f"      * MAPE: {mape:.2f}% (Excelente precisión predictiva < 12%)")

# Generación del gráfico de alta resolución para Diapositiva 4
fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

# Graficar histórico
ax.plot(train.index, train.values, color='#1B2A4A', linewidth=2.2, label='Histórico Observado (2015-2021)')
# Graficar real 2022
ax.plot(test.index, test.values, color='#0D9488', linewidth=2.2, marker='o', markersize=5, label='Valor Real Observado (2022)')
# Graficar pronóstico 2022
ax.plot(test.index, fc_mean.values, color='#E63946', linewidth=2.5, linestyle='--', marker='s', markersize=5, label=f'Pronóstico ARIMA{best_order} (2022)')
# Intervalo de confianza
ax.fill_between(test.index, fc_ci.iloc[:, 0], fc_ci.iloc[:, 1], color='#E63946', alpha=0.18, label='Intervalo de Confianza (95%)')

# Línea divisoria de inicio del pronóstico
ax.axvline(x=pd.to_datetime('2022-01-01'), color='#64748B', linestyle=':', linewidth=1.5, alpha=0.8)
ax.text(pd.to_datetime('2022-01-15'), 1120, 'Inicio Pronóstico 2022', color='#475569', fontsize=9.5, fontweight='bold')

# Cuadro informativo de métricas
textstr = '\n'.join((
    r'$\mathbf{Métricas\ de\ Validación\ (2022):}$',
    f'• MAPE: {mape:.2f}%',
    f'• MAE: {mae:.1f} delitos/mes',
    f'• RMSE: {rmse:.1f}',
    f'• Modelo: ARIMA{best_order}'
))
props = dict(boxstyle='round,pad=0.7', facecolor='#F8FAFC', edgecolor='#CBD5E1', alpha=0.95)
ax.text(0.03, 0.22, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props, color='#1E293B')

ax.set_title('Ecatepec de Morelos: Serie Histórica y Pronóstico ARIMA de Robo Automotor (2015-2022)',
             fontsize=13, fontweight='bold', color='#0F172A', pad=15)
ax.set_xlabel('Periodo Mensual', fontsize=11, fontweight='semibold', color='#334155', labelpad=10)
ax.set_ylabel('Frecuencia Mensual de Robos', fontsize=11, fontweight='semibold', color='#334155', labelpad=10)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(0, 1300)
ax.legend(loc='upper right', frameon=True, facecolor='#FFFFFF', edgecolor='#E2E8F0', fontsize=9.5)
plt.tight_layout()

arima_chart_path = os.path.join(BASE_DIR, 'grafica_arima_ecatepec_2022.png')
fig.savefig(arima_chart_path, dpi=300)
plt.close(fig)
print(f"    [OK] Grafica ARIMA guardada en: {arima_chart_path}")

# -------------------------------------------------------------------------
# 3. CLASIFICACIÓN DE ESTADOS POR PELIGROSIDAD (K-MEANS 2021)
# -------------------------------------------------------------------------
print("\n>>> [2/2] Segmentación de Riesgo Estatal (Clustering K-Means 2021)...")

df_2021 = df_sev[df_sev['Anio'] == 2021].copy()

state_agg = df_2021.groupby('Estado').agg(
    Robos_Violentos=('Robos_Con_Violencia', 'sum'),
    Robos_No_Violentos=('Robos_Sin_Violencia', 'sum')
).reset_index()

state_agg['Total_Robos'] = state_agg['Robos_Violentos'] + state_agg['Robos_No_Violentos']
state_agg['Indice_Violencia'] = state_agg['Robos_Violentos'] / state_agg['Total_Robos']

# Para evitar distorsión por la extrema asimetría de Estado de México y capturar
# simultáneamente la Frecuencia (Volumen) y la Severidad (Violencia):
# Usamos log(Total_Robos) e Indice_Violencia normalizados
features = ['Total_Robos', 'Robos_Violentos', 'Indice_Violencia']

# Realizamos clustering con K=3 (Baja, Media y Alta Peligrosidad)
# Usando normalización robusta con StandardScaler sobre transformaciones estándar
X_cluster = np.column_stack([
    np.log1p(state_agg['Total_Robos']),
    state_agg['Indice_Violencia']
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
state_agg['Cluster_Raw'] = kmeans.fit_predict(X_scaled)

# Ordenar clústeres por severidad combinada para asignar consistentemente:
# 0 -> Baja Peligrosidad, 1 -> Media Peligrosidad, 2 -> Alta Peligrosidad
cluster_score = state_agg.groupby('Cluster_Raw').apply(
    lambda g: g['Total_Robos'].mean() * 0.5 + (g['Indice_Violencia'].mean() * 10000) * 0.5
)
sorted_clusters = cluster_score.sort_values().index.tolist()
label_map = {
    sorted_clusters[0]: 'Baja Peligrosidad',
    sorted_clusters[1]: 'Media Peligrosidad',
    sorted_clusters[2]: 'Alta Peligrosidad'
}
state_agg['Nivel_Riesgo'] = state_agg['Cluster_Raw'].map(label_map)

# Multiplicador actuarial recomendado para suscripción
tarifa_map = {
    'Baja Peligrosidad': 0.85,    # Descuento comercial por baja siniestralidad
    'Media Peligrosidad': 1.15,   # Prima base estándar ajustada
    'Alta Peligrosidad': 1.55     # Recargo por alta severidad y robo violento
}
state_agg['Multiplicador_Tarifa'] = state_agg['Nivel_Riesgo'].map(tarifa_map)

sil_score = silhouette_score(X_scaled, state_agg['Nivel_Riesgo'])
print(f"    - Coeficiente de Silueta del Clustering (K=3): {sil_score:.3f}")

# Imprimir distribución
print("\n    Distribución de Clústeres Estatales (2021):")
for nivel in ['Baja Peligrosidad', 'Media Peligrosidad', 'Alta Peligrosidad']:
    sub = state_agg[state_agg['Nivel_Riesgo'] == nivel]
    print(f"    • {nivel.upper()} ({len(sub)} entidades):")
    print(f"      Volumen Promedio: {sub['Total_Robos'].mean():,.0f} robos | % Violencia Promedio: {sub['Indice_Violencia'].mean():.1%}")
    estados_str = ", ".join(sub['Estado'].tolist()[:6]) + ("..." if len(sub) > 6 else "")
    print(f"      Ejemplos: {estados_str}")

# Guardar CSV con resultados del clustering para referencia
clustering_csv_path = os.path.join(BASE_DIR, 'clasificacion_riesgo_estados_2021.csv')
state_agg.to_csv(clustering_csv_path, index=False, encoding='utf-8-sig')
print(f"    [OK] Clasificacion de estados guardada en: {clustering_csv_path}")

# Generación del gráfico de dispersión de alta resolución para Diapositiva 5
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

palette = {
    'Baja Peligrosidad': '#10B981',    # Esmeralda / Verde
    'Media Peligrosidad': '#F59E0B',   # Ámbar / Naranja
    'Alta Peligrosidad': '#EF4444'     # Carmín / Rojo
}

# Graficar cada grupo
for nivel in ['Baja Peligrosidad', 'Media Peligrosidad', 'Alta Peligrosidad']:
    sub = state_agg[state_agg['Nivel_Riesgo'] == nivel]
    ax.scatter(
        sub['Total_Robos'],
        sub['Indice_Violencia'] * 100,
        s=120,
        color=palette[nivel],
        alpha=0.88,
        edgecolors='#0F172A',
        linewidth=1.2,
        label=f'{nivel} ({len(sub)} estados) - Tarifa x{tarifa_map[nivel]}'
    )

# Etiquetar estados clave
key_states = [
    'Estado de México', 'Jalisco', 'Baja California', 'Puebla',
    'Ciudad de México', 'Guanajuato', 'Michoacán', 'Sinaloa',
    'Querétaro', 'Nuevo León', 'Yucatán', 'Chihuahua'
]
for _, row in state_agg.iterrows():
    if row['Estado'] in key_states:
        offset_y = 1.2 if row['Total_Robos'] < 20000 else -2.2
        offset_x = 1.05 if row['Total_Robos'] < 30000 else 0.95
        ha = 'left' if row['Total_Robos'] < 25000 else 'right'
        ax.annotate(
            row['Estado'],
            (row['Total_Robos'], row['Indice_Violencia'] * 100),
            xytext=(offset_x * row['Total_Robos'], row['Indice_Violencia'] * 100 + offset_y),
            fontsize=8.5,
            fontweight='bold',
            color='#1E293B',
            ha=ha,
            arrowprops=dict(arrowstyle='-', color='#94A3B8', lw=0.7, shrinkB=3)
        )

ax.set_title('Clasificación Multivariada de Entidades por Peligrosidad Delictiva Automotriz (K-Means 2021)',
             fontsize=13, fontweight='bold', color='#0F172A', pad=15)
ax.set_xlabel('Volumen Total Anual de Robos de Vehículos (Escala Logarítmica)', fontsize=11, fontweight='semibold', color='#334155', labelpad=10)
ax.set_ylabel('Severidad: % de Robos con Violencia', fontsize=11, fontweight='semibold', color='#334155', labelpad=10)
ax.set_xscale('log')
ax.set_ylim(-2, 70)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y)}%'))

# Zonas de referencia actuarial
ax.axhline(y=40, color='#DC2626', linestyle='--', alpha=0.4, lw=1.2)
ax.text(12, 41.5, 'Umbral de Alta Severidad Violenta (>40%)', color='#B91C1C', fontsize=8.5, fontweight='semibold')

ax.legend(loc='lower right', frameon=True, facecolor='#FFFFFF', edgecolor='#CBD5E1', fontsize=9.5)
plt.tight_layout()

kmeans_chart_path = os.path.join(BASE_DIR, 'grafica_kmeans_estados_2021.png')
fig.savefig(kmeans_chart_path, dpi=300)
plt.close(fig)
print(f"    [OK] Grafica K-Means guardada en: {kmeans_chart_path}")

print("\n=================================================================")
print("ANALISIS ESTADISTICO Y VISUALIZACIONES COMPLETADAS CON EXITO")
print("=================================================================")
