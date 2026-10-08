# 🚗 Análisis de Delitos Municipales y Evaluación de Riesgo para Seguros de Automóviles

Proyecto final desarrollado para la industria de seguros de automóviles, enfocado en medir el riesgo geográfico a través de la incidencia delictiva municipal en México, permitiendo fijar primas de seguros competitivas y rentables.

---

## 📋 Tabla de Contenidos
1. [Contexto del Negocio](#contexto-del-negocio)
2. [Arquitectura y Base de Datos (SQL)](#arquitectura-y-base-de-datos-sql)
3. [Limpieza y Transformación (Python)](#limpieza-y-transformación-python)
4. [Análisis de Series de Tiempo y Pronósticos (ARIMA)](#análisis-de-series-de-tiempo-y-pronósticos-arima)
5. [Clustering de Estados por Peligrosidad (K-Means)](#clustering-de-estados-por-peligrosidad-k-means)
6. [Visualización Interactiva (Looker Studio)](#visualización-interactiva-looker-studio)
7. [Conclusiones y Recomendaciones de Negocio](#conclusiones-y-recomendaciones-de-negocio)

---

## 🏢 1. Contexto del Negocio
El objetivo principal de este análisis es procesar un conjunto extenso de datos de incidencia delictiva para ayudar a la compañía de seguros a cuantificar el riesgo de robo y siniestros por vehículo a nivel municipal. Esto facilita el diseño de tarifas diferenciadas que mitiguen pérdidas financieras en zonas de alta criminalidad y aumenten la penetración de mercado en zonas de baja incidencia.

---

## 🗄️ 2. Arquitectura y Base de Datos (SQL)
Los datos abiertos crudos se estructuraron e ingirieron en una base de datos relacional para garantizar consultas rápidas y seguras:
* **Estructura:** Tabla central `delitos_municipales` que almacena año, entidad federativa, municipio, tipo/subtipo de delito y número de casos.

---

## 🐍 3. Limpieza y Transformación (Python)
Mediante la librería Pandas se realizaron las siguientes tareas:
* Tratamiento de valores nulos y estandarización de cadenas de texto (nombres de municipios y entidades).
* Filtrado de delitos específicos relacionados con el sector automotor.

---

## 📈 4. Análisis de Series de Tiempo y Pronósticos (ARIMA)
* Se seleccionó un municipio estratégico de alta densidad vehicular.
* Se implementó un modelo de series de tiempo para identificar patrones históricos (2015-2021) y proyectar la tendencia de delitos vehiculares para el año **2022**.

---

## 🤖 5. Clustering de Estados por Peligrosidad (K-Means)
Utilizando técnicas de Machine Learning no supervisado (`Scikit-Learn` y `K-Means`):
* Se normalizaron las variables con `StandardScaler` para evitar sesgos poblacionales.
* Se clasificaron las entidades federativas de México en 2021 en tres niveles de riesgo: *Baja*, *Media* y *Alta Peligrosidad*.

---

## 📊 6. Visualización Interactiva
* Se integraron tableros dinámicos en **Looker Studio** para democratizar el acceso a los KPIs de riesgo y filtros por región.
* [🔗 Enlace al Tablero en Looker Studio](#) *(Insertar enlace público aquí)*

---

## 💡 7. Conclusiones y Recomendaciones
* **Tarifas Dinámicas:** Utilizar los clústeres de riesgo como multiplicadores directos en el cálculo de las primas de pólizas.
* **Suscripción Condicionada:** Establecer topes de aseguramiento o deducibles más altos en municipios catalogados dentro del clúster de alta peligrosidad.

---
*Desarrollado por **Marco Antonio Ruiz** - Analista de Datos*
