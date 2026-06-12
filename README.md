# Curvas de nivel COP30 para cuencas grandes

Aplicación Streamlit para descargar DEMs por partes desde OpenTopography, generar curvas de nivel por mosaicos y exportar un KMZ unificado.

## Objetivo

Trabajar con áreas grandes, por ejemplo 200.000 km², evitando procesar un DEM completo en una sola matriz. La aplicación divide el área de trabajo en varios DEM parciales, procesa cada tile en forma secuencial y escribe las curvas en un único KMZ.

## Entrada

- KMZ/KML con punto de control o polígono de cuenca.
- API Key de OpenTopography.
- DEM: COP30 por defecto. Alternativas: NASADEM, SRTMGL1, SRTMGL3, COP90, AW3D30.
- Área objetivo, radio o bbox manual.
- Número de DEM parciales, por defecto 10.
- Equidistancia de curvas de nivel.
- Resolución interna de procesamiento.

## Salida

- `curvas_cuenca_grande_unificado.kmz`
- `resumen_curvas_cuenca_grande.json`

## Main file path para Streamlit Cloud

```text
app.py
```

## Versión de Python recomendada

Usar Python 3.11 en Streamlit Cloud.

## Recomendaciones para áreas grandes

Para COP30 a 30 m, un área de 200.000 km² puede superar los 200 millones de celdas nativas. Por eso la app descarga por partes y remuestrea internamente. Para áreas grandes se recomienda:

- 10 a 40 DEM parciales.
- Resolución interna 150 m o 300 m.
- Equidistancia de curvas 50 m, 100 m o mayor.
- Simplificación de líneas 60 m o mayor.

El KMZ unificado contiene todos los segmentos de curva generados por tile. Para una unión topológica perfecta entre bordes de tiles puede requerirse postproceso GIS local.

## Seguridad API Key

La API Key se ingresa en campo password y no se guarda en el repositorio. No suba claves a GitHub.
