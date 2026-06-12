import io
import json
import math
import os
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
import streamlit as st
from affine import Affine
from rasterio.enums import Resampling
from shapely.geometry import LineString, MultiLineString, Polygon, Point, GeometryCollection, box
from shapely.ops import transform as shp_transform
from skimage import measure

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

APP_TITLE = "Curvas de nivel COP30 para cuencas grandes"
OPENTOPO_URL = "https://portal.opentopography.org/API/globaldem"

DEM_ARCSEC = {
    "COP30": 1.0,
    "NASADEM": 1.0,
    "SRTMGL1": 1.0,
    "SRTMGL3": 3.0,
    "COP90": 3.0,
    "AW3D30": 1.0,
}

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}

RECOMENDACIONES_CUENCAS_GRANDES = """
Parámetros recomendados para cuencas grandes:

- Número de DEM parciales: 10 a 40
- Resolución interna: 150 m o 300 m
- Equidistancia curvas: 50 m o 100 m
- Simplificación: 60 m o más
"""


st.set_page_config(page_title=APP_TITLE, layout="wide")

# -----------------------------------------------------------------------------
# Utilidades geográficas
# -----------------------------------------------------------------------------

def km_per_degree_lon(lat: float) -> float:
    return max(1e-6, 111.320 * math.cos(math.radians(lat)))


def bbox_area_km2(south: float, north: float, west: float, east: float) -> float:
    midlat = (south + north) / 2
    return abs((north - south) * 111.320) * abs((east - west) * km_per_degree_lon(midlat))


def bbox_from_point_area(lat: float, lon: float, area_km2: float, aspect_ew_ns: float = 1.20):
    """Crea un bbox rectangular aproximado alrededor de un punto.
    aspect_ew_ns > 1 alarga el bbox este-oeste.
    """
    height_km = math.sqrt(area_km2 / aspect_ew_ns)
    width_km = height_km * aspect_ew_ns
    dlat = height_km / 111.320 / 2
    dlon = width_km / km_per_degree_lon(lat) / 2
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def bbox_from_point_radius(lat: float, lon: float, radius_km: float):
    dlat = radius_km / 111.320
    dlon = radius_km / km_per_degree_lon(lat)
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon


def buffer_bbox_km(bounds, buffer_km: float):
    west, south, east, north = bounds
    midlat = (south + north) / 2
    dlat = buffer_km / 111.320
    dlon = buffer_km / km_per_degree_lon(midlat)
    return south - dlat, north + dlat, west - dlon, east + dlon


def split_bbox(south, north, west, east, n_tiles: int):
    # Elige filas/columnas equilibradas según proporción geográfica del bbox
    area_lat = abs(north - south) * 111.320
    area_lon = abs(east - west) * km_per_degree_lon((south + north) / 2)
    aspect = area_lon / max(area_lat, 1e-9)
    best = None
    for rows in range(1, n_tiles + 1):
        cols = math.ceil(n_tiles / rows)
        score = abs((cols / rows) - aspect) + 0.2 * abs(rows * cols - n_tiles)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    rows, cols = best[1], best[2]
    tiles = []
    for r in range(rows):
        s = south + (north - south) * r / rows
        n = south + (north - south) * (r + 1) / rows
        for c in range(cols):
            if len(tiles) >= n_tiles:
                break
            w = west + (east - west) * c / cols
            e = west + (east - west) * (c + 1) / cols
            tiles.append({"tile": len(tiles) + 1, "south": s, "north": n, "west": w, "east": e})
    return tiles, rows, cols


def estimate_cells_from_bbox(south, north, west, east, demtype="COP30"):
    arcsec = DEM_ARCSEC.get(demtype, 1.0)
    px_per_deg = 3600.0 / arcsec
    rows = int(abs(north - south) * px_per_deg)
    cols = int(abs(east - west) * px_per_deg)
    return rows, cols, rows * cols


def recommended_processing_resolution(area_km2: float):
    if area_km2 <= 5_000:
        return 30
    if area_km2 <= 25_000:
        return 90
    if area_km2 <= 100_000:
        return 150
    return 300

# -----------------------------------------------------------------------------
# Lectura KMZ/KML
# -----------------------------------------------------------------------------

def read_kml_text(uploaded) -> str:
    data = uploaded.read()
    uploaded.seek(0)
    name = uploaded.name.lower()
    if name.endswith(".kmz"):
        with zipfile.ZipFile(io.BytesIO(data), "r") as z:
            kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
            if not kml_names:
                raise ValueError("El KMZ no contiene archivo .kml")
            return z.read(kml_names[0]).decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")


def parse_coord_text(text: str):
    coords = []
    for item in text.replace("\n", " ").replace("\t", " ").split():
        parts = item.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                alt = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
                coords.append((lon, lat, alt))
            except Exception:
                continue
    return coords


def parse_kml_geometries(uploaded):
    kml_text = read_kml_text(uploaded)
    root = ET.fromstring(kml_text.encode("utf-8"))
    points = []
    polygons = []

    # Puntos explícitos
    for pt in root.findall(".//kml:Point", KML_NS):
        coord_el = pt.find("kml:coordinates", KML_NS)
        if coord_el is not None and coord_el.text:
            coords = parse_coord_text(coord_el.text)
            if coords:
                lon, lat, _ = coords[0]
                points.append((lon, lat))

    # Polígonos explícitos
    for poly_el in root.findall(".//kml:Polygon", KML_NS):
        coord_el = poly_el.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", KML_NS)
        if coord_el is not None and coord_el.text:
            coords = parse_coord_text(coord_el.text)
            if len(coords) >= 4:
                ring = [(lon, lat) for lon, lat, _ in coords]
                try:
                    poly = Polygon(ring)
                    if poly.is_valid and not poly.is_empty:
                        polygons.append(poly)
                except Exception:
                    pass

    # Si no hay Point, intenta usar la primera coordenada genérica como punto
    if not points:
        coord_elements = root.findall(".//kml:coordinates", KML_NS)
        for el in coord_elements:
            coords = parse_coord_text(el.text or "")
            if coords:
                lon, lat, _ = coords[0]
                points.append((lon, lat))
                break

    return points, polygons

# -----------------------------------------------------------------------------
# OpenTopography y procesamiento de contornos
# -----------------------------------------------------------------------------

def build_url(demtype, south, north, west, east, api_key):
    params = {
        "demtype": demtype,
        "south": f"{south:.8f}",
        "north": f"{north:.8f}",
        "west": f"{west:.8f}",
        "east": f"{east:.8f}",
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    q = "&".join([f"{k}={v}" for k, v in params.items()])
    return f"{OPENTOPO_URL}?{q}"


def mask_key(url: str, api_key: str):
    if not api_key:
        return url
    return url.replace(api_key, "***API_KEY_OCULTA***")


def download_tile(url, out_path, timeout=600):
    with requests.get(url, stream=True, timeout=timeout) as r:
        if r.status_code != 200:
            text = r.text[:500] if hasattr(r, "text") else ""
            raise RuntimeError(f"OpenTopography respondió {r.status_code}: {text}")
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    if os.path.getsize(out_path) < 10_000:
        raise RuntimeError("El archivo descargado es demasiado pequeño; revise bbox/API Key/demtype.")
    return os.path.getsize(out_path)


def extract_lines_from_geometry(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        lines = []
        for g in geom.geoms:
            lines.extend(extract_lines_from_geometry(g))
        return lines
    return []


def decimate_line(line: LineString, max_vertices: int):
    coords = list(line.coords)
    if len(coords) <= max_vertices:
        return line
    step = max(1, math.ceil(len(coords) / max_vertices))
    sampled = coords[::step]
    if sampled[-1] != coords[-1]:
        sampled.append(coords[-1])
    if len(sampled) < 2:
        return line
    return LineString(sampled)


def reproject_line_if_needed(line, src_crs):
    if src_crs is None or str(src_crs).upper().endswith("4326"):
        return line
    if Transformer is None:
        return line
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    return shp_transform(lambda x, y, z=None: transformer.transform(x, y), line)


def process_contours_for_tile(
    tif_path,
    interval_m: float,
    target_resolution_m: float,
    simplify_m: float,
    clip_polygon: Polygon | None,
    max_levels: int,
    max_vertices_per_line: int,
    min_line_points: int = 10,
):
    results = []
    with rasterio.open(tif_path) as src:
        # Estimación de resolución horizontal aproximada en metros
        midlat = (src.bounds.top + src.bounds.bottom) / 2
        res_x_m = abs(src.transform.a) * km_per_degree_lon(midlat) * 1000 if src.crs and src.crs.is_geographic else abs(src.transform.a)
        res_y_m = abs(src.transform.e) * 111_320 if src.crs and src.crs.is_geographic else abs(src.transform.e)
        native_res_m = max(1.0, (res_x_m + res_y_m) / 2)

        scale = max(1.0, target_resolution_m / native_res_m)
        out_height = max(10, int(src.height / scale))
        out_width = max(10, int(src.width / scale))

        data = src.read(
            1,
            out_shape=(out_height, out_width),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype("float32")

        # Transformación ajustada al tamaño remuestreado
        transform = src.transform * Affine.scale(src.width / out_width, src.height / out_height)
        arr = np.ma.filled(data, np.nan)
        valid = np.isfinite(arr)
        if not valid.any():
            return results, {"native_res_m": native_res_m, "out_width": out_width, "out_height": out_height, "levels": 0}

        zmin = float(np.nanmin(arr))
        zmax = float(np.nanmax(arr))
        if interval_m <= 0:
            raise ValueError("La equidistancia debe ser mayor que cero.")
        first = math.ceil(zmin / interval_m) * interval_m
        levels = np.arange(first, zmax + interval_m, interval_m, dtype=float)
        if len(levels) > max_levels:
            step = math.ceil(len(levels) / max_levels)
            levels = levels[::step]

        # Evita contornos falsos por nodata; normalmente OpenTopography viene completo
        arr2 = np.where(valid, arr, np.nanmedian(arr[valid]))
        simplify_deg = max(0.0, simplify_m / 111_320.0)

        for level in levels:
            try:
                contours = measure.find_contours(arr2, level=level)
            except Exception:
                continue
            for contour in contours:
                if contour.shape[0] < min_line_points:
                    continue
                rows = contour[:, 0]
                cols = contour[:, 1]
                xs = transform.c + cols * transform.a + rows * transform.b
                ys = transform.f + cols * transform.d + rows * transform.e
                coords = list(zip(xs, ys))
                try:
                    line = LineString(coords)
                except Exception:
                    continue
                if line.is_empty or line.length == 0:
                    continue
                line = reproject_line_if_needed(line, src.crs)
                if simplify_deg > 0:
                    line = line.simplify(simplify_deg, preserve_topology=False)
                if max_vertices_per_line > 0:
                    line = decimate_line(line, max_vertices_per_line)

                if clip_polygon is not None:
                    try:
                        inter = line.intersection(clip_polygon)
                    except Exception:
                        continue
                    lines = extract_lines_from_geometry(inter)
                else:
                    lines = [line]

                for ln in lines:
                    if ln.is_empty or len(ln.coords) < 2:
                        continue
                    results.append({"elev": float(level), "line": ln})

    meta = {
        "native_res_m": native_res_m,
        "out_width": out_width,
        "out_height": out_height,
        "levels": len(levels),
    }
    return results, meta

# -----------------------------------------------------------------------------
# Escritura KML/KMZ por streaming
# -----------------------------------------------------------------------------

def kml_header(title: str):
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
<name>{escape(title)}</name>
<Style id="contour_normal"><LineStyle><color>ff7f7f7f</color><width>1.2</width></LineStyle></Style>
<Style id="contour_index"><LineStyle><color>ff0000ff</color><width>2.4</width></LineStyle></Style>
<Style id="bbox_style"><LineStyle><color>ff00ffff</color><width>1.5</width></LineStyle><PolyStyle><color>2200ffff</color></PolyStyle></Style>
<Style id="poly_style"><LineStyle><color>ff00aa00</color><width>2.0</width></LineStyle><PolyStyle><color>2200ff00</color></PolyStyle></Style>
<Style id="point_style"><IconStyle><scale>1.1</scale><Icon><href>http://maps.google.com/mapfiles/kml/pushpin/ylw-pushpin.png</href></Icon></IconStyle></Style>
'''


def kml_footer():
    return "</Document>\n</kml>\n"


def coords_to_kml(coords):
    return " ".join([f"{x:.8f},{y:.8f},0" for x, y in coords])


def write_point_kml(f, lon, lat, name):
    f.write(f"""
<Placemark><name>{escape(name)}</name><styleUrl>#point_style</styleUrl><Point><coordinates>{lon:.8f},{lat:.8f},0</coordinates></Point></Placemark>
""")


def write_polygon_kml(f, polygon: Polygon, name: str, style="#poly_style"):
    if polygon is None or polygon.is_empty:
        return
    exterior = list(polygon.exterior.coords)
    f.write(f"""
<Placemark><name>{escape(name)}</name><styleUrl>{style}</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>{coords_to_kml(exterior)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
""")


def write_line_kml(f, line: LineString, elev: float, interval_m: float):
    if line.is_empty:
        return
    coords = list(line.coords)
    if len(coords) < 2:
        return
    # Curva índice cada 5 intervalos
    is_index = False
    try:
        is_index = abs((elev / interval_m) % 5) < 1e-6
    except Exception:
        pass
    style = "#contour_index" if is_index else "#contour_normal"
    f.write(f"""
<Placemark><name>Curva {elev:.0f} m</name><styleUrl>{style}</styleUrl><ExtendedData><Data name="elev_m"><value>{elev:.2f}</value></Data></ExtendedData><LineString><tessellate>1</tessellate><coordinates>{coords_to_kml(coords)}</coordinates></LineString></Placemark>
""")


def create_kmz(kml_path: Path, kmz_path: Path):
    with zipfile.ZipFile(kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(kml_path, arcname="doc.kml")

# -----------------------------------------------------------------------------
# Interfaz Streamlit
# -----------------------------------------------------------------------------

st.title(APP_TITLE)
st.caption("Aplicación para descargar DEMs por partes desde OpenTopography, generar curvas de nivel por mosaicos y exportar un KMZ unificado.")

with st.expander("Criterio técnico importante", expanded=True):
    st.write(
        "Para áreas muy grandes, por ejemplo 200.000 km², no conviene procesar todo el COP30 a 30 m en una sola matriz. "
        "La app divide el área en varios DEM parciales, procesa cada tile por separado y escribe las curvas en un único KMZ. "
        "Para que Streamlit Cloud no se caiga, se recomienda remuestrear internamente a 150 m o 300 m cuando el área es grande."
    )
    st.info(RECOMENDACIONES_CUENCAS_GRANDES)

col_a, col_b = st.columns([0.32, 0.68])

with col_a:
    st.subheader("1. Entrada")
    uploaded = st.file_uploader("KMZ/KML con punto o polígono de cuenca", type=["kmz", "kml"])
    api_key = st.text_input("API Key OpenTopography", type="password")
    demtype = st.selectbox("DEM OpenTopography", ["COP30", "NASADEM", "SRTMGL1", "SRTMGL3", "COP90", "AW3D30"], index=0)

    st.subheader("2. Área de trabajo")
    extent_mode = st.radio(
        "Cómo definir el área",
        ["Desde polígono KMZ/KML", "Desde punto + área objetivo", "Desde punto + radio", "Manual bbox"],
        index=1,
    )
    area_obj = st.number_input("Área objetivo aproximada km²", min_value=100.0, max_value=450000.0, value=200000.0, step=1000.0)
    radius_km = st.number_input("Radio desde el punto km", min_value=5.0, max_value=500.0, value=120.0, step=5.0)
    buffer_km = st.number_input("Buffer adicional para polígono km", min_value=0.0, max_value=100.0, value=10.0, step=5.0)

    st.subheader("3. División y curvas")
    n_tiles = st.number_input("Número de DEM parciales", min_value=2, max_value=80, value=10, step=1)
    interval_m = st.number_input("Equidistancia curvas de nivel m", min_value=1.0, max_value=500.0, value=50.0, step=5.0)
    target_resolution_m = st.selectbox("Resolución interna para procesar curvas", [30, 60, 90, 150, 300], index=3)
    simplify_m = st.number_input("Suavizado/simplificación de líneas m", min_value=0.0, max_value=500.0, value=60.0, step=10.0)
    max_levels = st.number_input("Máximo de cotas por tile", min_value=20, max_value=1000, value=250, step=10)
    max_vertices_per_line = st.number_input("Máximo vértices por curva", min_value=50, max_value=20000, value=2500, step=100)
    max_total_lines = st.number_input("Máximo total de curvas/segmentos", min_value=1000, max_value=500000, value=80000, step=1000)

    with st.expander("Parámetros recomendados para cuencas grandes", expanded=False):
        st.markdown(RECOMENDACIONES_CUENCAS_GRANDES)

    st.subheader("4. Bbox manual")
    man_s = st.number_input("south", value=-31.20, format="%.8f")
    man_n = st.number_input("north", value=-30.55, format="%.8f")
    man_w = st.number_input("west", value=-71.40, format="%.8f")
    man_e = st.number_input("east", value=-70.70, format="%.8f")

points, polygons = [], []
parse_error = None
if uploaded:
    try:
        points, polygons = parse_kml_geometries(uploaded)
    except Exception as e:
        parse_error = str(e)

# Define bbox
bbox = None
main_point = None
clip_polygon = None
if polygons:
    # Usa el polígono de mayor área angular aproximada
    clip_polygon = max(polygons, key=lambda p: p.area)
if points:
    lon0, lat0 = points[0]
    main_point = (lon0, lat0)

if extent_mode == "Manual bbox":
    bbox = (man_s, man_n, man_w, man_e)
elif extent_mode == "Desde polígono KMZ/KML" and clip_polygon is not None:
    bbox = buffer_bbox_km(clip_polygon.bounds, buffer_km)
elif main_point is not None:
    lon0, lat0 = main_point
    if extent_mode == "Desde punto + radio":
        bbox = bbox_from_point_radius(lat0, lon0, radius_km)
    else:
        bbox = bbox_from_point_area(lat0, lon0, area_obj)

with col_b:
    st.subheader("Diagnóstico del área")
    if parse_error:
        st.error(f"No se pudo leer el KMZ/KML: {parse_error}")
    if uploaded and not parse_error:
        st.success(f"Archivo leído. Puntos: {len(points)} | Polígonos: {len(polygons)}")
        if main_point:
            st.write(f"Punto detectado: lat {main_point[1]:.8f}, lon {main_point[0]:.8f}")
        if clip_polygon is not None:
            st.write(f"Polígono detectado. Bounds: {clip_polygon.bounds}")

    if bbox is None:
        st.warning("Sube un KMZ/KML con punto/polígono o usa bbox manual.")
    else:
        south, north, west, east = bbox
        area_est = bbox_area_km2(south, north, west, east)
        rows, cols, cells = estimate_cells_from_bbox(south, north, west, east, demtype)
        rec_res = recommended_processing_resolution(area_est)
        tiles, grid_rows, grid_cols = split_bbox(south, north, west, east, int(n_tiles))
        max_tile_cells = max(estimate_cells_from_bbox(t["south"], t["north"], t["west"], t["east"], demtype)[2] for t in tiles)
        proc_factor = max(1.0, target_resolution_m / 30.0)
        est_processed_cells_per_tile = int(max_tile_cells / (proc_factor * proc_factor))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Área bbox aprox.", f"{area_est:,.0f} km²")
        m2.metric("Celdas DEM nativo", f"{cells/1e6:,.1f} M")
        m3.metric("DEM parciales", f"{len(tiles)}")
        m4.metric("Celdas/tile procesadas", f"{est_processed_cells_per_tile/1e6:,.1f} M")

        st.code(
            f"south={south:.8f}\nnorth={north:.8f}\nwest ={west:.8f}\neast ={east:.8f}",
            language="text",
        )
        if area_est > 100000 and target_resolution_m < 150:
            st.warning(f"Área grande detectada. Se recomienda resolución interna ≥ {rec_res} m para evitar caída por memoria.")
        if est_processed_cells_per_tile > 5_000_000:
            st.error("Cada tile sigue siendo muy pesado. Aumenta número de DEM parciales o sube la resolución interna a 150/300 m.")
        elif est_processed_cells_per_tile > 2_000_000:
            st.warning("Carga media/alta. Puede funcionar, pero conviene intervalos de curva mayores y simplificación.")
        else:
            st.success("Configuración razonable para procesamiento por partes.")

        df_tiles = pd.DataFrame(tiles)
        df_tiles["area_km2"] = df_tiles.apply(lambda r: bbox_area_km2(r.south, r.north, r.west, r.east), axis=1)
        df_tiles["cells_native_M"] = df_tiles.apply(lambda r: estimate_cells_from_bbox(r.south, r.north, r.west, r.east, demtype)[2] / 1e6, axis=1)
        st.dataframe(df_tiles, use_container_width=True, height=260)

        first_url = build_url(demtype, tiles[0]["south"], tiles[0]["north"], tiles[0]["west"], tiles[0]["east"], api_key or "TU_API_KEY")
        st.write("Ejemplo URL tile 1:")
        st.code(mask_key(first_url, api_key), language="text")

run = st.button("Generar KMZ unificado de curvas", type="primary", disabled=(bbox is None))

if run:
    if not api_key:
        st.error("Debes ingresar API Key de OpenTopography.")
        st.stop()
    south, north, west, east = bbox
    if not (south < north and west < east):
        st.error("Bbox inválido: verifica south/north/west/east.")
        st.stop()

    tiles, grid_rows, grid_cols = split_bbox(south, north, west, east, int(n_tiles))
    total_lines = 0
    log_rows = []
    start = time.time()
    progress = st.progress(0)
    status = st.empty()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        kml_path = tmpdir / "doc.kml"
        kmz_path = tmpdir / "curvas_cuenca_grande_unificado.kmz"
        summary_path = tmpdir / "resumen_curvas_cuenca_grande.json"

        with open(kml_path, "w", encoding="utf-8") as kml:
            kml.write(kml_header("Curvas de nivel unificadas - cuenca grande"))

            if main_point:
                write_point_kml(kml, main_point[0], main_point[1], "Punto de control")
            if clip_polygon is not None:
                write_polygon_kml(kml, clip_polygon, "Polígono de cuenca ingresado", "#poly_style")
            write_polygon_kml(kml, box(west, south, east, north), "Bbox general de descarga", "#bbox_style")

            for idx, t in enumerate(tiles, start=1):
                if total_lines >= max_total_lines:
                    status.warning("Se alcanzó el máximo total de curvas configurado. Se detuvo para evitar caída por memoria/KMZ excesivo.")
                    break

                status.info(f"Tile {idx}/{len(tiles)}: descargando DEM {demtype}...")
                url = build_url(demtype, t["south"], t["north"], t["west"], t["east"], api_key)
                tif_path = tmpdir / f"tile_{idx:02d}_{demtype}.tif"
                tile_status = {"tile": idx, **t}
                try:
                    size_bytes = download_tile(url, tif_path)
                    tile_status["download_MB"] = round(size_bytes / (1024 * 1024), 2)
                except Exception as e:
                    tile_status["error"] = f"Descarga: {e}"
                    log_rows.append(tile_status)
                    st.error(f"Tile {idx}: {e}")
                    progress.progress(idx / len(tiles))
                    continue

                status.info(f"Tile {idx}/{len(tiles)}: generando curvas...")
                try:
                    lines, meta = process_contours_for_tile(
                        tif_path=tif_path,
                        interval_m=float(interval_m),
                        target_resolution_m=float(target_resolution_m),
                        simplify_m=float(simplify_m),
                        clip_polygon=clip_polygon,
                        max_levels=int(max_levels),
                        max_vertices_per_line=int(max_vertices_per_line),
                    )
                    written = 0
                    for obj in lines:
                        if total_lines >= max_total_lines:
                            break
                        write_line_kml(kml, obj["line"], obj["elev"], float(interval_m))
                        written += 1
                        total_lines += 1
                    tile_status.update(meta)
                    tile_status["curves_written"] = written
                except Exception as e:
                    tile_status["error"] = f"Curvas: {e}"
                    st.error(f"Tile {idx}: {e}")
                finally:
                    try:
                        tif_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    log_rows.append(tile_status)
                    progress.progress(idx / len(tiles))

            kml.write(kml_footer())

        create_kmz(kml_path, kmz_path)
        elapsed = time.time() - start
        summary = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "demtype": demtype,
            "bbox": {"south": south, "north": north, "west": west, "east": east},
            "bbox_area_km2": bbox_area_km2(south, north, west, east),
            "n_tiles_requested": int(n_tiles),
            "n_tiles_processed": len(log_rows),
            "interval_m": float(interval_m),
            "target_resolution_m": float(target_resolution_m),
            "simplify_m": float(simplify_m),
            "total_contour_segments": int(total_lines),
            "elapsed_seconds": round(elapsed, 2),
            "tiles": log_rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        status.success(f"Proceso terminado. Curvas/segmentos escritos: {total_lines:,}. Tiempo: {elapsed/60:.1f} min.")
        st.subheader("Descargas")
        with open(kmz_path, "rb") as f:
            st.download_button("Descargar KMZ unificado", f, file_name="curvas_cuenca_grande_unificado.kmz", mime="application/vnd.google-earth.kmz")
        with open(summary_path, "rb") as f:
            st.download_button("Descargar resumen JSON", f, file_name="resumen_curvas_cuenca_grande.json", mime="application/json")

        st.subheader("Resumen de tiles")
        st.dataframe(pd.DataFrame(log_rows), use_container_width=True)

st.divider()
st.markdown(
    "**Nota:** el KMZ unificado contiene todos los segmentos generados por tile en un solo archivo. "
    "Para áreas muy grandes, la unión topológica perfecta de líneas entre bordes de tiles puede requerir un postproceso GIS local; "
    "esta app prioriza estabilidad, descarga por partes y visualización unificada en Google Earth."
)
