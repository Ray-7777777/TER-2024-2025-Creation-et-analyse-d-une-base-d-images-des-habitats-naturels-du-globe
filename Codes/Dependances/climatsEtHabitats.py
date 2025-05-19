import os
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import rasterio
from shapely.geometry import Point
from pyproj import Transformer
import xml.etree.ElementTree as ET

def climats(coordinates, shapefile, species_name, dataset_dir):
    # Construire GeoDataFrame des points en WGS84 (EPSG:4326)
    geometry = [Point(lon, lat) for lat, lon in coordinates]
    geo_df = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    # Reprojeter le shapefile en EPSG:4326
    shp = shapefile.to_crs(geo_df.crs)

    # Intersection spatiale
    intersection = gpd.sjoin(geo_df, shp, how="inner", predicate='intersects')

    # Création du répertoire de l'espèce
    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)

    # Écriture des données climatiques
    out_file = os.path.join(species_dir, 'climate_data.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        for _, row in intersection.iterrows():
            f.write(f"{row['CLIMATE']}, {row['SUB-CLIMAT']}, {row['SUB-SUB-CL']}\n")
    print(f"[INFO] Climat enregistré pour {species_name} -> {out_file}")


def ecoregions(coordinates, shapefile, species_name, dataset_dir):
    # Construire GeoDataFrame des points en WGS84 (EPSG:4326)
    geometry = [Point(lon, lat) for lat, lon in coordinates]
    geo_df = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")

    # Reprojeter le shapefile en EPSG:4326
    shp = shapefile.to_crs(geo_df.crs)

    # Intersection spatiale
    intersection = gpd.sjoin(geo_df, shp, how="inner", predicate='intersects')

    # Création du répertoire de l'espèce
    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)

    # Écriture des données d'écorégions
    out_file = os.path.join(species_dir, 'ecoregions_data.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        for _, row in intersection.iterrows():
            f.write(f"{row['ECO_NAME']}\n")
    print(f"[INFO] Ecorégions enregistrées pour {species_name} -> {out_file}")

def avonet_habitats(avonet_file, sheet_name, species_name, dataset_dir):
    try:
        df = pd.read_excel(avonet_file, sheet_name=sheet_name)
    except Exception as e:
        print(f"[ERROR] Lecture AVONET échouée: {e}")
        return
    species_formatted = species_name.replace('_', ' ')
    row = df[df['Species2']==species_formatted]
    if row.empty:
        print(f"[WARN] Espèce non trouvée dans AVONET: {species_formatted}")
        return
    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)
    out_file = os.path.join(species_dir, 'habitats_data.txt')
    habitats = row['Habitat'].tolist()
    lifestyle = row['Primary.Lifestyle'].tolist()
    with open(out_file, 'w', encoding='utf-8') as f:
        for h in habitats:
            f.write(f"{h}\n")
        for l in lifestyle:
            f.write(f"{l}\n")
    print(f"[INFO] Habitats AVONET enregistrés pour {species_name} -> {out_file}")

def ecosystemes(coordinates, dossier_raster, species_name, dataset_dir, src_epsg='EPSG:3857'):
    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)
    xml_file = os.path.join(os.path.dirname(dossier_raster), 'map-details.xml')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    efg_map = {m.get('efg_code'): m.find('Functional_group').text for m in root.findall('Maps/Map')}
    transformer = Transformer.from_crs(src_epsg, 'EPSG:4326', always_xy=True)
    out_file = os.path.join(species_dir, 'ecosystemes_data.txt')
    with open(out_file, 'w', encoding='utf-8') as f:
        for lat, lon in coordinates:
            info = []
            for fname in os.listdir(dossier_raster):
                code = '.'.join(fname.split('.')[:2])
                path = os.path.join(dossier_raster, fname)
                with rasterio.open(path) as src:
                    x, y = transformer.transform(lon, lat) if src_epsg!='EPSG:4326' else (lon, lat)
                    if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
                        continue
                    row, col = src.index(x, y)
                    val = src.read(1)[row, col]
                    if val in (1,2):
                        occurrence = 'Major' if val==1 else 'Minor'
                        info.append(f"{efg_map.get(code, code)}, {occurrence}")
            if info:
                f.write(f"Coordinates ({lat},{lon}): {', '.join(info)}\n")
    print(f"[INFO] Ecosystèmes enregistrés pour {species_name} -> {out_file}")

def parse_legend(legend_path, band_numbers):
    text = open(legend_path, encoding='latin-1').read()
    mapping = {}
    for part in text.split("Biome Inventory layer ")[1:]:
        hdr, rest = part.split(":",1)
        layer = int(hdr.strip().split()[0])
        if layer in band_numbers:
            val_map = {}
            entries = [e.strip() for e in rest.split(';') if e.strip()]
            for e in entries:
                if ',' in e:
                    v,lbl = e.split(',',1)
                    try:
                        val_map[int(v)] = lbl.strip()
                    except:
                        pass
            mapping[layer] = val_map
    return mapping

def raster_classifications(coordinates,
                           raster_path,
                           bands,
                           band_names,
                           species_name,
                           geo_folder,
                           legend_map=None,
                           src_epsg="EPSG:4326"):
    """
    Extrait pour chaque (lat,lon) les valeurs de `bands` dans `raster_path`,
    et les mappe via legend_map (dict band->{val:label}).
    Produit `raster_classes.txt`.
    """
    species_dir = os.path.join(geo_folder, species_name)
    os.makedirs(species_dir, exist_ok=True)

    # Ouvrir le raster une seule fois (multi-bandes)
    src = rasterio.open(raster_path)
    transformer = Transformer.from_crs(src_epsg, src.crs.to_string(), always_xy=True)
    legend = legend_map or {}

    out_txt = os.path.join(species_dir, 'raster_classes.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write('Latitude,Longitude,' + ','.join(band_names) + "\n")
        for lat, lon in coordinates:
            # reprojeter
            x, y = transformer.transform(lon, lat)
            try:
                row, col = src.index(x, y)
            except Exception:
                vals = [''] * len(bands)
            else:
                vals = []
                for b in bands:
                    raw = int(src.read(b)[row, col])
                    if b in legend:
                        vals.append(str(legend[b].get(raw, raw)))
                    else:
                        vals.append(str(raw))
            f.write(f"{lat},{lon}," + ",".join(vals) + "\n")

    src.close()
    print(f"[INFO] Classes raster enregistrées pour {species_name} -> {out_txt}")
    return out_txt

def carte(dataset_dir):
    for species_name in os.listdir(dataset_dir):
        species_dir = os.path.join(dataset_dir, species_name)
        coords_file = os.path.join(species_dir, 'coordinates.txt')
        m = folium.Map(location=[0,0], zoom_start=2)
        if os.path.isfile(coords_file):
            with open(coords_file) as f:
                for line in f:
                    if line.startswith('Coordinates'):
                        lat,lon = map(float, line.split(':')[1].split(','))
                        folium.Marker([lat,lon], popup=species_name).add_to(m)
        out = os.path.join(species_dir, 'carte_oiseaux.html')
        m.save(out)
        print(f"[INFO] Carte espèce {species_name} -> {out}")

def carte_globale(all_coords, output_path):
    """
    all_coords: dict espèce -> list[(lat,lon)]
    crée une carte unique avec tous les points.
    """
    m = folium.Map(location=[0,0], zoom_start=2)
    for sp, coords in all_coords.items():
        for lat, lon in coords:
            folium.Marker([lat, lon], popup=sp,
                          icon=folium.Icon(color='green')).add_to(m)
    m.save(output_path)
    print(f"[INFO] Carte globale -> {output_path}")
