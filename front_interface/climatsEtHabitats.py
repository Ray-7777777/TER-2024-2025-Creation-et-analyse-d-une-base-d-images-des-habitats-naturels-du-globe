import os
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio
from pyproj import Transformer
import xml.etree.ElementTree as ET

def climats(coordinates, shapefile, species_name, dataset_dir):
    geometry = [Point(lon, lat) for lat, lon in coordinates]  # Créer une géométrie Point pour chaque coordonnée
    geo_df = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4004")  # Assure-toi que le CRS correspond

    # Faire l'intersection avec le shapefile (qui peut contenir des polygones, par exemple)
    intersection = gpd.sjoin(geo_df, shapefile, how="inner", predicate='intersects')

    # Récupérer les valeurs associées à chaque point
    species_dir = os.path.join(dataset_dir, species_name)
    with open(f'{species_dir}/climate_data.txt', 'w') as file:
        for idx, row in intersection.iterrows():
            lat, lon = row.geometry.y, row.geometry.x
            # Suppose que 'attribut' est une colonne de la table attributaire du shapefile
            climate = row['CLIMATE']
            subclimate = row['SUB-CLIMAT']
            subsubclimate = row['SUB-SUB-CL']
            file.write(f"{climate}, {subclimate}, {subsubclimate}\n")
    
    coordinates_file = os.path.join(species_dir, 'coordinates.txt')
    with open(coordinates_file, 'w') as file:
        for lat, lon in coordinates:
            file.write(f"Coordinates: {lat}, {lon}\n")


def ecoregions(coordinates, shapefile, species_name, dataset_dir):
    geometry = [Point(lon, lat) for lat, lon in coordinates]  # Créer une géométrie Point pour chaque coordonnée
    geo_df = gpd.GeoDataFrame(geometry=geometry, crs="EPSG:4326")  # Assure-toi que le CRS correspond

    # Faire l'intersection avec le shapefile (qui peut contenir des polygones, par exemple)
    intersection = gpd.sjoin(geo_df, shapefile, how="inner", predicate='intersects')

    # Récupérer les valeurs associées à chaque point
    species_dir = os.path.join(dataset_dir, species_name)
    with open(f'{species_dir}/ecoregions_data.txt', 'w') as file:
        for idx, row in intersection.iterrows():
            lat, lon = row.geometry.y, row.geometry.x
            # Suppose que 'attribut' est une colonne de la table attributaire du shapefile
            ecoregion = row['ECO_NAME']
            file.write(f"{ecoregion}\n")

def avonet_habitats(avonet_file, sheet_name, species_name, dataset_dir):
    try:
        df = pd.read_excel(avonet_file, sheet_name=sheet_name)
    except FileNotFoundError:
        print(f"Erreur : Le fichier {avonet_file} n'a pas été trouvé.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Erreur : Le fichier {avonet_file} est vide.")
        return None
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier {avonet_file} : {e}")
        return None
    
    # Rechercher la ligne où le nom de l'espèce correspond à `species_name`
    species_name_formatted = species_name.replace('_', ' ')
    species_row = df[df['Species2'] == species_name_formatted]
    
    if species_row.empty:
        print(f"Espèce '{species_name}' non trouvée dans le fichier.")
        return None
    
    # Créer le répertoire pour l'espèce si nécessaire
    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)
    
    # Extraire les données des habitats pour cette espèce
    habitats = species_row['Habitat'].values
    lifestyle = species_row['Primary.Lifestyle'].values
    
    if len(habitats) == 0:
        print(f"Aucun habitat trouvé pour l'espèce '{species_name}'.")
        return None
    
    # Enregistrer les données dans un fichier texte
    with open(f'{species_dir}/habitats_data.txt', 'w') as file:
        for habitat in habitats:
            file.write(f"{habitat}\n")
        
        for life in lifestyle:
            file.write(f"{life}\n")


def ecosystemes(coordinates, dossier_raster, species_name, dataset_dir, src_epsg='epsg:3857'):
    species_dir = os.path.join(dataset_dir, species_name)
    results = {}

    # Définir le chemin du fichier XML
    xml_file = os.path.join(dossier_raster, '..', 'map-details.xml')  # Construction du chemin relatif pour le fichier XML

    # Charger le fichier XML contenant les descriptions
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Créer un dictionnaire pour stocker les descriptions associées aux codes efg_code
    efg_map = {}

    # Parcourir le fichier XML et remplir le dictionnaire
    for map_element in root.findall('Maps/Map'):  # Chemin des informations dans le XML
        efg_code = map_element.get('efg_code')  # Extraire le code efg_code
        description = map_element.find('Functional_group').text  # Extraire la description (Functional_group)
        efg_map[efg_code] = description

    # Initialisation du dictionnaire pour stocker les résultats de tous les points
    for coords in coordinates:
        results[coords] = {
            "inside_rasters": []  # Liste des groupes fonctionnels et types d'occurrence associés
        }

    # Définir le système de projection source (par défaut en EPSG:3857) et cible (EPSG:4326)
    transformer = Transformer.from_crs(src_epsg, 'epsg:4326', always_xy=True)

    # Parcourir tous les fichiers du dossier
    for fichier in os.listdir(dossier_raster):
        raster_file = os.path.join(dossier_raster, fichier)

        # Extraire le code efg_code du nom du fichier raster
        efg_code = '.'.join(fichier.split('.')[:2]) 

        # Ouvrir le fichier raster
        with rasterio.open(raster_file) as src:
            # Pour chaque coordonnée, créer un Point et tester s'il est à l'intérieur du raster
            for coords in coordinates:
                # Transformer les coordonnées (si nécessaire) en EPSG:4326 (latitude, longitude)
                if src_epsg != 'epsg:4326':
                    lon, lat = transformer.transform(coords[0], coords[1])
                else:
                    lat, lon = coords 

                point = Point(lon, lat) 

                # Vérifier si le point est dans l'étendue du raster
                if not src.bounds[0] <= point.x <= src.bounds[2] or not src.bounds[1] <= point.y <= src.bounds[3]:
                    continue  # Si le point est en dehors des limites du raster, on passe au suivant

                # Convertir les coordonnées géographiques en coordonnées du raster (ligne, colonne)
                row, col = src.index(point.x, point.y)

                # Vérifier la valeur du pixel dans le raster
                value = src.read(1)[row, col]
                
                # Si la valeur est 1 (Major occurrence) ou 2 (Minor occurrence), on les ajoute à la liste
                if value == 1 or value == 2:
                    occurrence_type = "Major occurrence" if value == 1 else "Minor occurrence"
                    results[coords]["inside_rasters"].append(f"{description}, {occurrence_type}")

    # Enregistrer les résultats dans un fichier texte
    output_file = os.path.join(species_dir, 'ecosystemes_data.txt')
    with open(output_file, 'w') as file:
        # Pour chaque paire de coordonnées, concaténer toutes les descriptions des groupes fonctionnels
        for coords, data in results.items():
            if data["inside_rasters"]:
                # Concaténer les informations de tous les rasters pour cette coordonnée
                line = f"Coordinates {coords}: " + ", ".join(data["inside_rasters"])
                file.write(line + "\n")

def parse_legend(legend_path, band_numbers):
    text = open(legend_path, encoding="latin-1").read()
    mapping = {}
    for part in text.split("Biome Inventory layer ")[1:]:
        hdr, rest = part.split(":", 1)
        layer = int(hdr.strip().split()[0])
        if layer in band_numbers:
            entries = [e.strip() for e in rest.split(";") if e.strip()]
            val_map = {}
            for e in entries:
                if "," in e:
                    v, lbl = e.split(",", 1)
                    try:
                        val_map[int(v.strip())] = lbl.strip()
                    except:
                        pass
            mapping[layer] = val_map
    return mapping

def raster_classifications(coordinates, raster_path, bands, band_names,
                           species_name, geo_folder, legend_map=None,
                           src_epsg="epsg:4326"):
    transformer = Transformer.from_crs(src_epsg, "epsg:4326", always_xy=True)
    results = {}
    species_dir = os.path.join(geo_folder, species_name)
    os.makedirs(species_dir, exist_ok=True)

    with rasterio.open(raster_path) as src:
        for lat, lon in coordinates:
            row, col = src.index(lon, lat)
            vals = {}
            for b_idx, name in zip(bands, band_names):
                try:
                    raw = int(src.read(b_idx)[row, col])
                except:
                    raw = None
                if legend_map and b_idx in legend_map:
                    vals[name] = legend_map[b_idx].get(raw)
                else:
                    vals[name] = raw
            results[(lat, lon)] = vals

    out_txt = os.path.join(species_dir, "raster_classes.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Latitude,Longitude," + ",".join(band_names) + "\n")
        for (lat, lon), vals in results.items():
            line = ",".join(str(vals[n]) if vals[n] is not None else "" for n in band_names)
            f.write(f"{lat},{lon},{line}\n")

    return results

def compute_intersections(all_coords, shapefile_clim, shapefile_eco):
    # construit un seul GeoDataFrame pour toutes les espèces
    rows = []
    for sp, pts in all_coords.items():
        for lat, lon in pts:
            rows.append({"Espèce": sp, "lat": lat, "lon": lon})
    pts_gdf = gpd.GeoDataFrame(
        rows, geometry=[Point(r["lon"], r["lat"]) for r in rows], crs="EPSG:4326"
    )
    gdf_clim = gpd.read_file(shapefile_clim).to_crs(epsg=4326)
    gdf_eco  = gpd.read_file(shapefile_eco).to_crs(epsg=4326)

    clim_join = gpd.sjoin(pts_gdf, gdf_clim, how="left", predicate="intersects")
    eco_join  = gpd.sjoin(pts_gdf, gdf_eco,  how="left", predicate="intersects")

    out = {}
    for idx, row in pts_gdf.iterrows():
        lat, lon, sp = row["lat"], row["lon"], row["Espèce"]
        c = clim_join.loc[idx]
        e = eco_join.loc[idx]
        out[(sp, lat, lon)] = {
            "Climat":         c.get("CLIMATE"),
            "Sub-climat":     c.get("SUB-CLIMAT"),
            "Sub-sub-climat": c.get("SUB-SUB-CL"),
            "Ecorégion":      e.get("ECO_NAME")
        }
    return out

def carte(dataset_dir):
    # Créer une carte centrée sur une zone approximative
    m = folium.Map(location=[0, 0], zoom_start=2)

    # Parcourir chaque dossier d'espèce
    for species_name in os.listdir(dataset_dir):
        species_dir = os.path.join(dataset_dir, species_name)
        coordinates_file = os.path.join(species_dir, 'coordinates.txt')

        if os.path.isfile(coordinates_file):
            with open(coordinates_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Coordinates:"):
                        try:
                            # Extraction des coordonnées
                            _, coords = line.split(":")
                            lat, lon = map(float, coords.split(','))

                            # Ajouter un marqueur sur la carte
                            folium.Marker(
                                location=[lat, lon], 
                                popup=species_name, 
                                icon=folium.Icon(color="blue")
                            ).add_to(m)

                        except ValueError as e:
                            print(f"Erreur en lisant les coordonnées pour {species_name}: {line} ({e})")

        # Enregistrer la carte et l'ouvrir dans un navigateur
        m.save(f"{species_dir}/carte_oiseaux.html")
        print("Carte enregistrée sous 'carte_oiseaux.html'")
