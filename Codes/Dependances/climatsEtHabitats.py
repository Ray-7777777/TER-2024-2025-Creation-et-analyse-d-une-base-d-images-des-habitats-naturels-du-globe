import os
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
    start = time.time()
    print(f"\n📥 [ecosystemes] Début du traitement pour {species_name} ({len(coordinates)} points)")

    species_dir = os.path.join(dataset_dir, species_name)
    results = {coord: {"inside_rasters": []} for coord in coordinates}

    # Charger le fichier XML
    try:
        xml_file = os.path.join(dossier_raster, '..', 'map-details.xml')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        efg_map = {el.get('efg_code'): el.find('Functional_group').text for el in root.findall('Maps/Map')}
    except Exception as e:
        print(f"❌ Erreur lors du chargement du XML : {e}")
        return

    # Transformation de coordonnées si nécessaire
    transformer = Transformer.from_crs(src_epsg, 'epsg:4326', always_xy=True)

    raster_files = [f for f in os.listdir(dossier_raster) if f.endswith('.tif')]
    print(f"📂 {len(raster_files)} fichiers .tif détectés dans le dossier raster")

    for idx, fichier in enumerate(raster_files):
        raster_path = os.path.join(dossier_raster, fichier)
        efg_code = '.'.join(fichier.split('.')[:2])
        description = efg_map.get(efg_code, "Description inconnue")

        print(f"\n🔢 [{idx + 1}/{len(raster_files)}] Traitement du raster : {fichier}")

        try:
            with rasterio.open(raster_path) as src:
                print(f"✅ Raster ouvert : {fichier}")
                for i, coord in enumerate(coordinates):
                    try:
                        print(f"  ➡️ Coordonnée {i + 1} : {coord}")
                        if src_epsg != 'epsg:4326':
                            lon, lat = transformer.transform(coord[0], coord[1])
                        else:
                            lat, lon = coord

                        point = Point(lon, lat)

                        if not (src.bounds.left <= point.x <= src.bounds.right and
                                src.bounds.bottom <= point.y <= src.bounds.top):
                            print("    ⛔ En dehors des limites du raster")
                            continue

                        value = next(src.sample([(point.x, point.y)]))[0]
                        print(f"    📍 Valeur lue : {value}")

                        if value == 1 or value == 2:
                            occurrence = "Major occurrence" if value == 1 else "Minor occurrence"
                            results[coord]["inside_rasters"].append(f"{description}, {occurrence}")

                    except Exception as pixel_error:
                        print(f"    ⚠️ Erreur lecture pixel : {pixel_error}")
                        continue

        except Exception as raster_error:
            print(f"❌ Erreur à l'ouverture du fichier {fichier} : {raster_error}")
            continue

    # Sauvegarde des résultats
    output_file = os.path.join(species_dir, 'ecosystemes_data.txt')
    try:
        with open(output_file, 'w') as f:
            for coord, data in results.items():
                if data["inside_rasters"]:
                    line = f"Coordonnées {coord}: " + ", ".join(data["inside_rasters"])
                    f.write(line + "\n")
    except Exception as e:
        print(f"❌ Erreur lors de l'écriture du fichier résultats : {e}")

    duration = round(time.time() - start, 2)
    print(f"\n✅ [ecosystemes] Traitement terminé pour {species_name} en {duration} secondes")
