import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import rasterio
from rasterio.coords import BoundingBox
from pyproj import Transformer
import numpy as np
import xml.etree.ElementTree as ET
import re

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
    for map_element in root.findall('Maps/Map'):  # Correction du chemin dans le XML
        efg_code = map_element.get('efg_code')  # Extraire le code efg_code
        description = map_element.find('Functional_group').text  # Extraire la description (Functional_group)
        efg_map[efg_code] = description  # Associer efg_code à la description

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
        # Exemple: "M2.2.web.orig_v2.0.tif" -> "M2.2" et pas seulement "M2"
        efg_code = '.'.join(fichier.split('.')[:2])  # Extraire "M2.2" au lieu de juste "M2"

        # Vérifier si ce code existe dans le dictionnaire des descriptions
        description = efg_map.get(efg_code, "Description not found")  # Utiliser la description si trouvée

        # Ouvrir le fichier raster
        with rasterio.open(raster_file) as src:
            # Pour chaque coordonnée, créer un Point et tester s'il est à l'intérieur du raster
            for coords in coordinates:
                # Transformer les coordonnées (si nécessaire) en EPSG:4326 (latitude, longitude)
                if src_epsg != 'epsg:4326':
                    lon, lat = transformer.transform(coords[0], coords[1])  # Transformation vers EPSG:4326
                else:
                    lat, lon = coords  # Si déjà en EPSG:4326, on les utilise telles quelles

                point = Point(lon, lat)  # Crée le point avec les coordonnées en EPSG:4326

                # Vérifier si le point est dans l'étendue du raster
                if not src.bounds[0] <= point.x <= src.bounds[2] or not src.bounds[1] <= point.y <= src.bounds[3]:
                    continue  # Si le point est en dehors des limites du raster, on passe au suivant

                # Convertir les coordonnées géographiques en coordonnées du raster (ligne, colonne)
                row, col = src.index(point.x, point.y)

                # Vérifier la valeur du pixel dans le raster
                value = src.read(1)[row, col]  # Lire la valeur du pixel à la position (row, col)
                
                # Si la valeur est 1 (Major occurrence) ou 2 (Minor occurrence), on les ajoute à la liste
                if value == 1 or value == 2:
                    occurrence_type = "Major occurrence" if value == 1 else "Minor occurrence"
                    # Ajouter seulement le groupe fonctionnel et le type d'occurrence à la liste
                    results[coords]["inside_rasters"].append(f"{description}, {occurrence_type}")

    # Enregistrer les résultats dans un fichier texte
    output_file = os.path.join(species_dir, 'ecosystemes_data.txt')
    with open(output_file, 'w') as file:
        # Pour chaque paire de coordonnées, concaténer toutes les descriptions des groupes fonctionnels
        for coords, data in results.items():
            if data["inside_rasters"]:
                # Concaténer les informations de tous les rasters pour cette coordonnée
                line = f"Coordinates {coords}: " + ", ".join(data["inside_rasters"])
                # Écrire la ligne dans le fichier texte
                file.write(line + "\n")
