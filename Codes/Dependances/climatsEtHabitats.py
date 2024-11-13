import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

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