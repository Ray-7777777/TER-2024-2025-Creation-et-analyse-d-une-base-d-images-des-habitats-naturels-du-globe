import os
import requests
import csv

# Dossier parent pour stocker le fichier CSV
dataset_dir = 'birds_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Fichier CSV pour stocker les métadonnées des observations
csv_file = 'bird_observations_metadata.csv'

# Nombre d'espèces et d'images par espèce
num_species = 1000
num_images_per_species = 200

# Fonction pour récupérer les espèces d'oiseaux
def get_bird_species():
    species_list = []
    page = 1

    while len(species_list) < num_species:
        url = "https://api.inaturalist.org/v1/taxa"
        params = {
            'taxon_id': 3,  # ID taxonomique pour la classe "Aves"
            'rank': 'species',
            'per_page': 100,  # Nombre d'espèces par page
            'page': page
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                species_list.extend(data['results'])
            else:
                break
        else:
            print(f"Erreur lors de la récupération des espèces : {response.status_code}")
            break

        if len(data['results']) < 100:
            break

        page += 1

    return species_list[:num_species]

# Fonction pour récupérer les métadonnées des observations
def get_metadata_for_species(taxon_id, species_name):
    observations = []
    page = 1

    while len(observations) < num_images_per_species:
        url = "https://api.inaturalist.org/v1/observations"
        params = {
            'taxon_id': taxon_id,
            'per_page': 100,
            'page': page,
            'order_by': 'created_at'
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                observations.extend(data['results'])
            else:
                break
        else:
            print(f"Erreur lors de la récupération des observations pour {species_name}. Statut: {response.status_code}")
            break

        if len(data['results']) < 100:
            break

        page += 1

    print(f"Nombre d'observations récupérées pour {species_name}: {len(observations)}")

    # Garder les `num_images_per_species` premières observations
    observations = observations[:num_images_per_species]

    # Extraire les métadonnées et les écrire dans le fichier CSV
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        for obs in observations:
            # Vérifier que 'geojson' et 'coordinates' existent
            if 'geojson' in obs and obs['geojson'] and 'coordinates' in obs['geojson'] and obs['geojson']['coordinates']:
                coordinates = obs['geojson']['coordinates']
                lat = coordinates[1]  # Latitude
                lon = coordinates[0]  # Longitude
                date = obs['observed_on']  # Date de l'observation

                # Vérification des photos et exclusion des vidéos
                photo_url = None
                if 'photos' in obs:  # S'il y a des photos
                    for photo in obs['photos']:  # Parcourir toutes les photos
                        url = photo['url']
                        if url.lower().endswith(('.jpg', '.jpeg', '.png')):  # Vérifier si c'est une image
                            photo_url = url
                            break  # On prend la première image valide trouvée

                if photo_url:  # Ajouter uniquement si une image a été trouvée
                    writer.writerow([species_name, lat, lon, date, photo_url])
                else:
                    print(f"Aucune image trouvée pour {species_name} (seulement des vidéos ou fichiers non supportés).")
            else:
                print(f"Observation pour {species_name} ignorée (pas de coordonnées valides).")

# Récupérer les premières `num_species` espèces d'oiseaux
bird_species = get_bird_species()

# Écrire les en-têtes du fichier CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['species_name', 'latitude', 'longitude', 'observation_date', 'image_url'])

# Collecter les métadonnées pour chaque espèce d'oiseau
for species in bird_species:
    species_name = species['name'].replace(" ", "_")  # Nom de l'espèce
    taxon_id = species['id']  # ID taxonomique de l'espèce
    print(f"Collecte des métadonnées pour {species_name}...")
    get_metadata_for_species(taxon_id, species_name)

print("Collecte des métadonnées terminée.")
