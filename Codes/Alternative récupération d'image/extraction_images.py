import os
import requests

# Dossier parent pour toutes les espèces d'oiseaux
dataset_dir = 'birds_dataset'
os.makedirs(dataset_dir, exist_ok=True)

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

# Fonction pour télécharger les images d'une espèce
def download_images_for_species(taxon_id, species_name):
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
            print(f"Erreur lors de la récupération des observations pour {species_name}.")
            break

        if len(data['results']) < 100:
            break

        page += 1

    observations = observations[:num_images_per_species]

    photo_urls = []
    for obs in observations:
        if 'photos' in obs:
            for photo in obs['photos']:
                photo_id = photo['id']
                extension = photo['url'].split('.')[-1]  # Extraire l'extension depuis l'URL
                photo_url = f"https://static.inaturalist.org/photos/{photo_id}/original.{extension}"  # URL format original
                photo_urls.append(photo_url)

    photo_urls = photo_urls[:num_images_per_species]

    species_dir = os.path.join(dataset_dir, species_name)
    os.makedirs(species_dir, exist_ok=True)

    for i, url in enumerate(photo_urls):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(f'{species_dir}/{species_name}_{i+1}.{url.split(".")[-1]}', 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Erreur lors du téléchargement de l'image {i+1} pour {species_name}.")
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {i+1} pour {species_name}: {e}")

# Récupérer les premières `num_species` espèces d'oiseaux
bird_species = get_bird_species()

# Télécharger les images pour chaque espèce d'oiseau
for species in bird_species:
    species_name = species['name'].replace(" ", "_")
    taxon_id = species['id']
    print(f"Téléchargement des images pour {species_name}...")
    download_images_for_species(taxon_id, species_name)

print("Téléchargement des images terminé.")
