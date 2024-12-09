import os
import requests
import random

# Dossier parent pour toutes les espèces d'oiseaux
dataset_dir = r'..\Donnees\birds_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Nombre d'espèces et d'images par espèce
num_species = 5
num_images_per_species = 10

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

        # Arrêter si moins de 100 résultats sont retournés (fin des résultats)
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
            print(f"Erreur lors de la récupération des observations pour {species_name}. Statut: {response.status_code}")
            break

        if len(data['results']) < 100:
            break

        page += 1

    print(f"Nombre d'observations récupérées pour {species_name}: {len(observations)}")

    # Garder les `num_images_per_species` premières observations
    observations = observations[:num_images_per_species]

    # Extraire et télécharger les photos spécifiques à chaque observation
    photo_urls = []
    for obs in observations:
        if 'photos' in obs:  # Vérifier s'il y a des photos dans l'observation
            for photo in obs['photos']:
                # Utiliser l'URL de la photo existante et ajouter "original" au bon format
                base_url = photo['url'].rsplit('/', 1)[0]  # Récupère la base de l'URL
                if photo['url'].endswith(".jpg") or photo['url'].endswith(".jpeg"):
                    # Ajouter "original" à l'URL pour télécharger la version originale
                    photo_url = f"{base_url}/original.jpg" if photo['url'].endswith(".jpg") else f"{base_url}/original.jpeg"
                    print(f"Téléchargement de l'image : {photo_url}")
                    photo_urls.append(photo_url)

    # Limiter aux `num_images_per_species`
    photo_urls = photo_urls[:num_images_per_species]

    # Mélanger les URLs pour randomiser la sélection des 80% et 20%
    random.shuffle(photo_urls)

    # Diviser les images en 80% pour train et 20% pour test
    split_index = int(0.8 * len(photo_urls))
    train_urls = photo_urls[:split_index]
    test_urls = photo_urls[split_index:]

    # Créer les sous-dossiers pour l'entraînement et le test
    species_train_dir = os.path.join(dataset_dir, species_name, 'train')
    species_test_dir = os.path.join(dataset_dir, species_name, 'test')
    os.makedirs(species_train_dir, exist_ok=True)
    os.makedirs(species_test_dir, exist_ok=True)

    # Télécharger les images pour l'ensemble d'entraînement
    for i, url in enumerate(train_urls):
        try:
            file_extension = url.split('.')[-1]
            response = requests.get(url)
            with open(f'{species_train_dir}/{species_name}_{i+1}.{file_extension}', 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {i+1} pour {species_name}: {e}")

    # Télécharger les images pour l'ensemble de test
    for i, url in enumerate(test_urls):
        try:
            file_extension = url.split('.')[-1]
            response = requests.get(url)
            with open(f'{species_test_dir}/{species_name}_{i+1}.{file_extension}', 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {i+1} pour {species_name}: {e}")

# Récupérer les premières `num_species` espèces d'oiseaux
bird_species = get_bird_species()

# Télécharger les images pour chaque espèce d'oiseau
for species in bird_species:
    species_name = species['name'].replace(" ", "_")  # Nom de l'espèce
    taxon_id = species['id']  # ID taxonomique de l'espèce
    print(f"Téléchargement des images pour {species_name}...")
    download_images_for_species(taxon_id, species_name)

print("Téléchargement des images terminé.")
