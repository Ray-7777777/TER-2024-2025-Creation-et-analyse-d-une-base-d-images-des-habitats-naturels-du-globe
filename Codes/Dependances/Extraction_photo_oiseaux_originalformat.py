import os
import requests
import geopandas as gpd
import climatsEtHabitats

# Dossier parent pour toutes les espèces d'oiseaux
dataset_dir = '/Donnees/birds_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Shapefile pour les climats
shapefile_climats = gpd.read_file("/Donnees/climates/climates.shp")

# Shapefile pour les ecoregions
shapefile_ecoregions = gpd.read_file("/Donnees/Ecoregions/wwf_terr_ecos.shp")

raster_ecosystemes = "/Donnees/Ecosystemes/raster"

# Csv avonet pour les habitats
avonet = "/Donnees/avonet/AVONET2_eBird.xlsx"
sheet_name = "AVONET2_eBird"

# Nombre d'espèces et d'images par espèce
num_species = 50
num_images_per_species = 5


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
    coordinates = []

    for obs in observations:
        # Vérification si 'obs' est un dictionnaire valide
        if obs and isinstance(obs, dict):
            geojson_coords = obs.get('geojson', {}).get('coordinates', None)

            if geojson_coords:  # Vérifie si des coordonnées sont disponibles
                longitude, latitude = geojson_coords  # Les coordonnées sont dans l'ordre [longitude, latitude]
                coordinates.append((latitude, longitude))  # Ajoute les coordonnées à la liste
            else:
                print(f"Aucune coordonnée disponible pour l'observation {obs.get('id', 'ID inconnu')}")
        else:
            print(f"Observation invalide ou mal formatée pour {species_name}")

        # Vérifie si 'photos' existe et contient des données
        if 'photos' in obs:
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

    # Créer un sous-dossier pour l'espèce
    if photo_urls:
        species_dir = os.path.join(dataset_dir, species_name)
        os.makedirs(species_dir, exist_ok=True)

        # Télécharger les images, gérer les erreurs pour chaque image
        for i, url in enumerate(photo_urls):
            try:
                # Extraire l'extension du fichier à partir de l'URL (ex: .jpg, .png)
                file_extension = url.split('.')[-1]
                response = requests.get(url)
                response.raise_for_status()  # Lève une exception si le téléchargement échoue
                with open(f'{species_dir}/{species_name}_{i+1}.{file_extension}', 'wb') as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Erreur lors du téléchargement de l'image {i+1} pour {species_name}: {e}")
                continue  # Passer à l'image suivante même en cas d'erreur

    # Appel des fonctions pour récupérer le climat, le(s) ecosystème(s), l'habitat et l'écorégion associés aux coordonnées de l'observation
    if coordinates:
        climatsEtHabitats.climats(coordinates, shapefile_climats, species_name, dataset_dir)
        climatsEtHabitats.ecoregions(coordinates, shapefile_ecoregions, species_name, dataset_dir)
        climatsEtHabitats.ecosystemes(coordinates, raster_ecosystemes, species_name, dataset_dir)
        #climatsEtHabitats.carte(dataset_dir)
    climatsEtHabitats.avonet_habitats(avonet, sheet_name, species_name, dataset_dir)
    

# Récupérer les premières `num_species` espèces d'oiseaux
bird_species = get_bird_species()

# Télécharger les images pour chaque espèce d'oiseau
# Récupérer les premières `num_species` espèces d'oiseaux
bird_species = get_bird_species()

# Télécharger les images pour chaque espèce d'oiseau
for species in bird_species:
    species_name = species['name'].replace(" ", "_")  # Nom de l'espèce
    taxon_id = species['id']  # ID taxonomique de l'espèce
    print(f"Téléchargement des images pour {species_name}...")
    
    try:
        download_images_for_species(taxon_id, species_name)  # Télécharger les images pour l'espèce
    except Exception as e:
        print(f"Erreur lors du téléchargement des images pour {species_name}: {e}")
        continue  # Passer à l'espèce suivante en cas d'erreur

print("Téléchargement des images terminé.")

