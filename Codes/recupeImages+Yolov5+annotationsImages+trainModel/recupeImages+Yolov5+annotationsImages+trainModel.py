"""
Première partie: Téléchargement des images
"""
import os
import requests

# Dossier parent pour toutes les espèces d'oiseaux
dataset_dir = 'birds_dataset'
os.makedirs(dataset_dir, exist_ok=True)

# Nombre d'espèces et d'images par espèce
num_species = 2
num_images_per_species = 3

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

    # Créer un sous-dossier pour l'espèce
    if photo_urls:
        species_dir = os.path.join(dataset_dir, species_name)
        os.makedirs(species_dir, exist_ok=True)

        # Télécharger les images
        for i, url in enumerate(photo_urls):
            try:
                # Extraire l'extension du fichier à partir de l'URL (ex: .jpg, .png)
                file_extension = url.split('.')[-1]
                response = requests.get(url)
                with open(f'{species_dir}/{species_name}_{i+1}.{file_extension}', 'wb') as f:
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



"""
Deuxième partie: annotations des images pour traitement
"""
import os
import torch
import random
from pathlib import Path
from PIL import Image
import subprocess
import sys

# Fonction pour installer un package via pip
def install_package(package_name):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

# Cloner le dépôt YOLOv5 si ce n'est pas déjà fait
if not os.path.exists('yolov5'):
    print("Clonage du dépôt YOLOv5...")
    subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5"])
else:
    print("Le dépôt YOLOv5 existe déjà.")

# Installer les dépendances YOLOv5
print("Installation des dépendances YOLOv5...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"])

# Vérifier si PyTorch est installé, sinon l'installer
try:
    import torch
except ImportError:
    print("PyTorch n'est pas installé, installation en cours...")
    install_package('torch')

# Charger le modèle YOLOv5
print("Chargement du modèle YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

print("Modèle YOLOv5 chargé avec succès !")

# Dossier de base contenant les sous-dossiers d'espèces
base_dir = 'birds_dataset'

# Mapping des noms d'espèces à des IDs
species_ids = {species: idx for idx, species in enumerate(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, species))}

# Dossiers pour l'entraînement et la validation
train_images_dir = 'birds_dataset/train/images'
val_images_dir = 'birds_dataset/val/images'
train_labels_dir = 'birds_dataset/train/labels'
val_labels_dir = 'birds_dataset/val/labels'

# Créer les dossiers de sortie s'ils n'existent pas
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# Fonction pour générer les annotations YOLO pour chaque image
def generate_annotations(image_path, species_name, output_annotation_path):
    # Charger l'image
    img = Image.open(image_path)
    img_width, img_height = img.size  # Obtenir la taille de l'image

    # Faire la prédiction avec YOLOv5
    results = model(img)

    # Extraire les boîtes englobantes (bounding boxes)
    detections = results.xywh[0]  # Obtenir les boîtes au format [x_center, y_center, width, height]

    # Sauvegarder l'annotation au format YOLO avec le nom de l'espèce
    with open(output_annotation_path, 'w') as f:
        for detection in detections:
            x_center, y_center, width, height = detection[:4]
            confidence = detection[4]
            class_id = int(detection[5])

            # Filtrer pour garder uniquement les oiseaux (classe ID pour oiseau = 14 dans COCO)
            if class_id == 14:
                # Normaliser les coordonnées
                x_center_normalized = x_center.item() / img_width
                y_center_normalized = y_center.item() / img_height
                width_normalized = width.item() / img_width
                height_normalized = height.item() / img_height

                # Remplacer le class_id par l'ID correspondant à l'espèce
                class_id = species_ids[species_name]

                # Écrire l'annotation dans le fichier
                f.write(f'{class_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n')

# Parcourir chaque dossier d'espèces
for species_folder in os.listdir(base_dir):
    species_path = os.path.join(base_dir, species_folder)

    if os.path.isdir(species_path):
        # Obtenir la liste des images dans ce dossier d'espèces
        image_files = [f for f in os.listdir(species_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Diviser en 80% d'images pour l'entraînement et 20% pour la validation
        random.shuffle(image_files)
        train_split = int(0.8 * len(image_files))
        train_files = image_files[:train_split]
        val_files = image_files[train_split:]

        for image_file in train_files:
            image_path = os.path.join(species_path, image_file)

            # Chemin de sauvegarde pour les annotations
            annotation_file = os.path.splitext(image_file)[0] + '.txt'
            output_annotation_path = os.path.join(train_labels_dir, annotation_file)

            # Générer les annotations correspondantes
            generate_annotations(image_path, species_folder, output_annotation_path)

            # Vérifier si le fichier d'annotation est vide
            if os.path.getsize(output_annotation_path) > 0:
                # Copier les images dans le dossier train avec le nom original
                output_image_path = os.path.join(train_images_dir, image_file)
                Image.open(image_path).save(output_image_path)

                # Supprimer l'image source après la copie
                os.remove(image_path)
            else:
                # Supprimer l'image originale si l'annotation est vide
                os.remove(image_path)

            # Supprimer l'annotation si elle est vide
            if os.path.getsize(output_annotation_path) == 0:
                os.remove(output_annotation_path)

        for image_file in val_files:
            image_path = os.path.join(species_path, image_file)

            # Chemin de sauvegarde pour les annotations
            annotation_file = os.path.splitext(image_file)[0] + '.txt'
            output_annotation_path = os.path.join(val_labels_dir, annotation_file)

            # Générer les annotations correspondantes
            generate_annotations(image_path, species_folder, output_annotation_path)

            # Vérifier si le fichier d'annotation est vide
            if os.path.getsize(output_annotation_path) > 0:
                # Copier les images dans le dossier val avec le nom original
                output_image_path = os.path.join(val_images_dir, image_file)
                Image.open(image_path).save(output_image_path)

                # Supprimer l'image source après la copie
                os.remove(image_path)
            else:
                # Supprimer l'image originale si l'annotation est vide
                os.remove(image_path)

            # Supprimer l'annotation si elle est vide
            if os.path.getsize(output_annotation_path) == 0:
                os.remove(output_annotation_path)

        # Supprimer le dossier d'espèce s'il est vide
        if not os.listdir(species_path):
            os.rmdir(species_path)

        print(f"Images et annotations traitées pour l'espèce {species_folder}.")

# Générer le fichier data.yaml
data_yaml_path = 'data.yaml'
with open(data_yaml_path, 'w') as f:
    f.write(f'train: ../{train_images_dir}\n')  # Ajout de ../
    f.write(f'val: ../{val_images_dir}\n')      # Ajout de ../
    f.write(f'nc: {len(species_ids)}\n')         # Nombre de classes (espèces)
    f.write('names: [\n')
    for species in species_ids.keys():
        f.write(f"  '{species}',\n")  # Les noms d'espèces
    f.write(']\n')

print(f"Fichier data.yaml généré à l'emplacement : {data_yaml_path}")

# Commande pour entraîner le modèle YOLOv5
print("Entraînement du modèle YOLOv5...")
subprocess.run([sys.executable, 'yolov5/train.py', '--img', '640', '--epochs', '3', '--data', data_yaml_path, '--weights', 'yolov5m.pt'])

print("Entraînement terminé !")
