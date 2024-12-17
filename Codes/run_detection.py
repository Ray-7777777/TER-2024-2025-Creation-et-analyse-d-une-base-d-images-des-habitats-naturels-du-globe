"""import os
import cv2


# Fonction pour récupérer les fragments de background
def extract_background(image_path, bbox_file, output_dir):
    ""
    Extrait le background d'une image en excluant la zone définie par les boîtes englobantes (bounding boxes).

    :param image_path: Chemin vers l'image originale
    :param bbox_file: Chemin vers le fichier texte contenant les coordonnées des bounding boxes
    :param output_dir: Répertoire où enregistrer l'image de background
    ""
    if not os.path.exists(image_file):
        print(f"Le fichier image {image_file} n'existe pas.")
        return


    # Charger l'image originale
    image = cv2.imread(image_path)

    # Lire les coordonnées des boîtes englobantes depuis le fichier .txt
    with open(bbox_file, 'r') as f:
        for line in f:
            # YOLOv5 format : class x_center y_center width height
            bbox = line.strip().split()
            _, x_center, y_center, width, height = map(float, bbox)

            # Convertir les coordonnées relatives en absolues
            img_h, img_w = image.shape[:2]
            x_center = int(x_center * img_w)
            y_center = int(y_center * img_h)
            width = int(width * img_w)
            height = int(height * img_h)

            # Calculer les coordonnées de la bounding box
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Remplacer la zone de l'objet par du noir (ou une autre couleur/background)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Noir dans la zone de l'objet

    # Sauvegarder l'image de background
    os.makedirs(output_dir, exist_ok=True)
    background_output_path = os.path.join(output_dir, f'background_{os.path.basename(image_path)}')
    cv2.imwrite(background_output_path, image)
    print(f"Background extrait et sauvegardé : {background_output_path}")


# Exemple d'utilisation après la détection
image_file ="../Données/birds_dataset/Ardea_herodias/test/Ardea_herodias_1.jpeg"
bbox_file = '../Données/birds_dataset/Ardea_herodias/labels/exp/labels/Ardea_herodias_1.txt'
output_background_dir = r'..\Données\birds_dataset\Ardea_herodias\labels\exp'

# Extrait le background
extract_background(image_file, bbox_file, output_background_dir)
"""

import os
import subprocess

# Chemin vers le dossier des espèces
birds_dataset_path = r"../Donnees/birds_dataset"  # Remplacez par le chemin réel
detect_script_path = r"../../yolov5/detect.py"  # Remplacez par le chemin vers detect.py
weights_path = r"../../yolov5/yolov5s.pt"  # Chemin vers le fichier de poids du modèle
img_size = 640  # Taille d'image
conf_thres = 0.25  # Seuil de confiance

# Parcourir chaque espèce dans le dossier birds_dataset
for species in os.listdir(birds_dataset_path):
    species_path = os.path.join(birds_dataset_path, species)

    if os.path.isdir(species_path):
        train_path = os.path.join(species_path, 'train')

        if os.path.exists(train_path):
            print(f"Traitement des images dans le dossier : {train_path}")

            # Exécuter detect.py sur chaque image dans le dossier train
            command = [
                'python', detect_script_path,
                '--source', train_path,
                '--weights', weights_path,
                '--img-size', str(img_size),
                '--conf-thres', str(conf_thres),
                '--save-txt',  # Pour sauvegarder les résultats dans des fichiers texte
                '--project', train_path,  # Spécifie le dossier pour enregistrer les résultats
                '--name', 'results'  # Nom du dossier où seront enregistrés les résultats
            ]

            # Exécuter la commande
            subprocess.run(command)

print("Détection terminée pour toutes les espèces.")









