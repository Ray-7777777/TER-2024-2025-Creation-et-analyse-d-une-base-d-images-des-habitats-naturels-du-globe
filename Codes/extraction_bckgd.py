import os
import cv2


# Fonction pour récupérer les fragments de background
def extract_background_with_margin(image_path, bbox_file, output_dir, margin=0.1):
    """
    Extrait une zone autour de la bounding box d'une image, en incluant une marge.

    :param image_path: Chemin vers l'image originale
    :param bbox_file: Chemin vers le fichier texte contenant les coordonnées des bounding boxes
    :param output_dir: Répertoire où enregistrer l'image de background
    :param margin: Pourcentage de marge à ajouter autour de la bounding box (par défaut 10%)
    """
    # Vérifier si l'image existe
    if not os.path.exists(image_path):
        print(f"Le fichier image {image_path} n'existe pas.")
        return

    # Vérifier si le fichier des bounding boxes existe
    if not os.path.exists(bbox_file):
        print(f"Le fichier des boîtes englobantes {bbox_file} n'existe pas.")
        return

    # Charger l'image originale
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return

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

            # Calculer les coordonnées de la bounding box avec une marge
            margin_w = int(width * margin)
            margin_h = int(height * margin)

            # Ajouter une marge autour de la bounding box
            x1 = max(0, int(x_center - width / 2 - margin_w))  # S'assurer que les coordonnées restent dans l'image
            y1 = max(0, int(y_center - height / 2 - margin_h))
            x2 = min(img_w, int(x_center + width / 2 + margin_w))
            y2 = min(img_h, int(y_center + height / 2 + margin_h))

            # Extraire la zone du background
            background_area = image[y1:y2, x1:x2]

            # Sauvegarder l'image de background
            os.makedirs(output_dir, exist_ok=True)
            background_output_path = os.path.join(output_dir, f'background_{os.path.basename(image_path)}')
            cv2.imwrite(background_output_path, background_area)
            print(f"Background extrait et sauvegardé : {background_output_path}")


# Exemple d'utilisation après la détection
image_file = r"C:\Users\alvin\Documents\M1 MIASHS\S1\TER\TER-2024-2025-Creation-et-analyse-d-une-base-d-images-des-habitats-naturels-du-globe\Donnees\birds_dataset2\Ardea_alba\test\Ardea_alba_1.jpg"  # Chemin vers l'image originale
bbox_file = r"../Donnees/ birds_dataset/Ardea_herodias/labels/exp/labels/Ardea_herodias_1.txt"  # Chemin vers le fichier des bounding boxes
output_background_dir = "../Donnees/ birds_dataset/Ardea_herodias/labels/exp/crops"  # Répertoire pour sauvegarder le background

# Extrait le background avec une marge de 10%
extract_background_with_margin(image_file, bbox_file, output_background_dir, margin=0.1)

