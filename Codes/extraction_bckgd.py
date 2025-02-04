import os
import cv2
import numpy as np
import random

def extract_background(image_path, txt_file_path, output_dir, species, min_region_size=0.05, min_pixel_threshold=0.01):
    """
    Extrait les backgrounds d'une image en ignorant les régions masquées par les bounding boxes.

    Args:
        image_path (str): Chemin vers l'image.
        txt_file_path (str): Chemin vers le fichier contenant les bounding boxes YOLOv5.
        output_dir (str): Dossier de sortie pour enregistrer les backgrounds extraits.
        species (str): Nom de l'espèce correspondant à l'image.
        min_region_size (float): Surface minimale acceptable pour une région en fraction de l'image.
        min_pixel_threshold (float): Seuil minimal de pixels non masqués restants pour continuer.
    """
    if not os.path.exists(image_path):
        print(f"Erreur : L'image {image_path} est introuvable.")
        return

    if not os.path.exists(txt_file_path):
        print(f"Erreur : Le fichier texte {txt_file_path} est introuvable.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return

    image_height, image_width, _ = image.shape
    print(f"Traitement de l'image : {image_path} (taille={image_width}x{image_height})")

    # Charger le fichier YOLOv5
    with open(txt_file_path, 'r') as file:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        bbox_coords = []  # Stocker les coordonnées de toutes les bounding boxes
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Convertir les coordonnées normalisées en pixels
            x_center = int(x_center * image_width)
            y_center = int(y_center * image_height)
            width = int(width * image_width)
            height = int(height * image_height)

            # Ajouter une marge de 15% à la bounding box
            margin_w = int(width * 0.15)
            margin_h = int(height * 0.15)
            width += 2 * margin_w
            height += 2 * margin_h

            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(image_width - 1, x_center + width // 2)
            y2 = min(image_height - 1, y_center + height // 2)

            print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            bbox_coords.append((x1, y1, x2, y2))

            # Appliquer le masque
            mask[y1:y2 + 1, x1:x2 + 1] = 255

    # Identifier les zones non masquées
    regions_extracted = []
    min_area = int(min_region_size * image_width * image_height)
    ignored_regions = 0  # Compteur de régions ignorées

    while np.any(mask == 0):
        non_masked_indices = np.argwhere(mask == 0)

        if len(non_masked_indices) < min_pixel_threshold * image_width * image_height:
            print("Proportion de pixels non masqués trop faible. Arrêt.")
            break

        random_index = random.choice(non_masked_indices)
        start_x, start_y = random_index[1], random_index[0]
        print(f"Point aléatoire sélectionné : ({start_x}, {start_y})")

        inside_bbox = any(x1 <= start_x <= x2 and y1 <= start_y <= y2 for x1, y1, x2, y2 in bbox_coords)
        if inside_bbox:
            print(f"Point aléatoire à l'intérieur de la bounding box. Ignoré.")
            continue

        top, bottom, left, right = start_y, start_y, start_x, start_x
        expanded = True
        while expanded:
            expanded = False

            if top > 0 and not np.any(mask[top - 1, left:right + 1] == 255):
                top -= 1
                expanded = True

            if bottom < image_height - 1 and not np.any(mask[bottom + 1, left:right + 1] == 255):
                bottom += 1
                expanded = True

            if left > 0 and not np.any(mask[top:bottom + 1, left - 1] == 255):
                left -= 1
                expanded = True

            if right < image_width - 1 and not np.any(mask[top:bottom + 1, right + 1] == 255):
                right += 1

        region_width = right - left + 1
        region_height = bottom - top + 1
        region_area = region_width * region_height

        print(f"Région détectée : top={top}, bottom={bottom}, left={left}, right={right}")
        print(f"Dimensions de la région : largeur={region_width}, hauteur={region_height}, surface={region_area}")

        if region_width <= 0 or region_height <= 0 or region_area < min_area:
            print(f"Région invalide détectée ({region_width}x{region_height}). Ignorée.")
            ignored_regions += 1
            if ignored_regions >= 3:
                print("Trop de régions ignorées. Arrêt.")
                break
            continue

        mask[top:bottom + 1, left:right + 1] = 255
        region = image[top:bottom + 1, left:right + 1]
        if region.size == 0:
            print("Erreur : Région extraite vide. Ignorée.")
            ignored_regions += 1
            if ignored_regions >= 3:
                print("Trop de régions ignorées. Arrêt.")
                break
            continue

        regions_extracted.append((region_area, region))

    # Trier les régions par taille décroissante
    regions_extracted.sort(reverse=True, key=lambda x: x[0])

    # Sauvegarder les trois plus grandes régions
    species_output_dir = os.path.join(output_dir, species)
    os.makedirs(species_output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]  # Récupérer le nom de base de l'image
    for i, (area, region) in enumerate(regions_extracted[:3]):
        region_path = os.path.join(species_output_dir, f'{base_name}_{i + 1}.jpeg')
        cv2.imwrite(region_path, region)
        print(f"Région de fond sauvegardée dans : {region_path}")

    print(f"Extraction terminée. {len(regions_extracted[:3])} régions de fond sauvegardées.")

# Exemple d'exécution
parent_folder = r"../Donnees/birds_dataset2"  # Dossier des images
output_folder = r"../Donnees/birds_dataset/Background_extracted"  # Dossier de sortie pour les backgrounds

os.makedirs(output_folder, exist_ok=True)

for species in os.listdir(parent_folder):
    species_path = os.path.join(parent_folder, species)

    if os.path.isdir(species_path):
        train_path = os.path.join(species_path, 'train')
        bbox_folder = os.path.join(species_path, 'train', 'results', 'labels')

        if os.path.exists(train_path):
            for file in os.listdir(train_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(train_path, file)
                    bbox_file = os.path.join(bbox_folder, f"{os.path.splitext(file)[0]}.txt")

                    extract_background(image_path, bbox_file, output_folder, species)
