import os
import cv2
import numpy as np
import random

def extract_background(image_path, txt_file_path, output_dir, species, 
                       min_region_size=0.01, min_pixel_threshold=0.01, max_ignored_regions=5):
    """
    Extrait les backgrounds d'une image en ignorant les régions masquées par les bounding boxes.
    Utilise deux masques : permanent_mask (zones des objets) et visited_mask (zones déjà extraites).

    Args:
        image_path (str): Chemin vers l'image.
        txt_file_path (str): Chemin vers le fichier TXT contenant les bounding boxes YOLO (format : class, x_center, y_center, width, height).
        output_dir (str): Dossier racine de sortie pour les backgrounds.
        species (str): Nom de l'espèce (sera utilisé pour créer un sous-dossier).
        min_region_size (float): Fraction minimale de la surface totale de l'image pour accepter une région.
        min_pixel_threshold (float): Fraction minimale de pixels non masqués restants pour continuer l'extraction.
        max_ignored_regions (int): Nombre maximal de régions invalides avant d'arrêter.
    
    Returns:
        List[str]: Liste des chemins vers les images de background extraites.
    """
    if not os.path.exists(image_path):
        print(f"Erreur : L'image {image_path} est introuvable.")
        return []
    if not os.path.exists(txt_file_path):
        print(f"Erreur : Le fichier texte {txt_file_path} est introuvable.")
        return []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return []

    image_height, image_width, _ = image.shape
    print(f"Traitement de l'image : {image_path} (taille={image_width}x{image_height})")

    # Initialisation des masques
    permanent_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    visited_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Lecture du fichier TXT et remplissage du permanent_mask
    with open(txt_file_path, 'r') as file:
        for line in file:
            try:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
            except Exception as e:
                print("Erreur lors de la lecture de la ligne:", line)
                continue
            x_center = int(x_center * image_width)
            y_center = int(y_center * image_height)
            width = int(width * image_width)
            height = int(height * image_height)
            margin_w = int(width * 0.15)
            margin_h = int(height * 0.15)
            width += 2 * margin_w
            height += 2 * margin_h
            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(image_width - 1, x_center + width // 2)
            y2 = min(image_height - 1, y_center + height // 2)
            print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            permanent_mask[y1:y2+1, x1:x2+1] = 255

    extracted_paths = []
    regions_extracted = 0
    ignored_regions = 0
    min_area = int(min_region_size * image_width * image_height)

    while np.any((permanent_mask == 0) & (visited_mask == 0)):
        non_masked_indices = np.argwhere((permanent_mask == 0) & (visited_mask == 0))
        if len(non_masked_indices) < min_pixel_threshold * image_width * image_height:
            print("Proportion de pixels non masqués trop faible. Arrêt.")
            break

        random_index = random.choice(non_masked_indices)
        start_x, start_y = random_index[1], random_index[0]
        print(f"Point aléatoire sélectionné : ({start_x}, {start_y})")
        
        top, bottom, left, right = start_y, start_y, start_x, start_x
        expanded = True
        while expanded:
            expanded = False
            if top > 0 and permanent_mask[top-1, left:right+1].max() == 0:
                top -= 1
                expanded = True
            if bottom < image_height - 1 and permanent_mask[bottom+1, left:right+1].max() == 0:
                bottom += 1
                expanded = True
            if left > 0 and permanent_mask[top:bottom+1, left-1].max() == 0:
                left -= 1
                expanded = True
            if right < image_width - 1 and permanent_mask[top:bottom+1, right+1].max() == 0:
                right += 1
                expanded = True

        region_width = right - left + 1
        region_height = bottom - top + 1
        region_area = region_width * region_height
        print(f"Région détectée : top={top}, bottom={bottom}, left={left}, right={right}, surface={region_area}")

        if region_width <= 0 or region_height <= 0 or region_area < min_area:
            print(f"Région invalide détectée ({region_width}x{region_height}). Ignorée.")
            ignored_regions += 1
            if ignored_regions >= max_ignored_regions:
                print("Trop de régions ignorées. Arrêt de l'extraction.")
                break
            continue

        visited_mask[top:bottom+1, left:right+1] = 255
        region = image[top:bottom+1, left:right+1]
        species_output_dir = os.path.join(output_dir, species)
        os.makedirs(species_output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        region_filename = f"{base_name}_bg_{regions_extracted+1}.jpeg"
        region_path = os.path.join(species_output_dir, region_filename)
        cv2.imwrite(region_path, region)
        print(f"Région de fond sauvegardée dans : {region_path}")
        extracted_paths.append(region_path)
        regions_extracted += 1

    print(f"Extraction terminée. {regions_extracted} régions de fond sauvegardées.")
    return extracted_paths

