import os
import cv2
import numpy as np
import random

# Paramètres globaux
min_region_size = 0.01  # Fraction minimale de la surface totale de l'image
min_pixel_threshold = 0.01  # Fraction minimale des pixels restants pour continuer
max_ignored_regions = 5  # Nombre maximal de régions ignorées avant d'arrêter

# Dossiers des données
parent_folder = '../Donnees/birds_dataset'  # Dossier contenant les espèces d'oiseaux
detect_base = './runs/detect'  # Base des sorties YOLO\
output_folder = '../Donnees/background_extracted'  # Dossier de sortie

# Création du dossier de sortie
os.makedirs(output_folder, exist_ok=True)

def extract_backgrounds(image_path, txt_file_path, output_dir):
    """Extrait les backgrounds d'une image tout en évitant les régions masquées."""
    
    # Vérification des fichiers
    if not os.path.exists(image_path):
        print(f"Erreur : L'image {image_path} est introuvable.")
        return

    if not os.path.exists(txt_file_path):
        print(f"Erreur : Le fichier texte {txt_file_path} est introuvable.")
        return

    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return

    image_height, image_width, _ = image.shape
    print(f"Taille de l'image : largeur={image_width}, hauteur={image_height}")

    # Initialisation des masques
    permanent_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    visited_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    # Charger le fichier YOLOv5 et parser les 6 colonnes
    with open(txt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                print(f"Format inattendu dans {txt_file_path} : « {line.strip()} »")
                continue

            # Désempaqueter les 6 valeurs : classe, centre_x, centre_y, largeur, hauteur, confiance
            class_id, x_center, y_center, width, height, confidence = map(float, parts[:6])
            class_id = int(class_id)

            # Conversion en pixels
            x_center = int(x_center * image_width)
            y_center = int(y_center * image_height)
            width    = int(width    * image_width)
            height   = int(height   * image_height)

            # Ajouter une marge de 15% à la bounding box
            margin_w = int(width * 0.15)
            margin_h = int(height * 0.15)
            width  += 2 * margin_w
            height += 2 * margin_h

            x1 = max(0, x_center - width // 2)
            y1 = max(0, y_center - height // 2)
            x2 = min(image_width - 1, x_center + width // 2)
            y2 = min(image_height - 1, y_center + height // 2)

            # Logging des valeurs, incluant la confiance
            print(f"Detection – classe={class_id}, conf={confidence:.2f}, "
                  f"bbox=({x1},{y1}) -> ({x2},{y2})")

            permanent_mask[y1:y2 + 1, x1:x2 + 1] = 255

    # Sauvegarde du masque permanent pour debug
    debug_perm_path = os.path.join(output_dir, f'debug_permanent_mask_{os.path.basename(image_path)}.jpg')
    cv2.imwrite(debug_perm_path, permanent_mask)

    regions_extracted = 0
    ignored_regions = 0
    min_area = int(min_region_size * image_width * image_height)

    # Region-growing pour extraire les backgrounds
    while np.any((permanent_mask == 0) & (visited_mask == 0)):
        non_masked_indices = np.argwhere((permanent_mask == 0) & (visited_mask == 0))
        start_y, start_x = random.choice(non_masked_indices)
        top = bottom = start_y
        left = right = start_x
        expanded = True

        while expanded:
            expanded = False
            if top > 0 and permanent_mask[top - 1, left:right + 1].max() == 0:
                top -= 1; expanded = True
            if bottom < image_height - 1 and permanent_mask[bottom + 1, left:right + 1].max() == 0:
                bottom += 1; expanded = True
            if left > 0 and permanent_mask[top:bottom + 1, left - 1].max() == 0:
                left -= 1; expanded = True
            if right < image_width - 1 and permanent_mask[top:bottom + 1, right + 1].max() == 0:
                right += 1; expanded = True

        region_width = right - left + 1
        region_height = bottom - top + 1
        region_area = region_width * region_height

        if region_area < min_area:
            ignored_regions += 1
            if ignored_regions >= max_ignored_regions:
                print("Trop de régions ignorées. Arrêt de l'extraction.")
                break
            continue

        visited_mask[top:bottom + 1, left:right + 1] = 255
        region = image[top:bottom + 1, left:right + 1]
        output_path = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}_bg_{regions_extracted + 1}.jpg"
        )
        cv2.imwrite(output_path, region)
        print(f"Région de fond sauvegardée dans : {output_path}")

        regions_extracted += 1

    # Sauvegarde du masque visité pour debug
    debug_vis_path = os.path.join(output_dir, f'debug_visited_mask_{os.path.basename(image_path)}.jpg')
    cv2.imwrite(debug_vis_path, visited_mask)

    print(f"Extraction terminée pour {image_path}: {regions_extracted} régions de fond sauvegardées.\n")

# Parcours des espèces et images
for species in os.listdir(parent_folder):
    species_path = os.path.join(parent_folder, species)
    if not os.path.isdir(species_path):
        continue

    # Dossier de sortie spécifique à l'espèce
    species_output_dir = os.path.join(output_folder, species)
    os.makedirs(species_output_dir, exist_ok=True)

    # Parcours des images
    for file in os.listdir(species_path):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(species_path, file)
        bbox_file = os.path.join(
            detect_base, species, 'labels', f"{os.path.splitext(file)[0]}.txt"
        )

        extract_backgrounds(image_path, bbox_file, species_output_dir)
