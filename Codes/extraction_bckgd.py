import os
import cv2
import numpy as np
import random

# Chemins des fichiers
image_path = '../Donnees/Anas_platyrhynchos_6.jpeg'
txt_file_path = '../Donnees/Anas_platyrhynchos_6.txt'
output_dir = '../Donnees/'

# Paramètres
min_region_size = 0.01  # Fraction minimale de la surface totale de l'image
min_pixel_threshold = 0.01  # Fraction minimale des pixels restants pour continuer

# Vérification des fichiers
if not os.path.exists(image_path):
    print(f"Erreur : L'image {image_path} est introuvable.")
    exit()

if not os.path.exists(txt_file_path):
    print(f"Erreur : Le fichier texte {txt_file_path} est introuvable.")
    exit()

# Charger l'image
image = cv2.imread(image_path)
if image is None:
    print(f"Erreur : Impossible de charger l'image {image_path}")
    exit()

# Dimensions de l'image
image_height, image_width, _ = image.shape
print(f"Taille de l'image : largeur={image_width}, hauteur={image_height}")

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

        # Calculer les nouvelles coordonnées de la bounding box avec marge
        x1 = max(0, x_center - width // 2)
        y1 = max(0, y_center - height // 2)
        x2 = min(image_width - 1, x_center + width // 2)
        y2 = min(image_height - 1, y_center + height // 2)

        print(f"Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        bbox_coords.append((x1, y1, x2, y2))

        # Appliquer le masque à la région
        mask[y1:y2 + 1, x1:x2 + 1] = 255
        print(f"Masque appliqué pour la bounding box : x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        # Debug : Sauvegarder le masque après chaque mise à jour
        debug_mask_path = os.path.join(output_dir, f'debug_mask_bbox.jpg')
        debug_image = image.copy()
        debug_image[mask == 255] = [0, 0, 255]  # Marquer les zones masquées en rouge
        cv2.imwrite(debug_mask_path, debug_image)
        print(f"Masque mis à jour sauvegardé : {debug_mask_path}")

# Identifier les zones non masquées
regions_extracted = 0
min_area = int(min_region_size * image_width * image_height)  # Surface minimale

while np.any(mask == 0):  # Tant qu'il reste des zones non masquées
    # Identifier les indices des pixels non masqués
    non_masked_indices = np.argwhere(mask == 0)

    # Condition de sortie si trop peu de pixels non masqués restent
    if len(non_masked_indices) < min_pixel_threshold * image_width * image_height:
        print("Proportion de pixels non masqués trop faible. Arrêt.")
        break

    # Choisir un point aléatoire parmi les indices valides
    random_index = random.choice(non_masked_indices)
    start_x, start_y = random_index[1], random_index[0]
    print(f"Point aléatoire sélectionné : ({start_x}, {start_y})")

    # Vérifiez si le point est à l'intérieur d'une bounding box
    inside_bbox = any(x1 <= start_x <= x2 and y1 <= start_y <= y2 for x1, y1, x2, y2 in bbox_coords)
    if inside_bbox:
        print(f"Point aléatoire à l'intérieur de la bounding box. Ignoré.")
        continue

    # Agrandir la région à partir de ce point
    top, bottom, left, right = start_y, start_y, start_x, start_x

    # Expansion contrôlée dans les 4 directions
    expanded = True
    while expanded:
        expanded = False

        # Expansion vers le haut
        if top > 0 and not np.any(mask[top - 1, left:right + 1] == 255):
            top -= 1
            expanded = True

        # Expansion vers le bas
        if bottom < image_height - 1 and not np.any(mask[bottom + 1, left:right + 1] == 255):
            bottom += 1
            expanded = True

        # Expansion vers la gauche
        if left > 0 and not np.any(mask[top:bottom + 1, left - 1] == 255):
            left -= 1
            expanded = True

        # Expansion vers la droite
        if right < image_width - 1 and not np.any(mask[top:bottom + 1, right + 1] == 255):
            right += 1
            expanded = True

    # Calcul des dimensions
    region_width = right - left + 1
    region_height = bottom - top + 1
    region_area = region_width * region_height

    print(f"Région détectée : top={top}, bottom={bottom}, left={left}, right={right}")
    print(f"Dimensions de la région : largeur={region_width}, hauteur={region_height}, surface={region_area}")

    # Vérification de validité stricte
    if region_width <= 0 or region_height <= 0 or region_area < min_area:
        print(f"Région invalide détectée ({region_width}x{region_height}). Ignorée.")
        continue

    # Marquer la région comme traitée
    mask[top:bottom + 1, left:right + 1] = 255
    print(f"Pixels non masqués après mise à jour : {np.sum(mask == 0)}")

    # Sauvegarder cette région
    region = image[top:bottom + 1, left:right + 1]
    if region.size == 0:
        print("Erreur : Région extraite vide. Ignorée.")
        continue

    region_path = os.path.join(output_dir, f'background_region_{regions_extracted + 1}.jpg')
    cv2.imwrite(region_path, region)
    print(f"Région de fond sauvegardée dans : {region_path}")

    regions_extracted += 1

    # Debug : Sauvegarder le masque après chaque mise à jour
    debug_mask_path = os.path.join(output_dir, f'debug_mask_region_{regions_extracted}.jpg')
    debug_image = image.copy()
    debug_image[mask == 255] = [0, 0, 255]  # Masquer en rouge
    cv2.imwrite(debug_mask_path, debug_image)
    print(f"Masque après région sauvegardé : {debug_mask_path}")

print(f"Extraction terminée. {regions_extracted} régions de fond sauvegardées.")
