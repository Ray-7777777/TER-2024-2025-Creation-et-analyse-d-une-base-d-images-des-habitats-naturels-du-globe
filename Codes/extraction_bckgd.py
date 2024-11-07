import os
import cv2

def extract_margin_areas(image_path, bbox_file, output_dir, species, margin=0.75):
    """
    Extrait les zones correspondant aux marges autour de la bounding box d'une image.
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
        for index, line in enumerate(f):
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

            # Déterminer les coordonnées
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(img_w, int(x_center + width / 2))
            y2 = min(img_h, int(y_center + height / 2))

            # Définir les zones des marges
            top_margin = image[max(0, y1 - margin_h):y1, x1:x2]  # Marge supérieure
            bottom_margin = image[y2:min(img_h, y2 + margin_h), x1:x2]  # Marge inférieure
            left_margin = image[y1:y2, max(0, x1 - margin_w):x1]  # Marge gauche
            right_margin = image[y1:y2, x2:min(img_w, x2 + margin_w)]  # Marge droite

            # Créer un sous-dossier pour l'espèce
            species_output_dir = os.path.join(output_dir, species)
            os.makedirs(species_output_dir, exist_ok=True)

            # Sauvegarder les images des marges seulement si elles ne sont pas vides
            if top_margin.size > 0:
                cv2.imwrite(os.path.join(species_output_dir, f'top_margin_{os.path.splitext(os.path.basename(image_path))[0]}_{index + 1}.jpg'), top_margin)
            if bottom_margin.size > 0:
                cv2.imwrite(os.path.join(species_output_dir, f'bottom_margin_{os.path.splitext(os.path.basename(image_path))[0]}_{index + 1}.jpg'), bottom_margin)
            if left_margin.size > 0:
                cv2.imwrite(os.path.join(species_output_dir, f'left_margin_{os.path.splitext(os.path.basename(image_path))[0]}_{index + 1}.jpg'), left_margin)
            if right_margin.size > 0:
                cv2.imwrite(os.path.join(species_output_dir, f'right_margin_{os.path.splitext(os.path.basename(image_path))[0]}_{index + 1}.jpg'), right_margin)

            print(f"Marges extraites et sauvegardées pour : {image_path}")

def process_all_images(parent_folder, output_folder, margin=0.1):
    """
    Parcourt toutes les images dans un dossier et extrait les marges autour de la boîte englobante.
    """
    os.makedirs(output_folder, exist_ok=True)

    for species in os.listdir(parent_folder):
        species_path = os.path.join(parent_folder, species)

        if os.path.isdir(species_path):
            train_path = os.path.join(species_path, 'train')
            bbox_folder = os.path.join(species_path, 'train', 'results', 'labels')  # Chemin vers le dossier des boîtes englobantes

            if os.path.exists(train_path):
                for file in os.listdir(train_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(train_path, file)
                        bbox_file = os.path.join(bbox_folder, f"{os.path.splitext(file)[0]}.txt")  # Assurer que le nom du fichier correspond
                        extract_margin_areas(image_path, bbox_file, output_folder, species)

# Exécution de l'extraction des marges pour toutes les images
parent_folder = r"C:\Users\alvin\Documents\M1 MIASHS\S1\TER\TER-2024-2025-Creation-et-analyse-d-une-base-d-images-des-habitats-naturels-du-globe\Donnees\birds_dataset"  # Dossier des images
output_folder = r"../Donnees/birds_dataset/Background_margins"  # Dossier de sortie pour les images de marges

# Processus d'extraction
process_all_images(parent_folder, output_folder, margin=0.75)
