import os
import cv2
import numpy as np

def is_anthropic(image, threshold=0.10):
    """
    Détecte si une image contient une trace anthropique et si elle dépasse un certain pourcentage.
    """
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer un seuillage pour détecter les contours potentiels (simule une détection d'objets)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculer l'aire totale de l'image
    total_area = image.shape[0] * image.shape[1]
    anthropic_area = sum(cv2.contourArea(c) for c in contours)

    # Calculer le pourcentage d'aire anthropique
    anthropic_percentage = anthropic_area / total_area

    # Étiqueter selon le seuil
    return anthropic_percentage > threshold, anthropic_percentage

def process_backgrounds(backgrounds_folder):
    """
    Parcourt le dossier des backgrounds et détecte les traces anthropiques.
    """
    for species in os.listdir(backgrounds_folder):
        species_path = os.path.join(backgrounds_folder, species)

        if os.path.isdir(species_path):
            for file in os.listdir(species_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(species_path, file)
                    image = cv2.imread(image_path)

                    # Détecter si l'image est anthropique
                    is_anthropic_flag, percentage = is_anthropic(image)

                    if is_anthropic_flag:
                        print(f"{file} est étiqueté comme NON NATUREL (Trace anthropique: {percentage * 100:.2f}%)")
                    else:
                        print(f"{file} est étiqueté comme NATUREL")

# Exécution de la détection sur les backgrounds
backgrounds_folder = r"../Donnees/birds_dataset/background_margins"  # Chemin vers le dossier des backgrounds
process_backgrounds(backgrounds_folder)

