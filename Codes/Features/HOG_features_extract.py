import cv2
import os
import pandas as pd
import numpy as np
from skimage.feature import hog

def load_image(image_path, target_size=(128, 128)):  
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur de chargement de l'image: {image_path}")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size)  
    return resized_image


def extract_hog_features(image):
    hog_features, _ = hog(image,
                          orientations=9,
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2),
                          block_norm='L2-Hys',
                          visualize=True,
                          transform_sqrt=True,
                          feature_vector=True)
    return hog_features


def save_features_to_csv(folder_path, output_csv):
    data = []
    images_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

    for image_path in images_paths:
        gray_image = load_image(image_path)
        if gray_image is None:
            continue 

        features = extract_hog_features(gray_image)
        if len(features) != 8100: 
            print(f"Caractéristiques inattendues pour {image_path}, longueur: {len(features)}")
            continue  
        data.append([os.path.basename(image_path)] + features.tolist())

    if data:
        df = pd.DataFrame(data)
        df.columns = ['image_name'] + [f'feature_{i}' for i in range(len(features))]
        df.to_csv(output_csv, index=False)
        print(f"Base de données de caractéristiques sauvegardée dans {output_csv}")
    else:
        print("Aucune caractéristique extraite. Vérifiez le contenu du dossier.")


image_folder = '../../Donnees/bird'  # Remplacez par votre dossier d'images
output_csv = '../../Donnees/birds_HOG_features_database.csv'  # Nom du fichier CSV de sortie

# Sauvegarder les caractéristiques des images dans un fichier CSV
save_features_to_csv(image_folder, output_csv)
