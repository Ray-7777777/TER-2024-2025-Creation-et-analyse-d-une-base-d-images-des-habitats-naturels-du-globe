import torch
import pandas as pd
import numpy as np
import os
from torchvision import models, transforms
from PIL import Image

# Charger le modèle ResNet une fois pour éviter la surcharge
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.eval()

# Transformation d'image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Fonction pour extraire les caractéristiques
def extract_resnet_features(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Ajouter une dimension batch
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy()

# Fonction pour sauvegarder les caractéristiques dans un CSV
def save_features_to_csv(model, folder_path, output_csv, batch_size=50):
    data = []
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        try:
            features = extract_resnet_features(image_path, model)
            data.append([image_file] + features.flatten().tolist())
        except Exception as e:
            print(f"Erreur avec l'image {image_file}: {e}")
            continue

        # Sauvegarde partielle après chaque batch
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(image_files):
            df = pd.DataFrame(data)
            df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            data = []  # Réinitialiser le batch pour libérer de la mémoire

            print(f"Progression : {idx + 1}/{len(image_files)} images traitées et sauvegardées dans {output_csv}")

# Exemple d'utilisation
image_folder = '../../Donnees/bird'  # Remplacez par votre dossier d'images
output_csv = '../../Donnees/birds_ResNet_features_database.csv'  # Nom du fichier CSV de sortie

save_features_to_csv(model, image_folder, output_csv, batch_size=50)
