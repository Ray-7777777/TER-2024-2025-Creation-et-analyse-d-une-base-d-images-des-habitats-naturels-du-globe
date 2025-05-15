import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
import os
import numpy as np
import pandas as pd

# 1. Charger ResNet50 et définir l’extracteur de features sur layer4[-1].conv2
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
base_model.eval()
return_nodes = {'layer4.2.conv2': 'feat_map'}
feature_extractor = create_feature_extractor(base_model, return_nodes)

# 2. Transformation (taille native)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Extraction + pooling
def extract_embedding(path, pooling='avg'):
    img = Image.open(path).convert('RGB')
    t = transform(img)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    with torch.no_grad():
        feat_map = feature_extractor(t)['feat_map'].squeeze(0)  # (512, H', W')
    if pooling == 'avg':
        emb = feat_map.mean(dim=(1,2))
    elif pooling == 'max':
        emb = feat_map.amax(dim=(1,2))
    else:
        raise ValueError("pooling doit être 'avg' ou 'max'")
    return emb.cpu().numpy()

# 4. Script principal adapté aux backgrounds
background_folder = '../../Donnees/background_extracted'
output_csv       = '../../Donnees/backgrounds_conv2_features.csv'

# Collecte des paths et noms uniques
image_entries = []
for root, _, files in os.walk(background_folder):
    for f in files:
        if not f.lower().endswith(('.jpg','jpeg','png')):
            continue
        full_path = os.path.join(root, f)
        # nom unique : chemin relatif sans extension, separators remplacés par '_'
        rel = os.path.relpath(full_path, background_folder)
        name = os.path.splitext(rel)[0].replace(os.sep, '_')
        image_entries.append((full_path, name))

total = len(image_entries)
print(f"{total} fonds à traiter…")

# Préparer les colonnes du DataFrame
cols = ['image_name'] + [f'f{i:03d}' for i in range(512)]
rows = []

try:
    for idx, (path, name) in enumerate(image_entries, start=1):
        emb = extract_embedding(path, pooling='avg')
        rows.append([name] + emb.tolist())
        if idx % 100 == 0 or idx == total:
            print(f"  → {idx}/{total} embeddings extraits")
except KeyboardInterrupt:
    print(f"\nInterrompu après {idx}/{total}. Sauvegarde partielle…")

# Sauvegarde
df = pd.DataFrame(rows, columns=cols)
df.to_csv(output_csv, index=False)
print(f"Embeddings sauvegardés ({len(rows)}/{total}) dans '{output_csv}'")
