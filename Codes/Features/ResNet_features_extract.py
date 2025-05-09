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

# On veut récupérer la feature map juste après conv2 du dernier bottleneck
return_nodes = {'layer4.2.conv2': 'feat_map'}
feature_extractor = create_feature_extractor(base_model, return_nodes)

# 2. Transformation (on conserve la taille native)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. Analyse des tailles d’image
def analyze_image_sizes(folder):
    sizes = []
    for _, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg','jpeg','png')):
                try:
                    w,h = Image.open(os.path.join(folder,f)).size
                    sizes.append((w,h))
                except:
                    pass
    arr = np.array(sizes)
    if arr.size == 0:
        print("Aucune image trouvée.")
        return arr
    for idx, name in [(0,'Largeur'), (1,'Hauteur')]:
        v = arr[:, idx]
        print(f"{name} — min: {v.min()}, max: {v.max()}, moy: {v.mean():.1f}, std: {v.std():.1f}")
    return arr

# 4. Extraction + pooling
def extract_embedding(path, pooling='avg'):
    img = Image.open(path).convert('RGB')
    t = transform(img)
    if t.ndim == 3:
        t = t.unsqueeze(0)  # (1, C, H, W)
    with torch.no_grad():
        feat_map = feature_extractor(t)['feat_map'].squeeze(0)  # (512, H', W')
    if pooling == 'avg':
        emb = feat_map.mean(dim=(1,2))
    elif pooling == 'max':
        emb = feat_map.amax(dim=(1,2))
    else:
        raise ValueError("pooling doit être 'avg' ou 'max'")
    return emb.cpu().numpy()  # (512,)

# 5. Boucle principale avec gestion KeyboardInterrupt et progression
def save_embeddings(folder, out_csv, pooling='avg'):
    cols = ['prefix'] + [f'f{i:03d}' for i in range(512)]
    rows = []
    image_files = [
        f for _, _, files in os.walk(folder)
          for f in files
          if f.lower().endswith(('.jpg','jpeg','png'))
    ]
    total = len(image_files)
    print(f"{total} images à traiter…")

    try:
        for idx, fname in enumerate(image_files, 1):
            path = os.path.join(folder, fname)
            emb  = extract_embedding(path, pooling)

            # ─── ici, on calcule le préfixe en 1 ligne ───
            prefix = os.path.splitext(fname)[0].rsplit('_', 1)[0]
            #    └──────────── stem sans extension ─────────┘
            #           └─ on enlève tout ce qui suit le dernier '_' ┘

            rows.append([prefix] + emb.tolist())

            if idx % 100 == 0 or idx == total:
                print(f"  → {idx}/{total} images traitées")

    except KeyboardInterrupt:
        print(f"\nInterrompu après {idx}/{total} images. Sauvegarde partielle…")

    # Sauvegarde finale
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"Embeddings sauvegardés ({len(rows)}/{total}) dans '{out_csv}'.")

# 6. Script principal
image_folder = '../../Donnees/oiseaux_extraits'
output_csv   = '../../Donnees/birds_conv2_features.csv'

print("=== Statistiques des tailles d'images ===")
analyze_image_sizes(image_folder)

print("\n=== Extraction et sauvegarde des embeddings ===")
save_embeddings(image_folder, output_csv, pooling='avg')
