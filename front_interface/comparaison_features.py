# comparaison_features.py

import os
import pandas as pd
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import itertools
import streamlit as st

# --- 1) Chargement et préparation du modèle RESNET50 (caché) ---
@st.cache_resource(show_spinner=False)
def get_extractor_and_transform():
    """
    Charge ResNet50, fusionne les layers pour l'inférence optimisée,
    crée un feature extractor sur layer4.2.conv2, et retourne
    l'extracteur + la transformation ImageNet.
    """
    # Charger le modèle préentraîné
    base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    base_model.eval()

    # Définir le noeud de sortie souhaité
    return_nodes = {'layer4.2.conv2': 'feat_map'}
    extractor = create_feature_extractor(base_model, return_nodes)

    # Transformation ImageNet standard
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return extractor, transform

# --- 2) Extraction d'un embedding pour une seule image ---
def extract_embedding(path: str, extractor, transform, pooling: str = 'avg') -> np.ndarray:
    """
    Lit l'image, applique la transformation, passe par l'extracteur
    et pool (avg ou max) pour obtenir un vecteur 512-D.
    """
    img = Image.open(path).convert('RGB')
    t = transform(img)
    if t.ndim == 3:
        t = t.unsqueeze(0)
    with torch.no_grad():
        feat_map = extractor(t)['feat_map'].squeeze(0)
    if pooling == 'avg':
        emb = feat_map.mean(dim=(1, 2))
    elif pooling == 'max':
        emb = feat_map.amax(dim=(1, 2))
    else:
        raise ValueError("pooling must be 'avg' or 'max'")
    return emb.cpu().numpy()

# --- 3) Analyse des tailles d'images (optionnel) ---
def analyze_image_sizes(folder: str) -> np.ndarray:
    """
    Parcourt un dossier et collecte les tailles (w,h) de chaque image.
    Affiche min/max/moyenne/std pour debug.
    """
    sizes = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg', 'jpeg', 'png')):
                try:
                    w, h = Image.open(os.path.join(root, f)).size
                    sizes.append((w, h))
                except:
                    pass
    arr = np.array(sizes)
    if arr.size == 0:
        print(f"[DEBUG] Aucune image trouvée dans {folder}")
        return arr
    for idx, name in [(0, 'Largeur'), (1, 'Hauteur')]:
        v = arr[:, idx]
        print(f"[DEBUG] {name} — min: {v.min()}, max: {v.max()}, "
              f"moy: {v.mean():.1f}, std: {v.std():.1f}")
    return arr

# --- 4) Sauvegarde des embeddings d'oiseaux ---
def save_embeddings(folder: str, out_csv: str, pooling: str = 'avg'):
    """
    Parcourt toutes les images dans `folder`, extrait un embedding
    pour chacune, et sauve un CSV prefix + 512 colonnes f000..f511.
    """
    extractor, transform = get_extractor_and_transform()
    cols = ['prefix'] + [f'f{i:03d}' for i in range(512)]
    rows = []
    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith(('.jpg', 'jpeg', 'png'))
    ]
    total = len(image_files)
    print(f"[DEBUG] {total} images à traiter dans {folder}…")
    for idx, path in enumerate(image_files, 1):
        emb = extract_embedding(path, extractor, transform, pooling)
        prefix = os.path.splitext(os.path.basename(path))[0]
        rows.append([prefix] + emb.tolist())
        if idx % 100 == 0 or idx == total:
            print(f"[DEBUG]  → {idx}/{total} images traitées")
    pd.DataFrame(rows, columns=cols).to_csv(out_csv, index=False)
    print(f"[DEBUG] Embeddings sauvegardés dans '{out_csv}'")

# --- 5) Sauvegarde des features conv2 pour backgrounds ---
def save_features_to_csv(folder_path: str, output_csv: str,
                         batch_size: int = 50, pooling: str = 'avg'):
    """
    Idem save_embeddings, mais découpe en batches et append au CSV.
    """
    extractor, transform = get_extractor_and_transform()
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"[DEBUG] Ancien fichier {output_csv} supprimé.")
    data = []
    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder_path)
        for f in files
        if f.lower().endswith(('.jpg','jpeg','png'))
    ]
    total = len(image_files)
    print(f"[DEBUG] {total} backgrounds à traiter dans {folder_path}…")
    for idx, path in enumerate(image_files, 1):
        emb = extract_embedding(path, extractor, transform, pooling)
        prefix = os.path.splitext(os.path.basename(path))[0]
        data.append([prefix] + emb.tolist())
        if idx % batch_size == 0 or idx == total:
            n_feats = len(data[0]) - 1
            cols = ['prefix'] + [f'f{i:03d}' for i in range(n_feats)]
            pd.DataFrame(data, columns=cols).to_csv(
                output_csv, mode='a',
                header=not os.path.exists(output_csv),
                index=False
            )
            print(f"[DEBUG] {idx}/{total} backgrounds sauvegardés dans {output_csv}")
            data = []
    print(f"[DEBUG] Extraction backgrounds terminée.")

# --- 6) Moyenne des features par prefixe ---
def mean_features_from_csv(input_csv: str, output_csv: str):
    """
    Charge le CSV, regroupe par 'prefix' (avant dernier '_'),
    calcule la moyenne et sauve dans output_csv.
    """
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"[ERROR] Fichier vide ou mal formaté : {input_csv}")
        return
    df['prefix'] = df['prefix'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    num_cols = df.select_dtypes(include=[np.number]).columns
    grouped = df[['prefix'] + list(num_cols)] \
                .groupby('prefix').mean().reset_index()
    mapping = { old: f'f{idx:03d}'
                for idx, old in enumerate(grouped.columns)
                if old != 'prefix' }
    grouped.rename(columns=mapping, inplace=True)
    grouped.to_csv(output_csv, index=False)
    print(f"[DEBUG] Features moyennes sauvegardées dans {output_csv}")

# --- 7) Calcul des distances, stats et matrice de confusion ---
def compute_distances_and_stats(
    birds_csv: str, backgrounds_csv: str,
    out_distances_csv: str, out_pair_stats_csv: str,
    out_conf_matrix_csv: str
):
    # Vérification fichiers
    for p in (birds_csv, backgrounds_csv):
        if not os.path.exists(p):
            print(f"[ERROR] Fichier non trouvé : {p}")
            return

    # Charger et merger
    df_img = pd.read_csv(birds_csv)
    df_bg  = pd.read_csv(backgrounds_csv)
    df_bg['species'] = df_bg['prefix'].str.rsplit('_', 1).str[0]

    # Distances individuelles
    feat_img = [c for c in df_img if c.startswith('f')]
    feat_bg  = [c for c in df_bg if c.startswith('f')]
    df_merged = pd.merge(df_img, df_bg, on='prefix',
                         suffixes=('_img','_bg'))
    distances = pairwise_distances(
        df_merged[[*feat_img]],
        df_merged[[*feat_bg]],
        metric='euclidean'
    ).diagonal()
    pd.DataFrame({
        'prefix': df_merged['prefix'],
        'euclidean_distance': distances
    }).to_csv(out_distances_csv, index=False)

    # Stats globales
    stats = {
        'mean':   distances.mean(),
        'median': np.median(distances),
        'q1':     np.percentile(distances, 25),
        'q3':     np.percentile(distances, 75),
        'var':    distances.var(),
        'std':    distances.std()
    }
    pd.DataFrame.from_dict(stats, orient='index', columns=['value']) \
      .to_csv(out_pair_stats_csv, index=True)

    # Matrice de confusion
    scaler = StandardScaler()
    df_bg_scaled = df_bg.copy()
    df_bg_scaled[feat_bg] = scaler.fit_transform(df_bg[feat_bg])
    species_list = sorted(df_bg_scaled['species'].unique())
    pair_results = []
    for s1, s2 in itertools.product(species_list, species_list):
        a1 = df_bg_scaled[df_bg_scaled['species']==s1][feat_bg].values
        a2 = df_bg_scaled[df_bg_scaled['species']==s2][feat_bg].values
        if len(a1)==0 or len(a2)==0:
            mean_dist = np.nan
        else:
            d = pairwise_distances(a1, a2, metric='euclidean').ravel()
            mean_dist = d.mean() if d.size>0 else np.nan
        pair_results.append((s1, s2, mean_dist))
    conf_mat = pd.DataFrame(
        pair_results,
        columns=['species_1','species_2','mean_dist']
    ).pivot(index='species_1', columns='species_2', values='mean_dist')
    conf_mat.to_csv(out_conf_matrix_csv)
    print("[DEBUG] Calcul des distances et matrice de confusion terminé.")

# --- 8) Wrapper global du pipeline ---
def run_pipeline(
    birds_folder: str,
    bg_folder: str,
    csv_birds: str,
    csv_bg_conv2: str,
    csv_bg_means: str,
    csv_distances: str,
    species_stats_csv: str,
    conf_matrix_csv: str
):
    """
    Exécute toutes les étapes du pipeline :
      1) analyse tailles
      2) embeddings oiseaux
      3) features backgrounds
      4) moyenne backgrounds
      5) distances, stats, confusion
    """
    os.makedirs(os.path.dirname(csv_birds), exist_ok=True)

    analyze_image_sizes(birds_folder)
    save_embeddings(birds_folder, csv_birds, pooling='avg')
    save_features_to_csv(bg_folder, csv_bg_conv2,
                         batch_size=50, pooling='avg')
    mean_features_from_csv(csv_bg_conv2, csv_bg_means)
    compute_distances_and_stats(
        csv_birds, csv_bg_means,
        csv_distances, species_stats_csv, conf_matrix_csv
    )

if __name__ == "__main__":
    # Exécution locale pour tests
    birds_folder = 'static/oiseaux_extraits'
    bg_folder    = 'static/background/unknown'
    out_dir      = 'static/similarites'
    run_pipeline(
        birds_folder,
        bg_folder,
        os.path.join(out_dir, 'birds_conv2_features.csv'),
        os.path.join(out_dir, 'backgrounds_conv2_features.csv'),
        os.path.join(out_dir, 'backgrounds_ResNet_features_means.csv'),
        os.path.join(out_dir, 'birds_background_euclidean_distance_results.csv'),
        os.path.join(out_dir, 'euclidean_distance_by_species_pair.csv'),
        os.path.join(out_dir, 'euclidean_mean_distance_confusion_matrix.csv')
    )
    print("Pipeline terminé.")
