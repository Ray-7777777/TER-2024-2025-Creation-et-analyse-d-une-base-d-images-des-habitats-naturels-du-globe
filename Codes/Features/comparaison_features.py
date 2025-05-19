#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import itertools

# --- Chargement du modèle ResNet50 et création de l’extracteur sur layer4.2.conv2 ---
print("[DEBUG] Chargement du modèle ResNet50 avec extraction sur 'layer4.2.conv2'...")
base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
base_model.eval()
return_nodes = {'layer4.2.conv2': 'feat_map'}
feature_extractor = create_feature_extractor(base_model, return_nodes)
print("[DEBUG] Modèle et extracteur initialisés.")

# --- Transformation d'image (taille native) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- Analyse des tailles d’image ---
def analyze_image_sizes(folder):
    sizes = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.jpg','jpeg','png')):
                try:
                    w, h = Image.open(os.path.join(root, f)).size
                    sizes.append((w, h))
                except Exception:
                    pass
    arr = np.array(sizes)
    if arr.size == 0:
        print(f"Aucune image trouvée dans {folder}")
        return arr
    for idx, name in [(0, 'Largeur'), (1, 'Hauteur')]:
        v = arr[:, idx]
        print(f"{name} — min: {v.min()}, max: {v.max()}, moy: {v.mean():.1f}, std: {v.std():.1f}")
    return arr

# --- Extraction d’un embedding via pooling ---
def extract_embedding(path, pooling='avg'):
    img = Image.open(path).convert('RGB')
    tensor = transform(img)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        feat_map = feature_extractor(tensor)['feat_map'].squeeze(0)
    if pooling == 'avg':
        emb = feat_map.mean(dim=(1, 2))
    elif pooling == 'max':
        emb = feat_map.amax(dim=(1, 2))
    else:
        raise ValueError("pooling doit être 'avg' ou 'max'")
    return emb.cpu().numpy()

# --- Extraction et sauvegarde des embeddings d’oiseaux ---
def save_embeddings(folder, out_csv, pooling='avg'):
    cols = ['prefix'] + [f'f{i:03d}' for i in range(512)]
    rows = []
    image_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.lower().endswith(('.jpg', 'jpeg', 'png'))
    ]
    total = len(image_files)
    print(f"[DEBUG] {total} images à traiter dans {folder}...")
    for idx, path in enumerate(image_files, start=1):
        emb = extract_embedding(path, pooling)
        fname = os.path.basename(path)
        prefix = os.path.splitext(fname)[0].rsplit('_', 1)[0]
        rows.append([prefix] + emb.tolist())
        if idx % 100 == 0 or idx == total:
            print(f"  → {idx}/{total} images traitées")
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"[DEBUG] Embeddings sauvegardés ({len(rows)}/{total}) dans {out_csv}")

# --- Extraction des features conv2 pour les backgrounds ---
def save_features_to_csv(folder_path, output_csv, batch_size=50, pooling='avg'):
    if os.path.exists(output_csv):
        os.remove(output_csv)
        print(f"[DEBUG] Ancien fichier {output_csv} supprimé.")
    data, image_files = [], []
    for root, _, files in os.walk(folder_path):
        if '.ipynb_checkpoints' in root:
            continue
        for f in files:
            if f.lower().endswith(('.jpg','jpeg','png')):
                image_files.append(os.path.join(root, f))
    total = len(image_files)
    print(f"[DEBUG] {total} backgrounds à traiter dans {folder_path}...")
    for idx, image_path in enumerate(image_files, start=1):
        emb = extract_embedding(image_path, pooling)
        base = os.path.splitext(os.path.basename(image_path))[0]
        # Retirer le suffixe '_bg_N'
        prefix = base.split('_bg_')[0]
        data.append([prefix] + emb.tolist())
        if idx % batch_size == 0 or idx == total:
            n_feats = len(data[0]) - 1
            cols = ['prefix'] + [f'f{i:03d}' for i in range(n_feats)]
            pd.DataFrame(data, columns=cols).to_csv(
                output_csv, mode='a', header=not os.path.exists(output_csv), index=False
            )
            print(f"[DEBUG] {idx}/{total} backgrounds sauvegardés dans {output_csv}")
            data = []
    print("[DEBUG] Extraction backgrounds terminée.")

# --- Moyenne des features par préfixe ---
def mean_features_from_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    if df.empty:
        print("[ERREUR] Fichier vide ou mal formaté")
        return
    grouped = df.groupby('prefix').mean().reset_index()
    grouped.to_csv(output_csv, index=False)
    print(f"[DEBUG] Features moyennes sauvegardées dans {output_csv}")

# --- Fusion et calcul des distances + stats ---
def compute_distances_and_stats(
    birds_csv, backgrounds_csv,
    out_distances_csv, out_pair_stats_csv, out_conf_matrix_csv
):
    for path in (birds_csv, backgrounds_csv):
        if not os.path.exists(path):
            print(f"[ERREUR] Fichier non trouvé : {path}")
            return
    df_images = pd.read_csv(birds_csv)
    df_backgrounds = pd.read_csv(backgrounds_csv)
    df_merged = pd.merge(
        df_images, df_backgrounds, on='prefix', suffixes=('_image','_background')
    )
    if df_merged.empty:
        print("[ERREUR] Aucune correspondance entre images et backgrounds.")
        return
    feat_img = [c for c in df_merged.columns if c.endswith('_image')]
    feat_bg  = [c for c in df_merged.columns if c.endswith('_background')]
    distances = []
    for _, row in df_merged.iterrows():
        d = pairwise_distances(
            row[feat_img].values.reshape(1,-1),
            row[feat_bg].values.reshape(1,-1),
            metric='euclidean')[0,0]
        distances.append(d)
    df_dist = pd.DataFrame({'prefix':df_merged['prefix'],'euclidean_distance':distances})
    stats = { 'mean':df_dist['euclidean_distance'].mean(),
              'median':df_dist['euclidean_distance'].median(),
              'q1':np.percentile(df_dist['euclidean_distance'],25),
              'q3':np.percentile(df_dist['euclidean_distance'],75),
              'variance':df_dist['euclidean_distance'].var(),
              'std':df_dist['euclidean_distance'].std() }
    df_stats = pd.DataFrame({'prefix':list(stats.keys()),'euclidean_distance':list(stats.values())})
    df_dist.to_csv(out_distances_csv,index=False)
    df_stats.to_csv(out_distances_csv,mode='a',header=False,index=False)
    df_bg = pd.read_csv(backgrounds_csv)
    df_bg['species'] = df_bg['prefix']
    feat_cols = [c for c in df_bg.columns if c not in ['prefix','species']]
    scaler = StandardScaler(); df_bg[feat_cols]=scaler.fit_transform(df_bg[feat_cols])
    species_list = sorted(df_bg['species'].unique())
    results = []
    for sp1, sp2 in itertools.product(species_list, species_list):
        arr1 = df_bg[df_bg['species']==sp1][feat_cols].values
        arr2 = df_bg[df_bg['species']==sp2][feat_cols].values
        if sp1==sp2:
            mat = pairwise_distances(arr1,metric='euclidean') if len(arr1)>1 else np.array([])
            dists = mat[np.triu_indices_from(mat,k=1)] if mat.size else np.array([])
        else:
            dists = pairwise_distances(arr1,arr2,metric='euclidean').ravel() if arr1.size and arr2.size else np.array([])
        results.append({
            'species_1':sp1,'species_2':sp2,
            'n_samples_1':len(arr1),'n_samples_2':len(arr2),'num_pairs':len(dists),
            'mean_dist':dists.mean() if dists.size>0 else np.nan,
            'median_dist':np.median(dists) if dists.size>0 else np.nan,
            'q1_dist':np.percentile(dists,25) if dists.size>0 else np.nan,
            'q3_dist':np.percentile(dists,75) if dists.size>0 else np.nan,
            'var_dist':dists.var() if dists.size>0 else np.nan,
            'std_dist':dists.std() if dists.size>0 else np.nan})
    df_pairs = pd.DataFrame(results)
    df_pairs.to_csv(out_pair_stats_csv,index=False)
    conf_mat = df_pairs.pivot(index='species_1',columns='species_2',values='mean_dist')
    conf_mat.to_csv(out_conf_matrix_csv,index=True)
    print("[DEBUG] Calcul des distances et stats terminé.")

# --- Wrapper global intégrant toutes les étapes ---
def run_pipeline(
    birds_folder,bg_folder,
    birds_csv,bg_conv2_csv,bg_means_csv,
    distances_csv,species_stats_csv,conf_matrix_csv
):
    os.makedirs(os.path.dirname(birds_csv),exist_ok=True)
    analyze_image_sizes(birds_folder)
    save_embeddings(birds_folder,birds_csv)
    save_features_to_csv(bg_folder,bg_conv2_csv)
    mean_features_from_csv(bg_conv2_csv,bg_means_csv)
    compute_distances_and_stats(
        birds_csv,bg_means_csv,
        distances_csv,species_stats_csv,conf_matrix_csv)

if __name__ == "__main__":
    birds_folder = '/Donnees/oiseaux_extraits'
    bg_folder    = '/Donnees/backgrounds_extracted'
    out_dir      = '/Donnees/features'
    os.makedirs(out_dir, exist_ok=True)
    run_pipeline(
        birds_folder,
        bg_folder,
        os.path.join(out_dir,'birds_conv2_features.csv'),
        os.path.join(out_dir,'backgrounds_conv2_features.csv'),
        os.path.join(out_dir,'backgrounds_ResNet_features_means.csv'),
        os.path.join(out_dir,'birds_background_euclidean_distance_results.csv'),
        os.path.join(out_dir,'euclidean_distance_by_species_pair.csv'),
        os.path.join(out_dir,'euclidean_mean_distance_confusion_matrix.csv')
    )
    print("Pipeline terminé.")
