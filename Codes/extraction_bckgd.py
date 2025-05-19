#!/usr/bin/env python3
import os
import cv2
import numpy as np
import random

def extract_background(image_path, txt_file_path, output_dir, species,
                       min_region_size=0.01, min_pixel_threshold=0.01, max_ignored_regions=5):
    """
    Extrait les backgrounds d'une image en ignorant les régions masquées par les bounding boxes.
    Utilise deux masques : permanent_mask (zones des objets) et visited_mask (zones déjà extraites).

    Args:
        image_path (str): Chemin vers l'image.
        txt_file_path (str): Chemin vers le fichier TXT contenant les bounding boxes YOLO
                             (format : class, x_center, y_center, width, height, [conf]).
        output_dir (str): Dossier racine de sortie pour les backgrounds.
        species (str): Nom de l'espèce (sera utilisé pour créer un sous-dossier).
        min_region_size (float): Fraction minimale de la surface totale de l'image pour accepter une région.
        min_pixel_threshold (float): Fraction minimale de pixels non masqués restants pour continuer l'extraction.
        max_ignored_regions (int): Nombre maximal de régions invalides avant d'arrêter.

    Returns:
        List[str]: Liste des chemins vers les images de background extraites.
    """
    if not os.path.exists(image_path):
        print(f"Erreur : L'image {image_path} est introuvable.")
        return []
    if not os.path.exists(txt_file_path):
        print(f"Erreur : Le fichier texte {txt_file_path} est introuvable.")
        return []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Erreur : Impossible de charger l'image {image_path}")
        return []

    h, w, _ = image.shape
    print(f"Traitement de l'image : {image_path} (taille={w}x{h})")

    permanent_mask = np.zeros((h, w), dtype=np.uint8)
    visited_mask  = np.zeros((h, w), dtype=np.uint8)

    # Lecture et masquage des bounding boxes
    with open(txt_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                print("Ligne invalide dans le TXT :", line)
                continue
            class_id, x_c, y_c, bw, bh = map(float, parts[:5])
            cx = int(x_c * w)
            cy = int(y_c * h)
            bw = int(bw * w)
            bh = int(bh * h)
            # ajoute 15% de marge
            mw = int(bw * 0.15)
            mh = int(bh * 0.15)
            bw += 2*mw; bh += 2*mh
            x1 = max(0, cx - bw//2); y1 = max(0, cy - bh//2)
            x2 = min(w-1, cx + bw//2); y2 = min(h-1, cy + bh//2)
            print(f"  Bounding Box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            permanent_mask[y1:y2+1, x1:x2+1] = 255

    extracted_paths = []
    regions_extracted = 0
    ignored = 0
    min_area = int(min_region_size * w * h)

    # Extraction des régions non masquées
    while True:
        # indices des pixels libres
        free = np.argwhere((permanent_mask==0) & (visited_mask==0))
        if free.size == 0:
            break
        if free.shape[0] < min_pixel_threshold * w * h:
            print("Proportion de pixels non masqués trop faible. Arrêt.")
            break

        y0, x0 = random.choice(free)
        print(f"  Point de départ : ({x0},{y0})")
        top, bottom, left, right = y0, y0, x0, x0
        expanding = True
        while expanding:
            expanding = False
            if top>0    and permanent_mask[top-1,left:right+1].max()==0:
                top -=1; expanding=True
            if bottom<h-1 and permanent_mask[bottom+1,left:right+1].max()==0:
                bottom +=1; expanding=True
            if left>0   and permanent_mask[top:bottom+1,left-1].max()==0:
                left -=1; expanding=True
            if right<w-1 and permanent_mask[top:bottom+1,right+1].max()==0:
                right +=1; expanding=True

        rw, rh = right-left+1, bottom-top+1
        area = rw * rh
        print(f"  Région : top={top}, bottom={bottom}, left={left}, right={right}, surface={area}")

        if rw<=0 or rh<=0 or area < min_area:
            print(f"  Région invalide ({rw}x{rh}), ignorée.")
            ignored +=1
            if ignored >= max_ignored_regions:
                print("  Trop de régions ignorées. Arrêt.")
                break
            continue

        # marquer comme visité et sauvegarder
        visited_mask[top:bottom+1,left:right+1] = 255
        region = image[top:bottom+1, left:right+1]
        dest = os.path.join(output_dir, species)
        os.makedirs(dest, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_name = f"{base}_bg_{regions_extracted+1}.jpeg"
        out_path = os.path.join(dest, out_name)
        cv2.imwrite(out_path, region)
        print(f"  Region sauvegardée : {out_path}")
        extracted_paths.append(out_path)
        regions_extracted +=1

    print(f"Extraction terminée. {regions_extracted} régions sauvegardées.")
    return extracted_paths


if __name__ == "__main__":
    images_folder = "../Donnees/birds_dataset"
    bbox_folder   = "Dependances/runs/detect"
    output_dir    = "../Donnees/backgrounds_extracted"

    # Debug et création du dossier de sortie
    print("DEBUG ▶ CWD           =", os.getcwd())
    print("DEBUG ▶ images_folder =", images_folder, "→", os.path.isdir(images_folder))
    print("DEBUG ▶ bbox_folder   =", bbox_folder,   "→", os.path.isdir(bbox_folder))
    os.makedirs(output_dir, exist_ok=True)

    # Parcours des images
    for root, dirs, files in os.walk(images_folder):
        for fname in files:
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            image_path = os.path.join(root, fname)
            species    = os.path.basename(os.path.dirname(image_path))
            base       = os.path.splitext(fname)[0]
            txt_path   = os.path.join(bbox_folder, species, "labels", base + ".txt")

            if not os.path.isfile(txt_path):
                print(f"Erreur : Le fichier texte {txt_path} est introuvable.")
                print(f"0 fonds extraits pour {fname} ({species})")
                continue

            extracted = extract_background(
                image_path=image_path,
                txt_file_path=txt_path,
                output_dir=output_dir,
                species=species,
                min_region_size=0.01,
                min_pixel_threshold=0.01,
                max_ignored_regions=5
            )
            print(f"{len(extracted)} fonds extraits pour {fname} ({species})")
