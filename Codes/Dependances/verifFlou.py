#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import torch

def parse_args():
    p = argparse.ArgumentParser(
        description="Supprime les images floues détectées par un modèle YOLOv5 entraîné."
    )
    p.add_argument(
        "images_dir",
        help="Chemin vers le dossier racine des images (ex : /Donnees/birds_dataset)"
    )
    p.add_argument(
        "weights",
        help="Chemin vers le .pt du modèle flou (ex : Dependances/yolov5/best_flou.pt)"
    )
    p.add_argument(
        "--conf-thres", "-c",
        type=float,
        default=0.05,
        help="Seuil de confiance pour la détection (défaut 0.05)"
    )
    p.add_argument(
        "--iou-thres", "-i",
        type=float,
        default=0.45,
        help="Seuil IoU pour la NMS (défaut 0.45)"
    )
    return p.parse_args()

def main():
    args = parse_args()
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"[ERROR] Répertoire introuvable : {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Chargement du modèle flou YOLOv5 depuis {args.weights}…")
    # On utilise torch.hub en local pour ne pas télécharger le repo
    model = torch.hub.load(
        '/Codes/Dependances/yolov5',  # chemin local vers le dossier YOLOv5
        'custom',                     # type “custom” pour charger ton .pt perso
        path=args.weights,            # ton best_flou.pt
        source='local'                # FORCER l'usage du code local
    )
    model.conf = args.conf_thres     # seuil de confiance
    model.iou  = args.iou_thres      # seuil IoU

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    to_delete = []

    # balayage récursif
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in exts:
            continue

        # inference
        results = model(str(img_path))
        # si au moins une détection
        if results and results.xyxy[0].size(0) > 0:
            rel = img_path.relative_to(images_dir)
            n = int(results.xyxy[0].size(0))
            print(f"[BLUR] {rel} → {n} zone(s) floue(s) → suppression")
            to_delete.append(img_path)

    # suppression des fichiers détectés flous
    for p in to_delete:
        try:
            p.unlink()
        except Exception as e:
            print(f"[WARN] Impossible de supprimer {p}: {e}", file=sys.stderr)

    print(f"[INFO] {len(to_delete)} image(s) floue(s) supprimée(s).")

if __name__ == "__main__":
    main()
