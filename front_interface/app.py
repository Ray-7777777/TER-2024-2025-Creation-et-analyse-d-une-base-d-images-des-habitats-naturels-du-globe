#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import streamlit as st
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from extraction_bckgd import extract_background

############################
# Conversion du checkpoint (pour supporter best.pt V8 → legacy)
############################
if not os.path.exists('best_legacy.pt'):
    # charge et exporte en format PyTorch binaire compatible
    YOLO('best.pt').export(format='pt', legacy=True)

############################
# Chargement du modèle
############################
model = YOLO('best_legacy.pt')

############################
# Fonctions utilitaires
############################

def allowed_file(filename):
    """Vérifie l'extension du fichier."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def detect_bird(image_path: str, conf: float = 0.5, iou: float = 0.45):
    """
    Détecte un oiseau dans l'image via Ultralytics.YOLO
    Renvoie (found_box, img_annotée (BGR), metrics list).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        st.error(f"Erreur de lecture de l'image : {image_path}")
        return False, None, []

    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
    r = results[0]  # batch size = 1
    # boxes en format Nx6: x1, y1, x2, y2, confidence, class
    boxes = r.boxes.cpu().numpy()

    metrics = []
    for x1, y1, x2, y2, score, cls in boxes:
        X1, Y1, X2, Y2 = map(int, (x1, y1, x2, y2))
        label = f"Oiseau ({score:.2f})"
        cv2.rectangle(img_bgr, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
        cv2.putText(img_bgr, label, (X1, Y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        metrics.append({
            "class": int(cls),
            "confidence": float(score),
            "bbox": [X1, Y1, X2, Y2],
            "label": label
        })

    found_box = len(boxes) > 0
    return found_box, img_bgr, metrics

def save_bbox_txt(image_path: str, metrics: list[dict]) -> str:
    """
    Sauvegarde les bbox dans un .txt (class confidence x1 y1 x2 y2).
    Retourne le chemin du .txt généré.
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    txt_path = os.path.join(UPLOAD_FOLDER, f"{base}.txt")
    with open(txt_path, 'w') as f:
        for m in metrics:
            bbox = " ".join(map(str, m["bbox"]))
            f.write(f"{m['class']} {m['confidence']:.2f} {bbox}\n")
    return txt_path

def sobel_variance(image_gray: np.ndarray) -> float:
    """Variance du filtre de Sobel (sharpness)."""
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gmag = cv2.magnitude(sobel_x, sobel_y)
    return gmag.var()

def calculate_fft(image_gray: np.ndarray) -> float:
    """Métrique FFT pour flou."""
    fft = np.fft.fftshift(np.fft.fft2(image_gray))
    mag = np.log(np.abs(fft) + 1)
    return mag.mean()

def is_blurry(image_path: str,
             threshold_sobel: float = 100,
             threshold_fft: float = 3) -> tuple[bool, float, float]:
    """Détermine si image est floue (Sobel + FFT)."""
    img = cv2.imread(image_path)
    if img is None:
        st.error(f"Impossible de charger {image_path}")
        return False, 0.0, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    var_sobel = sobel_variance(gray)
    fft_m = calculate_fft(gray)
    return (var_sobel < threshold_sobel and fft_m < threshold_fft,
            var_sobel, fft_m)

def add_text_with_black_background(image: np.ndarray,
                                   var_sobel: float,
                                   fft_metric: float,
                                   position=(10, 30),
                                   font_scale=1.0,
                                   thickness=1) -> np.ndarray:
    """Ajoute metrics en haut à gauche sur un fond noir."""
    text = f"Sobel: {var_sobel:.1f}, FFT: {fft_metric:.2f}"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - h - 5), (x + w + 5, y + 5), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)
    return image

def compute_distribution(image_name: str,
                         df: pd.DataFrame,
                         n_bg: int = 9,
                         n_null: int = 1000) -> tuple[float, np.ndarray]:
    """
    Pour une image, calcule :
      - d_intra : moyenne euclidienne à n_bg embeddings de même espèce
      - null_means : distribution nulle avec n_null tirages sur autres espèces
    """
    emb0 = df.loc[df.image_name == image_name, df.columns[1:]].values.astype(float)
    if emb0.shape[0] != 1:
        raise ValueError(f"Image {image_name} absente ou dupliquée")
    emb0 = emb0[0:1, :]

    species = image_name.split('_')[0]
    same = df[(df.image_name.str.startswith(f"{species}_")) & (df.image_name != image_name)]
    if len(same) < n_bg:
        raise ValueError(f"{len(same)} images trouvées pour {species}, besoin de {n_bg}")
    emb_same = same.sample(n_bg, random_state=0).iloc[:,1:].values
    d_intra = cdist(emb0, emb_same, 'euclidean').mean()

    others = df[~df.image_name.str.startswith(f"{species}_")]
    emb_others = others.iloc[:,1:].values
    if len(emb_others) < n_bg:
        raise ValueError(f"Pas assez d'images hors {species} pour n_bg={n_bg}")

    null_means = np.empty(n_null, dtype=float)
    for i in range(n_null):
        idx = np.random.choice(len(emb_others), size=n_bg, replace=False)
        null_means[i] = cdist(emb0, emb_others[idx], 'euclidean').mean()

    return d_intra, null_means

############################
# Configuration des dossiers
############################
UPLOAD_FOLDER       = 'uploads'
DETECTED_FOLDER     = 'static/results'
NOT_DETECTED_FOLDER = 'static/no_results'
BACKGROUND_FOLDER   = 'static/background'
sharp_folder        = 'static/classified/sharp'
blurry_folder       = 'static/classified/blurry'

for d in (UPLOAD_FOLDER, DETECTED_FOLDER, NOT_DETECTED_FOLDER,
          BACKGROUND_FOLDER, sharp_folder, blurry_folder):
    os.makedirs(d, exist_ok=True)

############################
# Streamlit UI
############################
if "images_info" not in st.session_state:
    st.session_state["images_info"] = []

tab_detection, tab_background, tab_classification, tab_distribution = st.tabs(
    ["Détection", "Background", "Classification", "Distribution"]
)

# ---- Onglet Détection ----
with tab_detection:
    st.header("Détection d'oiseaux")
    uploaded = st.file_uploader("Uploader des images", type=["png","jpg","jpeg"],
                                accept_multiple_files=True)
    if uploaded:
        st.session_state["images_info"].clear()
        for file in uploaded:
            fname = secure_filename(file.name)
            path = os.path.join(UPLOAD_FOLDER, fname)
            with open(path, 'wb') as f: f.write(file.getbuffer())

            found, annotated, metrics = detect_bird(path)
            if annotated is None: continue
            dest = DETECTED_FOLDER if found else NOT_DETECTED_FOLDER
            save_path = os.path.join(dest, fname)
            cv2.imwrite(save_path, annotated)
            bbox_txt = save_bbox_txt(path, metrics) if metrics else None

            st.session_state["images_info"].append({
                "filename": fname,
                "original_path": path,
                "annotated_path": save_path,
                "metrics": metrics,
                "bbox_txt": bbox_txt,
                "found_box": found
            })

        for info in st.session_state["images_info"]:
            st.subheader(info["filename"])
            st.image(info["annotated_path"], use_container_width=True)
            if info["metrics"]:
                dfm = pd.DataFrame(info["metrics"])
                st.dataframe(dfm)
                if info["bbox_txt"] and os.path.exists(info["bbox_txt"]):
                    with open(info["bbox_txt"]) as f:
                        st.code(f.read())
                    with open(info["bbox_txt"], "rb") as fb:
                        st.download_button("Télécharger coords", fb.read(),
                                           file_name=os.path.basename(info["bbox_txt"]))

# ---- Onglet Background ----
with tab_background:
    st.header("Extraction du Background")
    if not st.session_state["images_info"]:
        st.warning("Aucune image uploadée")
    else:
        names = [i["filename"] for i in st.session_state["images_info"]]
        sel = st.multiselect("Choisir images", names)
        if st.button("Extraire"):
            for info in st.session_state["images_info"]:
                if info["filename"] in sel:
                    txt = info.get("bbox_txt")
                    if not txt or not os.path.exists(txt):
                        st.error(f"Pas de TXT pour {info['filename']}")
                        continue
                    paths = extract_background(info["original_path"],
                                               txt,
                                               BACKGROUND_FOLDER,
                                               species="Unknown")
                    cols = st.columns(3)
                    for idx, p in enumerate(paths):
                        with cols[idx % 3]:
                            st.image(p, caption=os.path.basename(p),
                                     use_container_width=True)
                            with open(p, "rb") as fb:
                                st.download_button(f"Télécharger {os.path.basename(p)}",
                                                   fb.read(),
                                                   file_name=os.path.basename(p))

# ---- Onglet Classification ----
with tab_classification:
    st.header("Classification : Flou vs Nette")
    if not st.session_state["images_info"]:
        st.warning("Aucune image uploadée")
    else:
        sobel_th = st.slider("Seuil Sobel", 0, 3000, 100)
        fft_th   = st.slider("Seuil FFT",   0.0, 10.0, 3.0, 0.1)
        for info in st.session_state["images_info"]:
            st.subheader(info["filename"])
            blurry, vs, fm = is_blurry(info["original_path"], sobel_th, fft_th)
            img = cv2.imread(info["original_path"])
            ann = add_text_with_black_background(img.copy(), vs, fm)
            ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            st.image(ann_rgb, caption=("Floue" if blurry else "Nette"), use_container_width=True)
            st.write(f"Sobel: {vs:.1f}, FFT: {fm:.2f}")

# ---- Onglet Distribution nulle ----
with tab_distribution:
    st.header("Test de distribution nulle")
    @st.cache_data
    def load_emb(path): return pd.read_csv(path)

    df_emb = load_emb("../Donnees/birds_conv2_features.csv")
    species = sorted(df_emb['image_name'].str.split('_').str[0].unique())
    sp = st.selectbox("Espèce", species)
    imgs = df_emb[df_emb['image_name'].str.startswith(f"{sp}_")]['image_name']
    sel_img = st.selectbox("Image", imgs)
    n_bg   = st.number_input("n_bg",   1, 100, 9)
    n_null = st.number_input("n_null",100,5000,1000,100)

    if st.button("Lancer test"):
        with st.spinner("Calcul en cours…"):
            try:
                d_intra, null_means = compute_distribution(sel_img, df_emb, n_bg, n_null)
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.markdown(f"**d_intra :** {d_intra:.3f}")
        st.markdown(f"**moyenne nulle :** {null_means.mean():.3f}")
        low, high = np.percentile(null_means, 2.5), np.percentile(null_means, 97.5)
        st.markdown(f"**95% CI :** [{low:.3f}, {high:.3f}]")

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.hist(null_means, bins=30, edgecolor='black')
        ax.axvline(d_intra, color='red', linestyle='--', label=f"d_intra={d_intra:.2f}")
        ax.set_xlabel("Distance moyenne")
        ax.set_ylabel("Fréquence")
        ax.legend()
        st.pyplot(fig)
