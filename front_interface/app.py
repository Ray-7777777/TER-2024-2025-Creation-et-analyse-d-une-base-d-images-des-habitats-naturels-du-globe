import sys, os

yolo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
sys.path.insert(0, yolo_root)

import cv2
import torch
import streamlit as st
from werkzeug.utils import secure_filename
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
import pandas as pd
import random
import numpy as np
import subprocess
import folium
import streamlit.components.v1 as components
import geopandas as gpd
from shapely.geometry import Point
from folium import Element

# Import de la fonction d'extraction du background
from extraction_bckgd import extract_background

from climatsEtHabitats import climats, ecoregions, avonet_habitats, ecosystemes, raster_classifications, parse_legend, compute_intersections, avonet_habitats

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), 'extraire_oiseaux.sh')

############################
# Fonctions de détection
############################

def allowed_file(filename):
    """Vérifie si le fichier a une extension valide."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def detect_bird(image_path):
    """Détecte un oiseau dans une image et renvoie l'image annotée et les métriques."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        st.error(f"Erreur de lecture de l'image : {image_path}")
        return False, None, []
    
    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))
    
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    
    results = model(img_tensor)
    preds = non_max_suppression(results, conf_thres=0.5, iou_thres=0.45)
    detections = preds[0]
    
    metrics = []
    scale_x = w_orig / 640.0
    scale_y = h_orig / 640.0
    found_box = False
    
    for *xyxy, conf, cls_idx in detections:
        if conf >= 0.5:
            found_box = True
            x1, y1, x2, y2 = [coord.item() for coord in xyxy]
            X1 = int(x1 * scale_x)
            Y1 = int(y1 * scale_y)
            X2 = int(x2 * scale_x)
            Y2 = int(y2 * scale_y)
            label = f"Oiseau ({conf:.2f})"
            cv2.rectangle(img_bgr, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (X1, Y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            metrics.append({
                "class": int(cls_idx),
                "confidence": conf.item(),
                "bbox": [X1, Y1, X2, Y2],
                "label": label
            })
    return found_box, img_bgr, metrics

def save_bbox_txt(image_path, metrics):
    """Sauvegarde les coordonnées des bounding boxes dans un fichier TXT au format : class, confiance, x1, y1, x2, y2."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_file_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.txt")
    txt_data = ""
    for m in metrics:
        bbox_str = " ".join(str(x) for x in m["bbox"])
        txt_data += f"{m['class']} {m['confidence']:.2f} {bbox_str}\n"
    with open(txt_file_path, 'w') as txt_file:
        txt_file.write(txt_data)
    return txt_file_path

############################
# Fonctions de classification (flou vs nette)
############################

def sobel_variance(image_gray):
    """Calcule la variance du filtre de Sobel pour détecter la netteté de l'image."""
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    # Pour les bords diagonaux :
    sobel_diag_45 = cv2.filter2D(image_gray, cv2.CV_64F, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    sobel_diag_135 = cv2.filter2D(image_gray, cv2.CV_64F, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
    gradient_magnitude_x = cv2.magnitude(sobel_x, sobel_y)
    gradient_magnitude_diag = cv2.magnitude(sobel_diag_45, sobel_diag_135)
    variance_x = gradient_magnitude_x.var()
    variance_diag = gradient_magnitude_diag.var()
    return (variance_x + variance_diag) / 2

def calculate_fft(image_gray):
    """Applique la transformée de Fourier pour mesurer le flou."""
    fft_image = np.fft.fft2(image_gray)
    fft_shifted = np.fft.fftshift(fft_image)
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    return np.mean(magnitude_spectrum)

def is_blurry(image_path, threshold_sobel=100, threshold_fft=3):
    """Détermine si une image est floue selon les seuils Sobel et FFT."""
    image = cv2.imread(image_path)
    if image is None:
        st.error(f"Impossible de charger l'image {image_path}")
        return False, 0, 0
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var_sobel = sobel_variance(image_gray)
    fft_metric = calculate_fft(image_gray)
    blurry = var_sobel < threshold_sobel and fft_metric < threshold_fft
    return blurry, var_sobel, fft_metric

def add_text_with_black_background(image, var_sobel, fft_metric, position=(10,30), font_scale=1, thickness=1):
    """Ajoute le texte des métriques sur l'image."""
    text = f'Sobel Variance: {var_sobel:.2f}, FFT: {fft_metric:.2f}'
    thickness = int(thickness)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (position[0], position[1] - text_height - 10),
                  (position[0] + text_width + 5, position[1] + 10), (0, 0, 0), cv2.FILLED)
    cv2.putText(image, text, (position[0], position[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
    return image

def crop_birds_from_labels(label_dir, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for txt_path in glob.glob(os.path.join(label_dir, "*.txt")):
        base = os.path.splitext(os.path.basename(txt_path))[0]
        for ext in (".jpg", ".jpeg", ".png"):
            img_path = os.path.join(image_dir, base + ext)
            if os.path.exists(img_path):
                break
        else:
            st.warning(f"Aucune image trouvée pour {base}")
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        with open(txt_path) as f:
            for i, line in enumerate(f):
                cls, xc, yc, bw, bh = map(float, line.split())
                x_min = int((xc - bw/2) * w)
                y_min = int((yc - bh/2) * h)
                x_max = x_min + int(bw * w)
                y_max = y_min + int(bh * h)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)
                crop = img[y_min:y_max, x_min:x_max]
                out_path = os.path.join(output_dir, f"{base}_{i}.jpg")
                cv2.imwrite(out_path, crop)
    st.success(f"✅ Recadrage terminé ! Fichiers dans {output_dir}")

############################
# Configuration des dossiers et chargement du modèle
############################

UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'static/results'
NOT_DETECTED_FOLDER = 'static/no_results'
BACKGROUND_FOLDER = os.path.join('static', 'background')
CROP_OUTPUT_DIR = os.path.join('static', 'oiseaux_extraits')
sharp_folder = os.path.join('static', 'classified', 'sharp')
blurry_folder = os.path.join('static', 'classified', 'blurry')
geo_folder = os.path.join('static', 'geo')

for folder in [UPLOAD_FOLDER, DETECTED_FOLDER, NOT_DETECTED_FOLDER, BACKGROUND_FOLDER, sharp_folder, blurry_folder]:
    os.makedirs(folder, exist_ok=True)

model = DetectMultiBackend('./best.pt', device=torch.device('cpu'))

############################
# Interface Streamlit avec onglets
############################

# Initialisation de la session pour stocker les infos
if "images_info" not in st.session_state:
    st.session_state["images_info"] = []

tab_detection, tab_background, tab_classification, tab_similarites, tab_cartes = st.tabs(["Détection", "Background", "Classification", "Similarités", "Données Géographiques"])

# ---------- Onglet Détection ----------
with tab_detection:
    st.header("Détection d'oiseaux")
    uploaded_files = st.file_uploader(
        "Téléverser des images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state["images_info"] = []  # Réinitialisation
        # Traitement de chaque fichier uploadé
        for file in uploaded_files:
            filename = secure_filename(file.name)
            original_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(original_path, 'wb') as f:
                f.write(file.getbuffer())

            # Détection YOLO
            found_box, annotated_img, metrics = detect_bird(original_path)
            if annotated_img is None:
                st.error(f"Erreur de détection pour {filename}")
                continue

            # Sauvegarde de l'image annotée
            save_dir = DETECTED_FOLDER if found_box else NOT_DETECTED_FOLDER
            annotated_path = os.path.join(save_dir, filename)
            cv2.imwrite(annotated_path, annotated_img)

            # Sauvegarde du fichier TXT de bounding box
            bbox_txt = save_bbox_txt(original_path, metrics) if metrics else None

            # Stocke les infos
            st.session_state["images_info"].append({
                "filename": filename,
                "original_path": original_path,
                "annotated_path": annotated_path,
                "bbox_txt": bbox_txt,
                "metrics": metrics
            })

        # Recadrage direct des oiseaux à partir des fichiers TXT
        os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)
        for info in st.session_state["images_info"]:
            txt_path = info.get("bbox_txt")
            if not txt_path or not os.path.exists(txt_path):
                continue
            img = cv2.imread(info["original_path"])
            h, w = img.shape[:2]
            with open(txt_path) as f:
                for i, line in enumerate(f):
                    parts = line.strip().split()
                    # Format attendu: class confidence x1 y1 x2 y2
                    if len(parts) != 6:
                        continue
                    _, _, x1, y1, x2, y2 = map(float, parts)
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = img[y1:y2, x1:x2]
                    out_name = f"{info['filename']}_{i}.jpg"
                    out_path = os.path.join(CROP_OUTPUT_DIR, out_name)
                    cv2.imwrite(out_path, crop)
        st.success(f"✅ Recadrage terminé ! Fichiers dans {CROP_OUTPUT_DIR}")

        # Affichage des images recadrées
        recrops = []
        if os.path.exists(CROP_OUTPUT_DIR):
            recrops = [os.path.join(CROP_OUTPUT_DIR, f)
                       for f in os.listdir(CROP_OUTPUT_DIR)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if recrops:
            st.subheader("Images recadrées")
            cols = st.columns(4)
            for idx, img_path in enumerate(recrops):
                with cols[idx % 4]:
                    st.image(img_path, use_container_width=True)
                    # Bouton de téléchargement du crop avec clé unique
                    with open(img_path, 'rb') as f:
                        st.download_button(
                            label="Télécharger",
                            data=f,
                            file_name=os.path.basename(img_path),
                            mime="image/jpeg",
                            key=f"download_crop_{os.path.basename(img_path)}"
                        )
        else:
            st.info("Aucune image recadrée trouvée.")

        # Affichage des résultats de détection
        st.subheader("Images détectées")
        for info in st.session_state["images_info"]:
            st.image(info["annotated_path"], caption=info["filename"], use_container_width=True)
    else:
        st.info("Aucune image téléversée.")

# ---------- Onglet Background ----------
with tab_background:
    st.header("Extraction du Background")
    if not st.session_state["images_info"]:
        st.warning("Aucune image n'a été téléversée dans l'onglet Détection.")
    else:
        filenames = [info["filename"] for info in st.session_state["images_info"]]
        selected_images = st.multiselect("Sélectionnez les images pour extraire le background", filenames)
        if st.button("Extraire le background"):
            if not selected_images:
                st.warning("Veuillez sélectionner au moins une image.")
            else:
                for info in st.session_state["images_info"]:
                    if info["filename"] in selected_images:
                        txt_file_path = info.get("bbox_txt")
                        if not txt_file_path or not os.path.exists(txt_file_path):
                            st.error(f"Le fichier TXT pour {info['filename']} est introuvable.")
                            continue
                        species = "Unknown"
                        extracted_paths = extract_background(info["original_path"], txt_file_path, BACKGROUND_FOLDER, species)
                        if extracted_paths:
                            num_cols = 3
                            cols = st.columns(num_cols)
                            for i, path in enumerate(extracted_paths):
                                with cols[i % num_cols]:
                                    st.image(path, caption=f"Background extrait : {os.path.basename(path)}", use_container_width=True)
                                    with open(path, "rb") as file:
                                        st.download_button(
                                            label=f"Télécharger {os.path.basename(path)}",
                                            data=file,
                                            file_name=os.path.basename(path),
                                            mime="image/jpeg"
                                        )
                        else:
                            st.error(f"Aucun background extrait pour {info['filename']}.")

# ---------- Onglet Classification (Flou vs Nette) ----------
with tab_classification:
    st.header("Classification : Flou ou Nette")
    if not st.session_state["images_info"]:
        st.warning("Aucune image n'a été téléversée dans l'onglet Détection.")
    else:
        # Sliders pour ajuster les seuils
        threshold_sobel = st.slider("Seuil Sobel", min_value=0, max_value=3000, value=100, step=1)
        threshold_fft = st.slider("Seuil FFT", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        
        st.write("Les images seront analysées avec ces seuils pour déterminer si elles sont floues ou nettes.")
        
        for info in st.session_state["images_info"]:
            st.subheader(f"Classification pour : {info['filename']}")
            image_path = info["original_path"]
            blurry, sobel_val, fft_val = is_blurry(image_path, threshold_sobel, threshold_fft)
            
            # Charger l'image pour l'annotation
            image = cv2.imread(image_path)
            if image is None:
                st.error(f"Erreur pour l'image {info['filename']}")
                continue
            # Ajouter le texte des métriques sur l'image
            annotated = add_text_with_black_background(image.copy(), sobel_val, fft_val)
            
            classification = "Floue" if blurry else "Nette"
            st.write(f"**Résultat :** {classification}")
            st.write(f"Sobel Variance: {sobel_val:.2f}, FFT Metric: {fft_val:.2f}")
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption=f"{info['filename']} - {classification}", use_container_width=True)

# ---------- Onglet Similarités ----------
with tab_similarites:
    st.header("Calcul des similarités")

    if st.button("📊 Calculer les similarités"):
        from comparaison_features import run_pipeline
        with st.spinner("Extraction des features et calcul des similarités…"):
            # 1) dossier des images oiseaux et backgrounds
            birds_folder = os.path.join("static", "oiseaux_extraits")
            bg_folder    = os.path.join("static", "background", "unknown")

            # 2) dossier de sortie pour les CSV
            out_folder = os.path.join("static", "similarites")
            os.makedirs(out_folder, exist_ok=True)

            # 3) chemins EXACTS vers les CSV qui seront créés
            csv_birds      = os.path.join(out_folder, "birds_conv2_features.csv")
            csv_bg_conv2   = os.path.join(out_folder, "backgrounds_conv2_features.csv")
            csv_bg_means   = os.path.join(out_folder, "backgrounds_ResNet_features_means.csv")
            csv_distances  = os.path.join(out_folder, "birds_background_euclidean_distance_results.csv")
            csv_pair_stats = os.path.join(out_folder, "euclidean_distance_by_species_pair.csv")
            csv_conf_mat   = os.path.join(out_folder, "euclidean_mean_distance_confusion_matrix.csv")

            # 4) Lancement du pipeline COMPLET (toutes les étapes)
            run_pipeline(
                birds_folder,
                bg_folder,
                csv_birds,
                csv_bg_conv2,
                csv_bg_means,
                csv_distances,
                csv_pair_stats,
                csv_conf_mat
            )

        # 6) Affichage de la matrice de confusion
        import pandas as pd
        import matplotlib.pyplot as plt

        if os.path.exists(csv_conf_mat):
            # a) Chargement du CSV (première colonne en index)
            df_conf = pd.read_csv(csv_conf_mat, index_col=0)

            # b) Affichage interactif
            st.markdown("### Matrice de confusion (distance Euclidienne moyenne)")
            st.dataframe(df_conf)

            # c) Heatmap Matplotlib
            fig, ax = plt.subplots()
            im = ax.imshow(df_conf.values, aspect='auto')
            # étiquettes axes
            ax.set_xticks(range(len(df_conf.columns)))
            ax.set_xticklabels(df_conf.columns, rotation=90)
            ax.set_yticks(range(len(df_conf.index)))
            ax.set_yticklabels(df_conf.index)
            ax.set_xlabel("Espèce background")
            ax.set_ylabel("Espèce oiseau")
            fig.colorbar(im, ax=ax, label="Distance moyenne")

            st.pyplot(fig)
        else:
            st.error(f"Fichier non trouvé : {os.path.basename(csv_conf_mat)}")


ECOREGIONS_SHP      = "../Donnees/Ecoregions/wwf_terr_ecos.shp"
CLIMATES_SHP        = "../Donnees/climates/climates.shp"
ECOSYS_RASTER_DIR   = "../Donnees/Ecosystemes/raster/"
RASTER_PATH  = "../Donnees/Raster_habitats/Biome_Inventory_RasterStack.tif"
BANDS        = [26, 9, 18, 23]
BAND_NAMES   = ["Leemans", "Higgins", "Friedl", "Olson"]
LEGEND_PATH = "../Donnees/Raster_habitats/Biome_Inventory_Legends.txt"
AVONET_FILE       = "../Donnees/avonet/AVONET2_eBird.xlsx"

# ---------- Onglet Cartes ----------
with tab_cartes:
    st.header("Projeter vos coordonnées par modalité")

    # --- Option : extractions spatiales (écosystèmes + biomes raster) ---
    include_spatial = st.checkbox(
        "Inclure l'extraction des écosystèmes et des biomes (très long)",
        value=True
    )

    # --- 1) Téléversement ---
    uploaded_files = st.file_uploader(
        "Téléchargez vos fichiers (latitude,longitude)",
        type=["csv", "txt"], accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("📄 Importez au moins un fichier pour démarrer.")
        st.stop()

    # --- 2) Lecture des coordonnées ---
    species_coords = {}
    for f in uploaded_files:
        raw = os.path.splitext(f.name)[0]
        try:
            df = pd.read_csv(f)
            pts = list(zip(df["latitude"], df["longitude"]))
            if pts:
                species_coords[raw] = pts
        except Exception:
            st.warning(f"Impossible de lire {f.name}")
    if not species_coords:
        st.error("Aucune coordonnée valide trouvée.")
        st.stop()

    # --- 3) Pré-chargement shapefiles (toujours) et légende raster (facultatif) ---
    gdf_clim = gpd.read_file(CLIMATES_SHP).to_crs(epsg=4326)
    gdf_eco  = gpd.read_file(ECOREGIONS_SHP).to_crs(epsg=4326)

    legend_map = {}
    if include_spatial:
        raw_legend = parse_legend(LEGEND_PATH, BANDS)
        legend_map = {
            str(int(band)): { str(int(k)): v for k, v in band_map.items() }
            for band, band_map in raw_legend.items()
        }
        import rasterio
        from pyproj import Transformer
        src = rasterio.open(RASTER_PATH)
        transformer = Transformer.from_crs("EPSG:4326", src.crs.to_string(), always_xy=True)

    # --- 4) Spatial join Climat/Ecorégions ---
    inter_map = compute_intersections(species_coords, CLIMATES_SHP, ECOREGIONS_SHP)

    # --- 5) Construction du tableau récapitulatif ---
    rows = []
    for raw, coords in species_coords.items():
        display = " ".join(raw.split("_")[:2])

        # a) AVONET
        av_str = ""
        try:
            avonet_habitats(AVONET_FILE, "AVONET2_eBird", display, geo_folder)
        except Exception:
            pass
        hab_file = os.path.join(geo_folder, display, "habitats_data.txt")
        if os.path.exists(hab_file):
            with open(hab_file, encoding="utf-8") as hf:
                av_str = "; ".join([L.strip() for L in hf if L.strip()])

        # b) Écosystèmes et Raster (facultatif)
        ecosys_map = {}
        raster_map = {}
        if include_spatial:
            sd = os.path.join(geo_folder, raw)
            os.makedirs(sd, exist_ok=True)
            ecosystemes(coords, ECOSYS_RASTER_DIR, raw, geo_folder)
            eco_txt = os.path.join(sd, "ecosystemes_data.txt")
            if os.path.exists(eco_txt):
                for L in open(eco_txt, encoding="utf-8"):
                    parts = L.strip().split("):", 1)
                    if len(parts) == 2:
                        cp, desc = parts
                        cp = cp.replace("Coordinates (", "")
                        lat, lon = map(float, cp.split(","))
                        ecosys_map[(lat, lon)] = desc.strip()

            # Extraction inline du raster
            for lat, lon in coords:
                x, y = transformer.transform(lon, lat)
                try:
                    row_idx, col_idx = src.index(x, y)
                except Exception:
                    raster_map[(lat, lon)] = {bn: None for bn in BAND_NAMES}
                    continue

                vals = {}
                for band, name in zip(BANDS, BAND_NAMES):
                    try:
                        raw_val = src.read(band)[row_idx, col_idx]
                        key = str(int(raw_val))
                        cls = legend_map.get(str(band), {}).get(key)
                    except Exception:
                        cls = None
                    vals[name] = cls
                raster_map[(lat, lon)] = vals

        # c) Remplissage des lignes
        for lat, lon in coords:
            inter = inter_map.get((raw, lat, lon), {})
            row = {
                "Espèce":         display,
                "Latitude":       lat,
                "Longitude":      lon,
                "Climat":         inter.get("Climat"),
                "Sub-climat":     inter.get("Sub-climat"),
                "Sub-sub-climat": inter.get("Sub-sub-climat"),
                "Ecorégion":      inter.get("Ecorégion"),
                "Ecosystème":     ecosys_map.get((lat, lon)),
                "AVONET":         av_str
            }
            if include_spatial:
                row.update(raster_map.get((lat, lon), {bn: None for bn in BAND_NAMES}))
            rows.append(row)

    df_summary = pd.DataFrame(rows)
    st.markdown("### Tableau récapitulatif de toutes les observations")
    st.dataframe(df_summary)

    # --- 6) Menus déroulants ---
    names  = ["Toutes espèces"] + [f.name for f in uploaded_files]
    choice = st.selectbox("Fichier à afficher", names)
    modality = st.selectbox("Modalité", ["Climats", "Ecorégions", "Écosystèmes"])
    if modality == "Climats":
        climat_level = st.selectbox("Niveau de détail", ["Climat", "Sub-climat", "Sub-sub-climat"])

    # --- 7) Affichage de la carte ---
    if st.button("Afficher la carte"):
        if choice != "Toutes espèces":
            raw2    = os.path.splitext(choice)[0]
            display = " ".join(raw2.split("_")[:2])
            df_plot = df_summary[df_summary["Espèce"] == display]
        else:
            df_plot = df_summary

        if df_plot.empty:
            st.error("Pas d'observations pour la sélection.")
            st.stop()

        m = folium.Map(
            location=[df_plot["Latitude"].mean(), df_plot["Longitude"].mean()],
            zoom_start=4
        )

        # superposition shapefile
        if modality in ("Climats", "Ecorégions"):
            shp = CLIMATES_SHP if modality == "Climats" else ECOREGIONS_SHP
            gdf = gpd.read_file(shp).to_crs(epsg=4326)
            if modality == "Climats":
                col = {"Climat": "CLIMATE", "Sub-climat": "SUB-CLIMAT", "Sub-sub-climat": "SUB-SUB-CL"}[climat_level]
            else:
                col = "ECO_NAME"
            cmap = {c: f"#{random.randint(0,0xFFFFFF):06x}" for c in gdf[col].unique()}
            folium.GeoJson(
                gdf.to_json(),
                style_function=lambda feat: {
                    "fillColor":   cmap.get(feat["properties"][col], "#CCCCCC"),
                    "color":       "#444444",
                    "weight":      1,
                    "fillOpacity": 0.5
                }
            ).add_to(m)

        # points et tooltips
        pcol = {sp: f"#{random.randint(0,0xFFFFFF):06x}" for sp in df_plot["Espèce"].unique()}
        for _, r in df_plot.iterrows():
            if modality == "Climats":
                tip = (
                    f"Climat : {r['Climat']}<br>"
                    f"Sub-climat : {r['Sub-climat']}<br>"
                    f"Sub-sub-climat : {r['Sub-sub-climat']}<br>"
                    f"Ecorégion : {r['Ecorégion']}<br>"
                    f"Ecosystème : {r['Ecosystème']}"
                )
            elif modality == "Ecorégions":
                tip = (
                    f"Ecorégion : {r['Ecorégion']}<br>"
                    f"Ecosystème : {r['Ecosystème']}"
                )
            else:
                tip = f"Ecosystème : {r['Ecosystème']}"

            if include_spatial:
                for bn in BAND_NAMES:
                    tip += f"<br>{bn} : {r.get(bn)}"

            tip += f"<br>AVONET : {r['AVONET']}"
            clr = pcol[r["Espèce"]]
            folium.CircleMarker(
                [r["Latitude"], r["Longitude"]],
                radius=6,
                color=clr,
                fill=True,
                fill_color=clr,
                fill_opacity=0.8,
                tooltip=tip,
                parse_html=True
            ).add_to(m)

        # légende des espèces
        legend_html = (
            '<div style="position: fixed; bottom:10px; left:10px; '
            'background:white; padding:10px; border:2px solid grey; '
            'z-index:9999; font-size:14px;"><b>Légende des espèces</b><br>'
        )
        for sp, clr in pcol.items():
            legend_html += (
                f'<i style="background:{clr}; width:10px; height:10px; '
                'display:inline-block; border-radius:50%; margin-right:5px;"></i>'
                f'{sp}<br>'
            )
        legend_html += '</div>'

        m.get_root().html.add_child(Element(legend_html))
        components.html(m._repr_html_(), height=600, scrolling=True)
