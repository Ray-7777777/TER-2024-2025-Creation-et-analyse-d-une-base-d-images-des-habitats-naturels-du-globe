import sys, os

import pathlib
# Ça redirige toute instantiation de PosixPath vers pathlib.Path (WindowsPath)
pathlib.PosixPath = pathlib.Path

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
from PIL import Image
from ultralytics import YOLO

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

def detect_bird(image_path, model):
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
    
    # Déplacer le tenseur vers le même device que le modèle
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    results = model(img_tensor)
    preds = non_max_suppression(results, conf_thres=0.5, iou_thres=0.45)
    detections = preds[0].cpu()  # Ramener sur CPU pour le post-traitement
    
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
    """Sauvegarde les coordonnées des bounding boxes dans un fichier TXT."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_file_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.txt")
    txt_data = ""
    for m in metrics:
        x1, y1, x2, y2 = m["bbox"]
        txt_data += f"{m['class']} {m['confidence']:.2f} {x1} {y1} {x2} {y2}\n"
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

# Fonction pour installer YOLOv12
@st.cache_resource
def setup_yolov12():
    try:
        import subprocess
        weights_url = "https://github.com/WongKinYiu/yolov12/releases/download/v1.0/yolov12s.pt"
        subprocess.run(["wget", weights_url, "-O", "best_human.pt"], check=True)
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'installation de YOLOv12: {str(e)}")
        return False

# Initialisation des modèles avec gestion d'erreur
@st.cache_resource(show_spinner=False)
def load_models():
    if not os.path.exists('./best_human.pt'):
        if not setup_yolov12():
            return None
            
    models = {}
    try:
        # YOLOv12 avec l'API ultralytics
        models['human_model'] = YOLO('best_human.pt')
        
        # YOLOv5 pour oiseaux (inchangé)
        models['bird_model'] = DetectMultiBackend('./best.pt', device=torch.device('cpu'))
        models['bird_model'].warmup()
        models['bird_model'].eval()
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {str(e)}")
        return None
        
    return models

# Initialisation des variables de session
if 'models' not in st.session_state:
    st.session_state.models = load_models()

if 'images_info' not in st.session_state:
    st.session_state.images_info = []

# Vérification des modèles
if st.session_state.models is None:
    st.error("⚠️ Erreur lors du chargement des modèles. L'application pourrait ne pas fonctionner correctement.")

# Réorganisation des onglets
tab_upload, tab_human, tab_detection, tab_background, tab_classification, tab_similarites, tab_cartes, tab_metriques = st.tabs([
    "Upload", "Traces Humaines", "Détection Oiseaux", "Background", "Classification", "Similarités", "Données Géographiques", "Métriques"
])

# ---------- Onglet Upload ----------
with tab_upload:
    st.header("Upload d'images")
    uploaded_files = st.file_uploader(
        "Téléverser des images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if uploaded_files:
        st.session_state.images_info = []  # Réinitialisation
        
        # Sauvegarde des images
        for file in uploaded_files:
            filename = secure_filename(file.name)
            original_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(original_path, 'wb') as f:
                f.write(file.getbuffer())
            
            # Stockage initial
            st.session_state.images_info.append({
                "filename": filename,
                "original_path": original_path,
                "annotated_path": None,
                "bbox_txt": None,
                "metrics": None,
                "bird_detected": False,
                "human_detected": False
            })
        st.success(f"{len(uploaded_files)} images téléversées avec succès!")

# ---------- Onglet Traces Humaines ----------
with tab_human:
    st.header("Détection de traces humaines (YOLOv12)")
    
    if not st.session_state["images_info"]:
        st.warning("⚠️ Veuillez d'abord uploader des images")
    else:
        if st.button("🔍 Lancer la détection de traces humaines"):
            if st.session_state.models is None or not isinstance(st.session_state.models, dict) or 'human_model' not in st.session_state.models:
                st.error("❌ Modèle YOLOv12 non disponible")
            else:
                human_model = st.session_state.models['human_model']
                
                # Réinitialiser le statut de détection humaine pour toutes les images
                for info in st.session_state["images_info"]:
                    info["human_detected"] = False
                
                for info in st.session_state["images_info"]:
                    with st.spinner(f"Analyse de {info['filename']}..."):
                        results = human_model.predict(
                            source=info["original_path"],
                            conf=0.25,
                            save=False,
                            save_txt=False,
                            augment=False
                        )
                        
                        img_bgr = cv2.imread(info["original_path"])
                        human_detected = False
                        
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf.item()
                                cls = int(box.cls.item())
                                
                                # Détecter uniquement les traces humaines (classe 0)
                                if cls == 0:
                                    human_detected = True
                                    info["human_detected"] = True  # Mettre à jour directement dans l'info

                                # Ne traiter que les classes 0 et 1
                                X1, Y1, X2, Y2 = map(int, [x1, y1, x2, y2])
                                class_name = "Traces humaines" if cls == 0 else "Oiseau"
                                color = (0, 255, 0) if cls == 0 else (255, 0, 0)  # Fixed syntax
                                
                                # Dessiner les boîtes
                                cv2.rectangle(img_bgr, (X1,Y1), (X2,Y2), color, 2)
                                label = f"{class_name} {conf:.2f}"
                                
                                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(img_bgr, 
                                            (X1, Y1 - text_size[1] - 4), 
                                            (X1 + text_size[0], Y1), 
                                            color, -1)
                                cv2.putText(img_bgr, label, (X1, Y1-4),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                        if human_detected:
                            status = "Traces humaines détectées 🏗️"
                            st.write("**Légende des couleurs:**")
                            st.markdown("- 🟢 Traces humaines")
                            st.markdown("- 🔴 Oiseau")
                        else:
                            status = "Aucune trace humaine"
                            
                        st.write(f"**Résultat pour {info['filename']} :** {status}")
                        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                use_container_width=True)

# ---------- Onglet Détection Oiseaux ----------
with tab_detection:
    st.header("Détection d'oiseaux (YOLOv5)")
    
    if not st.session_state["images_info"]:
        st.warning("⚠️ Veuillez d'abord uploader des images")
    else:
        if not any("human_detected" in info for info in st.session_state["images_info"]):
            st.warning("⚠️ Veuillez d'abord lancer la détection de traces humaines")
        else:
            if st.button("🔍 Lancer la détection d'oiseaux"):
                if st.session_state.models is None or 'bird_model' not in st.session_state.models:
                    st.error("❌ Modèle YOLOv5 non disponible")
                else:
                    bird_model = st.session_state.models['bird_model']
                    progress_bar = st.progress(0)
                    
                    for idx, info in enumerate(st.session_state["images_info"]):
                        with st.spinner(f"Analyse de {info['filename']}..."):
                            # Détection des oiseaux
                            found_box, annotated_img, metrics = detect_bird(info["original_path"], bird_model)
                            
                            if annotated_img is not None:
                                # Sauvegarder l'image annotée
                                save_dir = DETECTED_FOLDER if found_box else NOT_DETECTED_FOLDER
                                annotated_path = os.path.join(save_dir, info["filename"])
                                cv2.imwrite(annotated_path, annotated_img)
                                
                                # Si des oiseaux sont détectés, sauvegarder les bbox
                                if found_box and metrics:
                                    bbox_txt = save_bbox_txt(info["original_path"], metrics)
                                    info.update({
                                        "annotated_path": annotated_path,
                                        "bbox_txt": bbox_txt,
                                        "metrics": metrics,
                                        "bird_detected": True
                                    })
                                    st.success(f"✅ Oiseau détecté dans {info['filename']}")
                                else:
                                    info.update({
                                        "annotated_path": annotated_path,
                                        "bbox_txt": None,
                                        "metrics": [],
                                        "bird_detected": False
                                    })
                                    st.info(f"ℹ️ Pas d'oiseau détecté dans {info['filename']}")
                                
                                st.image(annotated_path, 
                                       caption=f"Détections sur {info['filename']}", 
                                       use_container_width=True)
                            
                            progress_bar.progress((idx + 1) / len(st.session_state["images_info"]))

# ---------- Onglet Background ----------
with tab_background:
    st.header("Extraction du Background")
    if not st.session_state["images_info"]:
        st.warning("⚠️ Veuillez d'abord uploader des images")
    else:
        images_processed = [info for info in st.session_state["images_info"] 
                          if "bird_detected" in info]
        
        if not images_processed:
            st.warning("⚠️ Veuillez d'abord lancer la détection d'oiseaux")
        else:
            filenames = [info["filename"] for info in images_processed]
            selected_images = st.multiselect(
                "Sélectionnez les images pour extraire le background", 
                filenames
            )
            
            if st.button("Extraire le background"):
                if not selected_images:
                    st.warning("Veuillez sélectionner au moins une image.")
                else:
                    for info in images_processed:
                        if info["filename"] in selected_images:
                            try:
                                # Debug info
                                st.write(f"Processing: {info['filename']}")
                                st.write(f"Original path: {info['original_path']}")
                                st.write(f"Bbox path: {info.get('bbox_txt', 'None')}")
                                
                                bbox_txt = None
                                if info.get("bird_detected", False):
                                    if info.get("bbox_txt") and os.path.exists(info["bbox_txt"]):
                                        bbox_txt = info["bbox_txt"]
                                    else:
                                        bbox_txt = save_bbox_txt(info["original_path"], info["metrics"])
                                        info["bbox_txt"] = bbox_txt
                                
                                species = "unknown"  # Keep lowercase consistent
                                extracted_paths = extract_background(
                                    info["original_path"],
                                    bbox_txt,
                                    BACKGROUND_FOLDER,
                                    species
                                )
                                
                                if extracted_paths:
                                    msg = "Background extrait" if info.get("bird_detected") else "Image originale utilisée comme background"
                                    st.success(f"✅ {msg} pour {info['filename']}")
                                    
                                    num_cols = 3
                                    cols = st.columns(num_cols)
                                    for i, path in enumerate(extracted_paths):
                                        with cols[i % num_cols]:
                                            st.image(path, 
                                                   caption=f"Background: {os.path.basename(path)}", 
                                                   use_container_width=True)
                                            with open(path, "rb") as file:
                                                st.download_button(
                                                    label=f"Télécharger",
                                                    data=file,
                                                    file_name=os.path.basename(path),
                                                    mime="image/jpeg"
                                                )
                                else:
                                    st.error(f"❌ Erreur lors de l'extraction pour {info['filename']}")
                            except Exception as e:
                                st.error(f"❌ Erreur pour {info['filename']}: {str(e)}")

# ---------- Onglet Classification (Flou vs Nette) ----------
with tab_classification:
    st.header("Classification : Flou ou Nette (Deep)")
    if not st.session_state["images_info"]:
        st.warning("Aucune image n'a été téléversée…")
    elif st.session_state.models is None or not isinstance(st.session_state.models, dict) or 'blur_model' not in st.session_state.models:
        st.error("Modèle de détection de flou non disponible")
    else:
        blur_model = st.session_state.models['blur_model']
        for info in st.session_state["images_info"]:
            st.subheader(info["filename"])
            img_bgr = cv2.imread(info["original_path"])
            h0, w0 = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (640, 640))
            tensor = torch.from_numpy(img_resized).permute(2,0,1).float()/255.0
            tensor = tensor.unsqueeze(0)

            # 1) Inference YOLO
            pred = blur_model(tensor)
            det = non_max_suppression(pred, conf_thres=0.05, iou_thres=0.45)[0]

            # 2) Remise à l’échelle et annotation
            blur_detected = False
            for *xyxy, conf, cls in det:
                # xyxy sont en 0–640
                x1, y1, x2, y2 = (xyxy[i].item() for i in range(4))
                # remap sur taille originale
                x1 = int(x1 * w0 / 640);  y1 = int(y1 * h0 / 640)
                x2 = int(x2 * w0 / 640);  y2 = int(y2 * h0 / 640)
                blur_detected = True
                # rectangle et label
                cv2.rectangle(img_bgr, (x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(img_bgr, f"Flou {conf:.2f}", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

            # 3) Affichage
            status = "Flou détecté 📦" if blur_detected else "Aucun flou"
            st.write(f"**Résultat Deep :** {status}")
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                     use_container_width=True)

            # 4) (optionnel) proposer les crops
            if blur_detected:
                crops = []
                # on énumère correctement chaque détection
                for i, box in enumerate(det):          
                    # box est un tensor de 6 éléments [x1,y1,x2,y2,conf,cls]
                    x1, y1, x2, y2, conf, cls_idx = box  

                    # 1) remise à l’échelle
                    x1 = int(x1 * w0 / 640)
                    y1 = int(y1 * h0 / 640)
                    x2 = int(x2 * w0 / 640)
                    y2 = int(y2 * h0 / 640)

                    # 2) clamp dans l'image
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w0, x2), min(h0, y2)

                    # 3) ignore les bboxes invalides
                    if x2 <= x1 or y2 <= y1:
                        st.warning(f"Ignoré bbox vide #{i} pour {info['filename']}")
                        continue

                    # extraction du crop de la zone floue
                    crop = img_bgr[y1:y2, x1:x2]
                    base, ext = os.path.splitext(info['filename'])   # "Acanthiza_pusilla_5", ".jpg"
                    cname = f"{base}_blur_{i}{ext}"                   # "Acanthiza_pusilla_5_blur_0.jpg"
                    out_path = os.path.join(blurry_folder, cname)
                    success = cv2.imwrite(out_path, crop)
                    if not success or not os.path.exists(out_path):
                        st.warning(f"Échec écriture crop #{i} pour {info['filename']}")
                        continue

                    crops.append(out_path)

                if crops:
                    st.write("### Zones floues extraites")
                    for path in crops:
                        st.image(path, width=150)

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

with tab_metriques:
    st.header("Métriques d'entraînement du modèle YOLOv5 pour détection d'oiseaux")
    
    # Chemin vers l'image des résultats
    results_path = os.path.join('metriques', 'results.png')
    
    if os.path.exists(results_path):
        # Charger et afficher l'image
        results_img = Image.open(results_path)
        st.image(results_img, caption="Métriques d'entraînement du modèle YOLOv5", use_container_width=True)
        
        # Bouton de téléchargement
        with open(results_path, 'rb') as f:
            st.download_button(
                label="Télécharger les métriques",
                data=f,
                file_name="results.png",
                mime="image/png"
            )
        
        # Explication des métriques
        st.markdown("""
        **Explication des métriques :**
        - **Loss (train/val)**: Évolution des pertes pendant l'entraînement et la validation
        - **mAP@0.5**: Précision moyenne à 50% d'IoU
        - **Precision**: Ratio des détections correctes parmi toutes les détections
        - **Recall**: Ratio des vrais positifs détectés parmi tous les vrais positifs
        """)
    else:
        st.warning("Fichier results.png introuvable. Veuillez vérifier qu'il se trouve dans le dossier yolov5.")


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
            if "latitude" not in df.columns or "longitude" not in df.columns:
                st.warning(f"Colonnes latitude/longitude manquantes dans {f.name}")
                continue
            pts = list(zip(df["latitude"], df["longitude"]))  # Fixed parentheses
            if pts:
                species_coords[raw] = pts
        except Exception as e:
            st.warning(f"Impossible de lire {f.name}: {str(e)}")
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

    # --- 6) Affichage de la carte ---
    st.write("### Carte des observations")
    if not rows:
        st.warning("Aucune donnée à afficher sur la carte.")
    else:
        df_plot = pd.DataFrame(rows)
        # centre la carte sur la moyenne des points
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