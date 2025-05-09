import os
import cv2
import torch
import streamlit as st
from werkzeug.utils import secure_filename
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
import pandas as pd
import random
import numpy as np

# Import de la fonction d'extraction du background
from extraction_bckgd import extract_background

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

############################
# Configuration des dossiers et chargement du modèle
############################

UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'static/results'
NOT_DETECTED_FOLDER = 'static/no_results'
BACKGROUND_FOLDER = os.path.join('static', 'background')
sharp_folder = os.path.join('static', 'classified', 'sharp')
blurry_folder = os.path.join('static', 'classified', 'blurry')

for folder in [UPLOAD_FOLDER, DETECTED_FOLDER, NOT_DETECTED_FOLDER, BACKGROUND_FOLDER, sharp_folder, blurry_folder]:
    os.makedirs(folder, exist_ok=True)

model = DetectMultiBackend('./best.pt', device=torch.device('cpu'))

############################
# Interface Streamlit avec onglets
############################

# Initialisation de la session pour stocker les infos
if "images_info" not in st.session_state:
    st.session_state["images_info"] = []

tab_detection, tab_background, tab_classification = st.tabs(["Détection", "Background", "Classification"])

# ---------- Onglet Détection ----------
with tab_detection:
    st.header("Détection d'oiseaux")
    uploaded_files = st.file_uploader("Téléverser des images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state["images_info"] = []  # Réinitialiser à chaque upload
        for file in uploaded_files:
            filename = secure_filename(file.name)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(image_path, 'wb') as f:
                f.write(file.getbuffer())
            
            found_box, annotated_img, metrics = detect_bird(image_path)
            if annotated_img is None:
                st.error(f"Erreur pour l'image {filename}")
                continue
            
            if found_box:
                save_path = os.path.join(DETECTED_FOLDER, filename)
            else:
                save_path = os.path.join(NOT_DETECTED_FOLDER, filename)
            cv2.imwrite(save_path, annotated_img)
            
            bbox_txt = None
            if metrics:
                bbox_txt = save_bbox_txt(image_path, metrics)
            
            st.session_state["images_info"].append({
                "filename": filename,
                "original_path": image_path,
                "annotated_path": save_path,
                "metrics": metrics,
                "bbox_txt": bbox_txt,
                "found_box": found_box
            })
        
        for info in st.session_state["images_info"]:
            st.subheader(f"Résultats pour : {info['filename']}")
            st.image(info["annotated_path"], caption=info['filename'], use_container_width=True)
            if info["metrics"]:
                st.write("**Métriques :**")
                df = pd.DataFrame(info["metrics"])
                st.dataframe(df)
                if info["bbox_txt"] and os.path.exists(info["bbox_txt"]):
                    with open(info["bbox_txt"], "r") as txt_file:
                        st.code(txt_file.read())
                    with open(info["bbox_txt"], "rb") as file:
                        st.download_button("Télécharger les coordonnées", data=file, file_name=os.path.basename(info["bbox_txt"]))
            else:
                st.write("Aucune détection d'oiseau.")

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




