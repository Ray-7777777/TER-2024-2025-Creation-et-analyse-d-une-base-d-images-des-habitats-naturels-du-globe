import sys
import os
import shutil
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))  # si besoin

import cv2
import torch

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# YOLOv5
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression

app = Flask(__name__)

# Dossiers
UPLOAD_FOLDER = './uploads'
DETECTED_FOLDER = './static/results'       # pour les images annotées
NOT_DETECTED_FOLDER = './static/no_results'  # pour les images sans bounding box
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTED_FOLDER, exist_ok=True)
os.makedirs(NOT_DETECTED_FOLDER, exist_ok=True)

# Extensions autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle YOLOv5
model = DetectMultiBackend('./best.pt', device=torch.device('cpu'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_bird(image_path):
    """
    Détecte s'il y a un oiseau dans l'image d'origine.
    - Redimensionne à 640x640 pour YOLO.
    - Applique NMS (conf >= 0.5).
    - Recalcule les coords pour l'image d'origine.
    - Dessine la bounding box (si trouvée) et renvoie True/False.
    Retourne:
      (True, img_bgr) si au moins une box >= 0.5 a été trouvée (img_bgr est l'image annotée),
      (False, img_bgr) sinon (img_bgr = l'image d'origine).
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Impossible de lire l'image : {image_path}")
        return False, None

    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    # Tenseur
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # Inférence brute
    results = model(img_tensor)
    # NMS
    preds = non_max_suppression(results, conf_thres=0.5, iou_thres=0.45)
    detections = preds[0]

    if detections is None or len(detections) == 0:
        return False, img_bgr  # aucune box trouvée => on renvoie l'image brute

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
            cv2.rectangle(img_bgr, (X1, Y1), (X2, Y2), (0,255,0), 2)
            cv2.putText(img_bgr, label, (X1, Y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0,255,0), 2)

    return found_box, img_bgr  # found_box = True/False, plus l'image finale

@app.route('/')
def index():
    return render_template('accueil.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """ Gère l'upload de plusieurs images et applique YOLO à chacune. """
    images_list = request.files.getlist("images")  # "images" => name="images" côté HTML

    if not images_list:
        return jsonify({"success": False, "error": "Aucune image envoyée."})

    results = []

    for file in images_list:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            found_box, img_bgr_final = detect_bird(image_path)
            if img_bgr_final is None:
                # Cas d'erreur lecture
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": "Impossible de lire l'image"
                })
                continue

            # Selon found_box, on enregistre dans DETECTED_FOLDER ou NOT_DETECTED_FOLDER
            if found_box:
                # => Sauvegarder l'image annotée dans DETECTED_FOLDER
                out_path = os.path.join(DETECTED_FOLDER, filename)
                cv2.imwrite(out_path, img_bgr_final)
                web_path = f"/static/results/{filename}"

                results.append({
                    "filename": filename,
                    "success": True,
                    "result_image": web_path
                })
            else:
                # => Sauvegarder l'image d'origine (sans annotation) dans NOT_DETECTED_FOLDER
                out_path = os.path.join(NOT_DETECTED_FOLDER, filename)
                cv2.imwrite(out_path, img_bgr_final)
                web_path = f"/static/no_results/{filename}"

                results.append({
                    "filename": filename,
                    "success": False,
                    "error": "Aucun oiseau détecté (conf>=0.5)",
                    "result_image": web_path  # on renvoie quand même l'image
                })

        else:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Fichier non autorisé."
            })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
