import os
import cv2
import torch
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import sys
import os
# Ajouter le chemin du dossier utils de yolov5 à sys.path
sys.path.append(os.path.join(os.getcwd(), 'yolov5', 'utils'))  # Assure-toi que yolov5/utils existe

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from config import Config

# Initialisation de l'application Flask
app = Flask(__name__)
app.config.from_object(Config)

# Vérifier si le dossier logs existe, sinon le créer
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configurer les logs
logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_dir, 'app.log'),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Dossiers
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DETECTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['NOT_DETECTED_FOLDER'], exist_ok=True)

# Charger le modèle YOLOv5
model = DetectMultiBackend(app.config['MODEL_PATH'], device=torch.device('cpu'))

def allowed_file(filename):
    """Vérifie si le fichier a une extension valide."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def detect_bird(image_path):
    """Détecte un oiseau dans une image et renvoie les résultats."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        app.logger.error(f"Erreur de lecture de l'image : {image_path}")
        return False, None, []

    h_orig, w_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (640, 640))

    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    # Inférence YOLOv5
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
            cv2.rectangle(img_bgr, (X1, Y1), (X2, Y2), (0,255,0), 2)
            cv2.putText(img_bgr, label, (X1, Y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            metrics.append({
                "class": int(cls_idx),
                "confidence": conf.item(),
                "bbox": [X1, Y1, X2, Y2],
                "label": label
            })

    return found_box, img_bgr, metrics

@app.route('/')
def index():
    return render_template('accueil.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Gère l'upload des images et applique YOLO pour détecter les oiseaux."""
    images_list = request.files.getlist("images")

    if not images_list:
        return jsonify({"error": "Aucune image envoyée"}), 400

    results = []
    for file in images_list:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            found_box, img_bgr_final, metrics = detect_bird(image_path)

            if img_bgr_final is None:
                results.append({"filename": filename, "success": False, "error": "Impossible de lire l'image"})
                continue

            if found_box:
                out_path = os.path.join(app.config['DETECTED_FOLDER'], filename)
                cv2.imwrite(out_path, img_bgr_final)
                web_path = f"/static/results/{filename}"
                results.append({"filename": filename, "success": True, "result_image": web_path, "metrics": metrics})
            else:
                out_path = os.path.join(app.config['NOT_DETECTED_FOLDER'], filename)
                cv2.imwrite(out_path, img_bgr_final)
                web_path = f"/static/no_results/{filename}"
                results.append({"filename": filename, "success": False, "result_image": web_path, "metrics": []})

        else:
            results.append({"filename": file.filename, "success": False, "error": "Fichier non autorisé."})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
