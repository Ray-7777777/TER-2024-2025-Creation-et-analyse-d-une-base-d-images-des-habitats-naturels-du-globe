import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configurer les dossiers pour l'upload et la sortie des images
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './static/backgrounds'

# Créer les dossiers si ce n'est pas déjà fait
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Définir les extensions de fichiers autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_background(image_path, txt_file_path, output_dir, species):
    # (Ton code d'extraction de background ici)
    # Assure-toi que le dossier output_dir et species existent et crée le dossier de sortie.
    pass  # Remplace cela par ton code existant pour l'extraction de background

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'image' not in request.files or 'txt' not in request.files:
        return redirect(request.url)

    image = request.files['image']
    txt_file = request.files['txt']

    # Vérifier si les fichiers ont un nom et si leur extension est autorisée
    if image and allowed_file(image.filename) and txt_file and allowed_file(txt_file.filename):
        # Sauvegarder les fichiers uploadés
        image_filename = secure_filename(image.filename)
        txt_filename = secure_filename(txt_file.filename)

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        txt_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

        image.save(image_path)
        txt_file.save(txt_path)

        # Appeler la fonction d'extraction de background
        species = 'example_species'  # Remplace par l'espèce spécifique si nécessaire
        extract_background(image_path, txt_path, OUTPUT_FOLDER, species)

        # Rediriger vers la page avec les résultats
        return redirect(url_for('display_backgrounds'))

    return redirect(url_for('index'))

@app.route('/backgrounds')
def display_backgrounds():
    # Afficher les backgrounds extraits
    backgrounds = os.listdir(OUTPUT_FOLDER)
    backgrounds = [bg for bg in backgrounds if bg.endswith(('jpg', 'jpeg', 'png'))]
    return render_template('backgrounds.html', backgrounds=backgrounds)

if __name__ == '__main__':
    app.run(debug=True)
