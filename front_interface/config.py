import os

class Config:
    # Configuration pour le dossier de fichiers uploadés
    UPLOAD_FOLDER = './uploads'
    DETECTED_FOLDER = './static/results'  # pour les images annotées
    NOT_DETECTED_FOLDER = './static/no_results'  # pour les images sans détection
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = './best.pt'  # Chemin vers le modèle YOLOv5
    LOG_FILE = './logs/app.log'  # Fichier de log principal

    @staticmethod
    def init_app(app):
        pass
