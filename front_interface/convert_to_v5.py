# convert_to_v5.py
from ultralytics import YOLO

# Charge votre modèle (V8 ou V5 exporté en V8)
model = YOLO('best.pt')

# Exporte-le au format PyTorch binaire « legacy » compatible YOLOv5
# Cela produit best_legacy.pt
model.export(format='pt', legacy=True)
print("Conversion terminée → best_legacy.pt")
