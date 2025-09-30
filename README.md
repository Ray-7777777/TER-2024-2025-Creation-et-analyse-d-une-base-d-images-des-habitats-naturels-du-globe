# TER-2024-2025-Creation-et-analyse-d-une-base-d-images-des-habitats-naturels-du-globe


## Description du projet:
Ce projet de recherche interdisciplinaire explore les liens potentiels entre les caractéristiques physiologiques des oiseaux et leur environnement dans un contexte de sélection sexuelle. L'objectif est de déterminer si certains traits physiques des oiseaux reflètent des éléments de leur habitat, non seulement pour le camouflage mais également comme stratégie de séduction.

## Participants du projet:
  GUILHEM Adrian: adrian.guilhem@etu.univ-montp3.fr
  INGABIRE Alvin : alvin.ingabire@etu.univ-montp3.fr
  NIYONKURU Berline: berline.niyonkuru@etu.univ-montp3.fr
  TAHARASTE Rayan: rayan.taharaste@etu.univ-montp3.fr
  ZEROUAL Walid: walid.zeroual@etu.univ-montp3.fr

## Structure du GIT:
  Dossier Bibliographie: regroupe tout les articles,livres et autres ressources qui ont été lues pour la réalisation de ce projet
  
  Dossier Données: regroupe toutes les données utilisées lors de ce projet
  
  Dossier Codes: regroupe tout le code généré pour la réalisation de ce projet
  
  Dossier Rendus: regroupe tout les rendus écrits lors des différentes étapes de ce projet
  
  Dossier Rapport: contient le rapport du projet
  
  Dossier Soutenance: contient tout les fichiers nécessaires à la soutenance de ce projet

## Pipeline de Traitement
Acquisition des données : Récupération d'images via l'API iNaturalist

Prétraitement : Détection et filtrage des images avec traces humaines et zones floues

Extraction des oiseaux : Modèle YOLOv5x optimisé pour la détection d'oiseaux

Segmentation des arrière-plans : Algorithme d'extraction automatique des environnements

Analyse des similarités : Calcul de distances entre caractéristiques visuelles

## Technologies Utilisées
Computer Vision : YOLOv5, YOLOv11, ResNet-50, OpenCV

Traitement d'images : Détection de flou, extraction de features

Data Science : Analyse statistique, clustering, similarités cosinus

Visualisation : Streamlit, cartes interactives

Infrastructure : Docker, Google Colab, GitHub
