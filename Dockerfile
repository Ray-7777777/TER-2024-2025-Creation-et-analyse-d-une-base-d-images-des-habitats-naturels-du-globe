# Utiliser l'image Python 3.12 (image de base)
FROM python:3.12-slim

# Installer les dépendances nécessaires et PowerShell
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    apt-transport-https \
    software-properties-common \
    lsb-release \
    dos2unix \
    libgl1-mesa-glx \
    build-essential \
    git \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
  && wget -q "https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb" -O packages-microsoft-prod.deb \
  && dpkg -i packages-microsoft-prod.deb \
  && apt-get update && apt-get install -y powershell \
  && rm packages-microsoft-prod.deb

# Installer ImageMagick depuis le dépôt GitHub
RUN git clone https://github.com/ImageMagick/ImageMagick.git \
  && cd ImageMagick \
  && ./configure --with-jpeg=yes --with-png=yes --with-tiff=yes \
  && make \
  && make install \
  && ldconfig /usr/local/lib \
  && cd .. \
  && rm -rf ImageMagick

# Ajouter ImageMagick au PATH
ENV PATH="/usr/local/bin:$PATH"

# Définir le répertoire de travail
WORKDIR /Codes

# --- Étape optimisée : copier d'abord requirements.txt ---

# Copier le fichier requirements.txt (s'il est dans le dossier de base)
COPY requirements.txt .

# Installer les dépendances Python (seul pip install sera relancé si requirements.txt change)
RUN pip install --no-cache-dir \
      --trusted-host pypi.org \
      --trusted-host pypi.python.org \
      --trusted-host files.pythonhosted.org \
      -r requirements.txt \
  && pip install ultralytics

# --- Ensuite copier le code et les données ---

# Copier le dossier Codes dans le conteneur
COPY Codes/ .

# Copier le dossier Donnees dans le conteneur
COPY Donnees /Donnees

# Après avoir copié Codes/ et Donnees/
RUN find /Codes/Dependances -type f \( -iname "*.sh" -o -iname "*.ps1" -o -iname "*.py" \) \
      -exec dos2unix {} \; \
    && dos2unix /Codes/principal.ps1

# Rendre les scripts exécutables
RUN chmod +x /Codes/Dependances/*.sh \
             /Codes/Dependances/*.ps1 \
             /Codes/Dependances/*.py \
             /Codes/principal.ps1

# Commande pour exécuter le script PowerShell
CMD ["pwsh", "./principal.ps1"]