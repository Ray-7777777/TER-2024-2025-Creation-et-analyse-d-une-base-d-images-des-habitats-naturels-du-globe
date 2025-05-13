# Utiliser l'image Python 3.12 (image de base)
FROM python:3.12-slim

# Installer les dépendances nécessaires et PowerShell
RUN apt-get update && apt-get install -y \
    wget \
    curl \ 
    gnupg \
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
    gdal-bin \
    libgdal-dev

# ✅ Installer PowerShell manuellement pour ARM64 (Mac M1/M2)
RUN curl -L https://github.com/PowerShell/PowerShell/releases/download/v7.4.1/powershell-7.4.1-linux-arm64.tar.gz -o /tmp/pwsh.tar.gz && \
    mkdir -p /opt/microsoft/powershell/7 && \
    tar -xzf /tmp/pwsh.tar.gz -C /opt/microsoft/powershell/7 && \
    ln -s /opt/microsoft/powershell/7/pwsh /usr/bin/pwsh && \
    rm /tmp/pwsh.tar.gz


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

# Copier le dossier `Codes` dans le conteneur
COPY Codes/ .

# Copier le dossier `Donnees` dans le conteneur
COPY Donnees /Donnees

ENV GDAL_VERSION=3.6.2


# Copier le fichier requirements.txt (s'il est dans le dossier de base)
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Installer ultralytics
RUN pip install ultralytics

# Rendre les scripts exécutables
RUN chmod +x /Codes/Dependances/*.sh /Codes/Dependances/*.ps1 /Codes/principal.ps1

# Nettoyer les fichiers de script pour enlever les caractères Windows (\r)
RUN dos2unix /Codes/Dependances/*.sh /Codes/Dependances/*.ps1 /Codes/principal.ps1

# Commande pour exécuter le script PowerShell
CMD ["pwsh", "./principal.ps1"]
