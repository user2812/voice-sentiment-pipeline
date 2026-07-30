"""
Point d'entrée pour le déploiement sur Hugging Face Spaces.
Spaces s'attend à trouver un fichier app.py à la racine du dépôt ; ce fichier
fait simplement le lien vers l'application Gradio réelle, définie dans
src/ui/app.py, pour ne pas dupliquer la logique du pipeline.
"""

import sys

sys.path.append(".")

from src.ui.app import demo

if __name__ == "__main__":
    demo.launch()
