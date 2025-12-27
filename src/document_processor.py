
# Import des outils nécessaires
from pathlib import Path  # Pour gérer les chemins de fichiers
from typing import List, Dict, Any  # Pour le typage (bonne pratique)
import logging  # Pour enregistrer ce qui se passe

logger = logging.getLogger(__name__)  # Crée un "journal" pour ce fichier

# ------------------------------------------------------------
# IDÉE 1 : Le Design Pattern "Strategy"
# ------------------------------------------------------------
# On crée une interface abstraite (comme un contrat)
# Tous les chargeurs doivent suivre ce contrat
class DocumentLoader:
    """Interface pour tous les chargeurs de documents"""
    def load(self, file_path: Path) -> str:
        # Méthode abstraite : doit être implémentée par les enfants
        pass

# ------------------------------------------------------------
# IDÉE 2 : Un chargeur par format (spécialisation)
# ------------------------------------------------------------
class TextLoader(DocumentLoader):
    """Charge les fichiers .txt"""
    def load(self, file_path: Path) -> str:
        # Ouvre le fichier en mode lecture (r) avec encodage UTF-8
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()  # Retourne tout le contenu
        
        # POURQUOI 'with open' ? 
        # Ça ferme automatiquement le fichier après lecture
        # Évite les fuites de mémoire

class PDFLoader(DocumentLoader):
    """Charge les fichiers .pdf"""
    def load(self, file_path: Path) -> str:
        try:
            from pypdf import PdfReader  # Import seulement si nécessaire
            reader = PdfReader(file_path)
            
            # IDÉE : Extraire le texte de chaque page
            # Pourquoi " ".join() ? Pour éviter d'avoir des sauts de ligne partout
            return " ".join(page.extract_text() for page in reader.pages)
            
        except ImportError:
            # Si pypdf n'est pas installé, on informe l'utilisateur
            raise ImportError("PyPDF2 required for PDF files")

# ------------------------------------------------------------
# IDÉE 3 : La classe principale qui orchestre tout
# ------------------------------------------------------------
class DocumentProcessor:
    """Traite les documents : charge, nettoie, découpe"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialise avec des paramètres configurables
        
        CHUNK_SIZE = 500 mots
        Pourquoi 500 ? C'est assez petit pour rentrer dans un LLM
        mais assez grand pour avoir du contexte
        
        CHUNK_OVERLAP = 50 mots
        Pourquoi l'overlap ? Pour éviter de couper une idée en deux
        Exemple : 
        Chunk 1: ..."La révolution française a commencé en"
        Chunk 2: "1789 avec la prise de la Bastille"
        Sans overlap, on perd le lien !
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Dictionnaire des chargeurs disponibles
        # Clé = extension, Valeur = instance du chargeur
        self.loaders = {
            '.txt': TextLoader(),
            '.pdf': PDFLoader(),
            # On peut facilement ajouter d'autres formats plus tard
        }
    
    def process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Traite un seul fichier et retourne des chunks"""
        
        # 1. Vérifier que le fichier existe
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # 2. Identifier l'extension (.txt, .pdf, etc.)
        suffix = file_path.suffix.lower()
        
        # 3. Trouver le bon chargeur
        loader = self.loaders.get(suffix)
        
        if not loader:
            # Si format non supporté
            raise ValueError(f"Unsupported file type: {suffix}")
        
        # 4. Charger le texte
        text = loader.load(file_path)
        
        # 5. Découper en chunks
        return self._create_chunks(text, file_path)
    
    def _create_chunks(self, text: str, source: Path) -> List[Dict[str, Any]]:
        """
        Découpe le texte en chunks qui se chevauchent
        
        IMAGINE : Un livre avec des post-its qui se superposent
        Pour ne jamais perdre le fil
        """
        words = text.split()  # Transforme le texte en liste de mots
        
        chunks = []  # Liste qui contiendra tous nos chunks
        
        # Boucle magique : avancer par pas de (chunk_size - overlap)
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            # Prendre un morceau de mots
            chunk_words = words[i:i + self.chunk_size]
            
            # Si le morceau est vide, on arrête
            if not chunk_words:
                break
            
            # Recréer le texte du chunk
            chunk_text = ' '.join(chunk_words)
            
            # Créer un objet chunk avec métadonnées
            chunk = {
                'text': chunk_text,  # Le texte du chunk
                'metadata': {
                    'source': source.name,  # Nom du fichier
                    'path': str(source),    # Chemin complet
                    'start_idx': i,         # Position départ dans le texte
                    'end_idx': i + len(chunk_words),  # Position fin
                    'word_count': len(chunk_words)    # Nombre de mots
                }
            }
            
            chunks.append(chunk)
        
        return chunks

# ------------------------------------------------------------
# CONCEPT CLÉ : Pourquoi cette structure ?
# ------------------------------------------------------------
"""
1. Séparation des responsabilités :
   - TextLoader gère .txt
   - PDFLoader gère .pdf
   - DocumentProcessor orchestre

2. Extensibilité :
   Ajouter un nouveau format = créer une nouvelle classe

3. Testabilité :
   On peut tester chaque chargeur séparément
"""