
import numpy as np  # Bibliothèque pour les calculs mathématiques
from typing import Union, List  # Pour le typage
import logging  # Pour le journal
from config.settings import Config  # Import de la configuration

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# IDÉE 1 : Le principe des embeddings
# ------------------------------------------------------------
"""
Un embedding = un vecteur de nombres qui représente le sens

Exemple :
"chat" = [0.1, 0.7, 0.3, 0.9, 0.2, ...]  # 384 nombres
"chien" = [0.2, 0.6, 0.4, 0.8, 0.3, ...]  # Proche de "chat"
"voiture" = [0.9, 0.1, 0.8, 0.2, 0.7, ...] # Loin de "chat"

Comment ça marche ?
Le modèle a appris sur des milliards de phrases que
"chat" et "chien" apparaissent dans des contextes similaires
"""

class EmbeddingService:
    def __init__(self):
        # Initialisation "lazy" : on ne charge le modèle que si nécessaire
        self.model = None  # Pas de modèle chargé au départ
        self.model_name = Config.EMBEDDING_MODEL  # "all-MiniLM-L6-v2"

        # Pourquoi ce modèle ?
        # - Petit (384 dimensions) → rapide
        # - Multilingue → comprend français et un peu darija
        # - Bien évalué → bon compromis qualité/performance

    def _load_model(self):
        """
        Charge le modèle seulement quand on en a besoin
        (Lazy Loading)

        POURQUOI ? Parce que charger le modèle prend:
        - 500MB de RAM
        - 5-10 secondes
        On veut éviter si on n'en a pas besoin
        """
        if self.model is None:  # Si pas encore chargé
            try:
                # Import spécifique ici pour éviter dépendance globale
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                logger.info("Model loaded successfully")

            except ImportError as e:
                # Si la bibliothèque n'est pas installée
                raise ImportError(f"Failed to load model: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Transforme du texte en vecteurs (embeddings)

        Union[str, List[str]] :
        Accepte une chaîne unique OU une liste de chaînes
        """
        self._load_model()  # Charge le modèle si pas déjà fait

        # Si c'est une seule chaîne, on la met dans une liste
        if isinstance(texts, str):
            texts = [texts]

        # Appel au modèle pour générer les embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,  # Pas de barre de progression
            normalize_embeddings=True  # IMPORTANT : normalise pour similarité cosinus
        )

        # Que fait normalize_embeddings ?
        # Il s'assure que tous les vecteurs ont une longueur de 1
        # Pourquoi ? Pour que la similarité cosinus fonctionne bien
        # Formule : cos(θ) = (A·B) / (|A||B|)
        # Si |A| = |B| = 1, alors cos(θ) = A·B (simple produit scalaire)

        return embeddings

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs

        POURQUOI cosinus ? Parce que ça mesure l'angle entre vecteurs
        - Vecteurs similaires = petit angle = cos proche de 1
        - Vecteurs différents = grand angle = cos proche de 0
        - Vecteurs opposés = angle 180° = cos = -1
        """
        # Vérifier que les vecteurs ont la même forme
        if a.shape != b.shape:
            raise ValueError("Vectors must have same shape")

        # Calculer la norme (longueur) de chaque vecteur
        norm_a = np.linalg.norm(a)  # √(a₁² + a₂² + ... + aₙ²)
        norm_b = np.linalg.norm(b)

        # Éviter division par zéro
        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Produit scalaire : a·b = a₁b₁ + a₂b₂ + ... + aₙbₙ
        dot_product = np.dot(a, b)

        # Similarité cosinus : (a·b) / (|a||b|)
        similarity = dot_product / (norm_a * norm_b)

        # Retourne un float entre -1 et 1
        return float(similarity)

# ------------------------------------------------------------
# EXEMPLE CONCRET :
# ------------------------------------------------------------
"""
texts = ["Le chat dort", "Le chien aboie", "La voiture roule"]
embeddings = service.encode(texts)

Résultat :
array([[ 0.1, 0.7, 0.3, ... ],  # embedding "chat"
       [ 0.2, 0.6, 0.4, ... ],  # embedding "chien" (proche)
       [ 0.9, 0.1, 0.8, ... ]]) # embedding "voiture" (loin)

similarité("chat", "chien") = 0.85
similarité("chat", "voiture") = 0.15
"""
