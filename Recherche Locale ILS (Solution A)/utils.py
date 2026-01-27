from shapes import *
from PIL import Image
import numpy as np
import pickle
import os

def save_genome(genome, filename):
    with open(filename, "wb") as f:
        pickle.dump(genome, f)

def load_genome(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    with open(filename, "rb") as f:
        return pickle.load(f)


def generate_random_shape(genome, w, h, shape_num = 100, shape = "ellipse"):

    Choosen_shape = None
    for _ in range(shape_num):
        if shape == "ellipse":
                Choosen_shape = Ellipse
        elif shape == "rectangle":
                 Choosen_shape = Rectangle
        else :
              Choosen_shape = Triangle

        genome.append(Choosen_shape.random(w, h))

def clone_genome(genA):
      
      genB = []

      for strand in genA :
            copy = strand.copy()
            genB.append(copy)
      
      return genB

def fitness_calc(target_array, generated_array):
    """
    Calcule la différence entre deux images.
    target_array : numpy array (int16 ou float)
    generated_array : numpy array (uint8 qui sort d'OpenCV)
    """
    
    # 1. On s'assure que l'image générée est aussi convertie pour le calcul
    # (Sinon on aura des problèmes de soustraction avec les uint8)
    # CORRECTION ICI : On passe en int64 pour éviter l'overflow du carré (255^2 > 32767)
    gen_pixels = generated_array.astype(np.int64)
    
    # IMPORTANT : Il faut aussi s'assurer que target est assez grand
    target_pixels = target_array.astype(np.int64)
    
    # 2. Différence pixel à pixel (Vectorisation)
    # Ça crée une matrice de différences instantanément
    diff = target_pixels - gen_pixels
    
    # 3. Carré des différences
    # Maintenant que c'est en int64, 65025 rentre sans devenir négatif
    sq_diff = np.square(diff)
    
    # 4. Somme de tout
    error_score = np.sum(sq_diff)
    
    return error_score

def fitness_local_change(target_arr, old_img_arr, new_img_arr, bbox):
    """
    Calcule la variation de fitness uniquement sur la zone modifiée (bbox).
    bbox = (x, y, w, h)
    Retourne: La différence de score (Delta)
    """
    x, y, w, h = bbox
    
    # 1. Sécuriser les coordonnées (clipping) pour ne pas sortir de l'image
    # C'est vital car get_bounds peut donner des coordonnées négatives si la forme dépasse
    H_img, W_img, _ = target_arr.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W_img, x + w)
    y2 = min(H_img, y + h)
    
    # Si la zone est invalide (hors image), pas de changement
    if x2 <= x1 or y2 <= y1:
        return 0

    # 2. Extraire uniquement les "patches" (morceaux d'image)
    # Conversion int64 ici pour éviter les conversions de toute l'image
    target_slice = target_arr[y1:y2, x1:x2].astype(np.int64)
    old_slice = old_img_arr[y1:y2, x1:x2].astype(np.int64)
    new_slice = new_img_arr[y1:y2, x1:x2].astype(np.int64)

    # 3. Calculer l'erreur locale AVANT et APRÈS
    # SSE = Somme des différences au carré
    old_error = np.sum(np.square(target_slice - old_slice))
    new_error = np.sum(np.square(target_slice - new_slice))

    # 4. Le Delta est : Ce qu'on a gagné (ou perdu)
    # Si new_error < old_error, le delta sera négatif (l'erreur totale baisse)
    delta = new_error - old_error
    
    return delta

def mutate_genome_dump(genome, w, h):

      copy = clone_genome(genome)
      for shape in copy :
            shape.perturb(w, h)
      return copy

def mutate_targeted_gene(genome, w, h, i):
      copy = clone_genome(genome)
      copy[i].perturb(w, h)

      return copy