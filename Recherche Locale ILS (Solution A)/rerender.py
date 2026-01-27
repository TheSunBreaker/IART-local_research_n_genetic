# renderer.py
# Moteur de rendu

import cv2
import numpy as np

def render_phenotype(genome, width, height, bg_color=(0, 0, 0)):
    """
    Transforme une liste d'objets Shape en image NumPy (pour le calcul de fitness)
    """
    # Création du fond (Noir par défaut)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        canvas[:] = bg_color # Remplir avec une autre couleur si besoin

    # Dessin des formes (Algorithme du peintre)
    for shape in genome:
        shape.draw(canvas)
        
    return canvas

def save_to_png(genome, width, height, filename):
    img = render_phenotype(genome, width, height)
    cv2.imwrite(filename, img)
    print(f"PNG sauvegardé : {filename}")

def save_to_svg(genome, width, height, filename):
    svg_content = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n'
    
    # Fond noir par défaut (pour être cohérent avec OpenCV)
    svg_content += f'\t<rect x="0" y="0" width="{width}" height="{height}" fill="black" />\n'
    
    for shape in genome:
        svg_content += f'\t{shape.get_svg_tag()}\n'
        
    svg_content += '</svg>'
    
    with open(filename, 'w') as f:
        f.write(svg_content)
    print(f"SVG sauvegardé : {filename}")