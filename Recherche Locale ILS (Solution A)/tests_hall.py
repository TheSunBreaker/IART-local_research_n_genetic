from shapes import *
import rerender
from PIL import Image
from utils import *

best_for_now = None #Cette variable retiendra le meileur pour l'instant, histoire de mettre à jour le SVG

current_canvas = None #Garde le tableau numpy de l'image courante

mon_genome = [] #Liste qui retiendra les paramètres des formes.

SHAPES_NBR = 100 #Nombre de formes

input_image = Image.open("mona.jpg").convert('RGB')

#On re-resolutionne l'image si nécessaire, de sorte qu'elle n'excède pas 1000*1000px. Histoire d'éviter des calculs inutilement complexes
MAX_W, MAX_H = 250, 250

# Redimensionne seulement si nécessaire, en gardant le ratio
input_image.thumbnail((MAX_W, MAX_H))


# 1. Définition de la taille de l'image
# Définie par la taille de l'image d'entrée 
W, H = input_image.size

# 4. Conversion en NumPy + Gestion des Types
# On utilise int16 pour éviter les bugs de soustraction (overflow) plus tard
target_matrix = np.array(input_image, dtype=np.int16)

# 5. CORRECTION DES COULEURS (Vital !)
# On inverse l'ordre des couleurs pour passer de RGB (PIL) à BGR (OpenCV)
# Sinon ton algo va confondre le rouge et le bleu
target_matrix = target_matrix[:, :, ::-1]
# mon_genome = [
#     # Rectangle bleu, tourné de 10 degrés
#     Rectangle(cx=200, cy=200, width=300, height=50, angle=10, 
#               r=50, g=50, b=255, alpha=150),
              
#     # Ellipse rouge, tournée de 90 degrés
#     Ellipse(cx=100, cy=100, rx=50, ry=80, angle=113, 
#             r=255, g=0, b=0, alpha=200),
            
#     # Triangle vert
#     Triangle(x1=300, y1=300, x2=350, y2=250, x3=380, y3=350, 
#              r=0, g=255, b=0, alpha=180)
# ]

# Génération Aléatoire de formes

# 3. Génération et Sauvegarde
# Pour la fitness, tu utiliseras juste : 
# img_pour_fitness = renderer.render_phenotype(mon_genome, W, H)


generate_random_shape(mon_genome, W, H, SHAPES_NBR, "rectangle")

# mon_genome_clone = clone_genome(mon_genome)

current_canvas = rerender.render_phenotype(mon_genome, W, H)

# Pour voir le résultat maintenant :
rerender.save_to_png(mon_genome, W, H, "test_classes.png")
rerender.save_to_svg(mon_genome, W, H, "test_classes.svg")

print("FitNess avant mutation : ")
print(fitness_calc(target_matrix, current_canvas))

print("Test de la fonction de perturbation.")
print("Perturbations en cours...")

perturbed_genome = mutate_genome_dump(mon_genome, W, H)

print("Génération du phénotype résultant...")

rerender.save_to_png(perturbed_genome, W, H, "test_mutation.png")
rerender.save_to_svg(perturbed_genome, W, H, "test_mutation.svg")

current_canvas = rerender.render_phenotype(perturbed_genome, W, H)

print("Fitness du muté :")
print(fitness_calc(target_matrix, current_canvas))