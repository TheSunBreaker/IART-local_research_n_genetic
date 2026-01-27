from shapes import *
import rerender
from PIL import Image
from utils import *
from seeker import *
import argparse
import warnings


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='IArt-Geom Pipeline')
parser.add_argument('--shape',
    choices=['ellipse', 'triangle', 'rectangle'], default='ellipse', help='Which shape do you want ?')
parser.add_argument('--n', default=118, type=int, help='How many shapes ?')
parser.add_argument('--time', default=30, type=int, help='In how much time max ? (in seconds)')
parser.add_argument('--input', default='mona.jpg', type=str, help='Input image name ?')
parser.add_argument('--output', default='mutated_phenotype', type=str, help='Output image name ? (Witouth the extension)')
parser.add_argument('--from_gen', default=None, type=str, help='The name of the genotype file (If you have one to start from).')



args = parser.parse_args()


if not (118 <= args.n <= 20000):
    parser.error("--n must be between 118 and 20000")

if not (30 <= args.time <= 3000):
    parser.error("--time must be between 30 and 3000")

mon_genome = [] #Liste qui retiendra les paramètres des formes.

SHAPES_NBR = args.n #Nombre de formes

input_image = Image.open(args.input).convert('RGB')

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

if args.from_gen is not None:
    print("Reprise depuis le fichier de sauvegarde :", args.from_gen)
    mon_genome = load_genome(args.from_gen)
else:
    generate_random_shape(mon_genome, W, H, SHAPES_NBR, args.shape)


print("Avant... Chargé.")

rerender.save_to_png(mon_genome, W, H, "base_phenotype.png")
rerender.save_to_svg(mon_genome, W, H, "base_phenotype.svg")

print("C'est partit pour ", args.time, "secondes ; avec ", args.n , args.shape, "s.")
print("Mutation en cours...")
iters, new_gen, bestFit = run_fitness_improver(mon_genome, W, H, target_matrix, args.output, time_limit=args.time)

print("Après", iters, "itérations, le résultat est...")

print("Prêt..., avec ", bestFit, " de fitness")