import argparse
import random
import time
import os
import glob
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageChops, ImageStat

# ============================================================
# Fonctions utilitaires

def clamp(x, a, b):
    return max(a, min(b, x))

LAST_DRAW_TIME = 0
DRAW_INTERVAL = 0.05
FITNESS_HISTORY = [] 

# Visualisation de la courbe de fitness

def draw_fitness_curve(history, width, height=150):
    if not history:
        return np.ones((height, width, 3), dtype=np.uint8) * 255

    graph = np.ones((height, width, 3), dtype=np.uint8) * 245 # Gris très clair
    
    # Sampling si trop de données
    if len(history) > width:
        indices = np.linspace(0, len(history) - 1, width).astype(int)
        data = [history[i] for i in indices]
    else:
        data = history

    min_v, max_v = min(data), max(data)
    if max_v == min_v: max_v += 1e-5
    
    points = []
    for i, val in enumerate(data):
        x = int((i / (len(data) - 1)) * width)
        ratio = (val - min_v) / (max_v - min_v)
        y = int(height - 10 - (ratio * (height - 20))) 
        points.append((x, y))

    if len(points) > 1:
        cv2.polylines(graph, [np.array(points)], isClosed=False, color=(0, 0, 180), thickness=2, lineType=cv2.LINE_AA)
    
    # Texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(graph, f"Start: {int(history[0])}", (5, height - 5), font, 0.4, (0, 0, 0), 1)
    cv2.putText(graph, f"Err: {int(history[-1])}", (width - 100, height - 5), font, 0.4, (0, 0, 0), 1)
    
    return graph

def update_view(target, genotype, bg, window_name, original_np):
    global LAST_DRAW_TIME
    if time.time() - LAST_DRAW_TIME < DRAW_INTERVAL:
        return

    # Render
    img = render(genotype, target.size, bg)
    arr = np.array(img.convert("RGB"))

    # Stack images
    combined = np.hstack((original_np, arr))
    
    # Resize if too big
    MAX_W = 1200
    h, w = combined.shape[:2]
    if w > MAX_W:
        r = MAX_W / w
        combined = cv2.resize(combined, (MAX_W, int(h * r)), interpolation=cv2.INTER_AREA)

    # Add graph
    graph = draw_fitness_curve(FITNESS_HISTORY, width=combined.shape[1], height=100)
    final = np.vstack((combined, graph))

    cv2.imshow(window_name, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    LAST_DRAW_TIME = time.time()

# ============================================================
# Coeur des opérations sur les formes

def get_bbox(shape, W, H):
    """ Retourne la bounding box (left, top, right, bottom) d'une forme avec une marge """
    t, x, y, w, h, ang, _ = shape
    # Rayon max approximatif pour couvrir la rotation
    rad = math.sqrt(w*w + h*h) + 2
    x0 = int(max(0, x - rad))
    y0 = int(max(0, y - rad))
    x1 = int(min(W, x + rad))
    y1 = int(min(H, y + rad))
    return (x0, y0, x1, y1)

def compute_color_for_shape(shape, target_img):
    t, x, y, w, h, ang, _ = shape
    W, H = target_img.size
    
    # Création masque local
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    if t == "ellipse":
        draw.ellipse([x-w, y-h, x+w, y+h], fill=255)
    else:
        draw.rectangle([x-w, y-h, x+w, y+h], fill=255)
    mask = mask.rotate(ang, center=(x, y))
    
    stat = ImageStat.Stat(target_img, mask)
    if stat.count[0] == 0: return (128, 128, 128, 0.5)
    
    r, g, b = stat.mean[:3]
    
    # Logique Alpha plus "dure" pour favoriser les détails
    area_pct = (w * h) / (W * H)
    alpha = 0.7 # Base plus haute
    if area_pct < 0.001: alpha = 1.0 # Petit point = Opaque
    elif area_pct < 0.01: alpha = 0.9
    
    return (r, g, b, alpha)

def render_shape_layer(shape, size):
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    t, x, y, w, h, ang, (r, g, b, a) = shape
    fill = (int(r), int(g), int(b), int(255 * a))
    
    if t == "ellipse":
        draw.ellipse([x - w, y - h, x + w, y + h], fill=fill)
    elif t == "rect":
        draw.rectangle([x - w, y - h, x + w, y + h], fill=fill)
    
    return layer.rotate(ang, center=(x, y))

def render(genotype, size, background):
    img = Image.new("RGBA", size, background)
    for s in genotype:
        img.alpha_composite(render_shape_layer(s, size))
    return img

def fitness_local(target, canvas, bbox):
    """ Calcule l'erreur SEULEMENT dans la bbox donnée. """
    # Crop des zones concernées
    t_crop = target.crop(bbox)
    c_crop = canvas.crop(bbox)
    
    diff = ImageChops.difference(t_crop, c_crop)
    hist = diff.histogram()
    
    err = 0
    # Somme pondérée des carrés (L2)
    for i in range(len(hist)):
        if i >= 768: break # Skip alpha
        if hist[i] > 0:
            err += (i % 256) ** 2 * hist[i]
    return err

def fitness_full(target, genotype, bg):
    """ Fitness globale (lente, pour init) """
    canvas = render(genotype, target.size, bg)
    diff = ImageChops.difference(target, canvas)
    hist = diff.histogram()
    err = 0
    for i in range(len(hist)):
        if i >= 768: break
        err += (i % 256) ** 2 * hist[i]
    return err

# ============================================================
# Optimisation avec ILS + Recuit Simulé

def mutate_shape(shape, W, H, target_img, progress):
    t, x, y, w, h, ang, (r, g, b, a) = shape
    
    # Mutation dynamique : Grande au début, fine à la fin
    scale = max(0.05, 1.0 - (progress * 0.95)) 
    
    # 1% de chance de "téléportation" (Respawn)
    if random.random() < 0.01:
        x = random.uniform(0, W)
        y = random.uniform(0, H)
        w = random.uniform(2, W*0.1)
        h = random.uniform(2, H*0.1)
        ang = random.uniform(0, 360)
        # On reset l'alpha pour la nouvelle position
        col = compute_color_for_shape([t,x,y,w,h,ang,None], target_img)
        return [t, x, y, w, h, ang, col]

    # Mutation normale
    x += random.gauss(0, W * 0.05 * scale)
    y += random.gauss(0, H * 0.05 * scale)
    w *= random.uniform(1.0 - (0.2*scale), 1.0 + (0.2*scale))
    h *= random.uniform(1.0 - (0.2*scale), 1.0 + (0.2*scale))
    ang += random.uniform(-20*scale, 20*scale)
    
    # Alpha mutation
    a = clamp(a + random.uniform(-0.1, 0.1), 0.1, 1.0)
    
    # Clamp bounds
    x = clamp(x, -W*0.1, W*1.1)
    y = clamp(y, -H*0.1, H*1.1)
    w = max(1, w)
    h = max(1, h)
    
    new_geo = [t, x, y, w, h, ang, None]
    col = compute_color_for_shape(new_geo, target_img)
    new_geo[-1] = (col[0], col[1], col[2], a)
    
    return new_geo

def mutate_z_index(genotype):
    if len(genotype) < 2: return genotype
    g = genotype[:]
    idx = random.randrange(len(g))
    shape = g.pop(idx)
    new_idx = random.randrange(len(g) + 1)
    g.insert(new_idx, shape)
    return g

def optimize(target, shape_type, n_shapes, total_time, window_name, original_np):
    W, H = target.size
    bg_color = mean_background(target) # Tuple RGB
    bg_rgba = (bg_color[0], bg_color[1], bg_color[2], 255)
    
    # Initialisation Hybride (Grille + Aléatoire)
    print("Initialisation...")
    genotype = []
    
    # 1. Grille grossière pour couvrir le fond
    grid_n = int(math.sqrt(n_shapes / 2))
    step_x, step_y = W / grid_n, H / grid_n
    for i in range(grid_n):
        for j in range(grid_n):
            x = i * step_x + step_x/2
            y = j * step_y + step_y/2
            w, h = step_x * 0.6, step_y * 0.6
            s = [shape_type, x, y, w, h, 0, None]
            s[-1] = compute_color_for_shape(s, target)
            genotype.append(s)

    # 2. Le reste en aléatoire pur
    while len(genotype) < n_shapes:
        s = [shape_type, random.uniform(0,W), random.uniform(0,H), 
             random.uniform(2, W*0.1), random.uniform(2, H*0.1), 
             random.uniform(0,360), None]
        s[-1] = compute_color_for_shape(s, target)
        genotype.append(s)

    # Initial Render & Fitness
    current_canvas = render(genotype, target.size, bg_rgba)
    current_err = fitness_full(target, genotype, bg_rgba)
    FITNESS_HISTORY.append(current_err)
    
    start_time = time.time()
    iteration = 0
    
    print(f"Lancement optimisation ({total_time}s)...")

    while True:
        elapsed = time.time() - start_time
        if elapsed > total_time: break
        iteration += 1
        progress = elapsed / total_time
        
        # Température pour Recuit Simulé (baisse avec le temps)
        temp = (1 - (progress/5)) * (current_err * 0.0001)

        # Copie de sauvegarde
        candidate_geno = [s[:] for s in genotype] # Shallow copy list
        
        # Choix mutation
        # Au début on accepte de changer l'ordre (Z), à la fin on fige pour le détail
        is_z_mutation = random.random() < 0.05
        
        if is_z_mutation:
            # Mutation Globale (changement ordre) -> Nécessite recalcul complet
            candidate_geno = mutate_z_index(candidate_geno)
            new_err = fitness_full(target, candidate_geno, bg_rgba)
            delta = new_err - current_err
        else:
            # Mutation Locale -> Optimisation Bounding Box
            idx = random.randrange(len(genotype))
            old_shape = genotype[idx]
            new_shape = mutate_shape(old_shape, W, H, target, progress)
            candidate_geno[idx] = new_shape
            
            # Calcul de la zone affectée (Union des deux bbox)
            bbox_old = get_bbox(old_shape, W, H)
            bbox_new = get_bbox(new_shape, W, H)
            
            # Bbox englobante
            u_x0 = min(bbox_old[0], bbox_new[0])
            u_y0 = min(bbox_old[1], bbox_new[1])
            u_x1 = max(bbox_old[2], bbox_new[2])
            u_y1 = max(bbox_old[3], bbox_new[3])
            union_bbox = (u_x0, u_y0, u_x1, u_y1)
            
            # Si la bbox est hors champ (bug float), on ignore
            if u_x1 <= u_x0 or u_y1 <= u_y0:
                continue

            # Erreur locale AVANT mutation (sur le canvas actuel)
            local_err_before = fitness_local(target, current_canvas, union_bbox)
            
            # Rendu TEMPORAIRE de la zone (Coûteux mais moins que full render)
            # Pour l'alpha blending correct, il faut re-rendre toute la pile sur cette zone
            # Optimisation: On re-rend tout le canvas avec la nouvelle liste, mais on ne garde que si validé
            candidate_canvas = render(candidate_geno, target.size, bg_rgba)
            
            # Erreur locale APRES mutation
            local_err_after = fitness_local(target, candidate_canvas, union_bbox)
            
            # Delta estimé (différence locale)
            delta = local_err_after - local_err_before
            
            # L'erreur globale approximée
            new_err = current_err + delta

        # Metropolis Acceptance
        accept = False
        if delta < 0:
            accept = True
        elif temp > 0:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accept = True
        
        if accept:
            genotype = candidate_geno
            current_err = new_err
            # Si c'était une mutation locale, on met à jour le canvas
            # Si z-index, on l'a déjà fait ou pas besoin (calcul full)
            if not is_z_mutation:
                current_canvas = candidate_canvas
            else:
                current_canvas = render(genotype, target.size, bg_rgba)

        # UI Updates
        if iteration % 20 == 0:
            FITNESS_HISTORY.append(current_err)
            update_view(target, genotype, bg_rgba, window_name, original_np)
        
        if iteration % 50 == 0:
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    return genotype

# ============================================================
# Détection couleur de fond moyenne

def mean_background(img):
    stat = ImageStat.Stat(img)
    try:
        r, g, b = stat.mean[:3]
        return (int(r), int(g), int(b))
    except:
        return (255, 255, 255)

# ============================================================
# Programme principal

def save_svg(genotype, size, background, path):
    W, H = size
    br, bg, bb = background
    with open(path, "w") as f:
        f.write(f'<svg width="{W}" height="{H}" xmlns="http://www.w3.org/2000/svg">\n')
        f.write(f'<rect width="100%" height="100%" fill="rgb({br},{bg},{bb})"/>\n')
        for s in genotype:
            t, x, y, w, h, ang, (r, g, b, a) = s
            fill = f'rgb({int(r)},{int(g)},{int(b)})'
            op = f'{a:.3f}'
            # Inversion angle pour SVG
            if t == "ellipse":
                f.write(f'<ellipse cx="{x:.2f}" cy="{y:.2f}" rx="{w:.2f}" ry="{h:.2f}" fill="{fill}" fill-opacity="{op}" transform="rotate({-ang:.2f} {x:.2f} {y:.2f})"/>\n')
            else:
                f.write(f'<rect x="{x-w:.2f}" y="{y-h:.2f}" width="{2*w:.2f}" height="{2*h:.2f}" fill="{fill}" fill-opacity="{op}" transform="rotate({-ang:.2f} {x:.2f} {y:.2f})"/>\n')
        f.write("</svg>")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--time", type=int, default=60)
    parser.add_argument("--shape", default="ellipse", choices=["ellipse", "rect"])
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    
    files = []
    if args.input:
        files = [os.path.join("images", args.input)]
    else:
        files = glob.glob(os.path.join("images", "*.png"))

    if not files:
        print("Aucun fichier trouvé.")
        return

    for f in files:
        print(f"Traitement de {f}...")
        try:
            target = Image.open(f).convert("RGBA")
        except: continue
        
        global FITNESS_HISTORY
        FITNESS_HISTORY = []
        
        wn = f"Optimisation: {os.path.basename(f)}"
        original_np = np.array(target.convert("RGB"))
        
        best = optimize(target, args.shape, args.n, args.time, wn, original_np)
        
        out = os.path.join("outputs", os.path.splitext(os.path.basename(f))[0] + ".svg")
        bg = mean_background(target)
        save_svg(best, target.size, bg, out)
        print(f"Sauvegardé : {out}")
        cv2.destroyWindow(wn)

if __name__ == "__main__":
    main()