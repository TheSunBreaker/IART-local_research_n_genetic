from utils import *
import rerender
import time
import matplotlib.pyplot as plt
import os

def get_union_bounds(bbox1, bbox2):
    """Calcule le rectangle englobant de deux rectangles"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Coordonnées min/max
    min_x = min(x1, x2)
    min_y = min(y1, y2)
    max_x = max(x1 + w1, x2 + w2)
    max_y = max(y1 + h1, y2 + h2)
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def run_fitness_improver(genome, w, h, target_img_arr, outout_img_name, iteration=999999999, renouncement_deg=50, current_time=None, time_limit=300, genome_save_path="best_genome.gen"):

    #Préparation des paramètres pour les frames
    n_steps = 50
    step_interval = time_limit / n_steps
    next_step_time = 0.0
    step_index = 0
    steps_dir = "steps"
    os.makedirs(steps_dir, exist_ok=True)


    if current_time is None: 
        current_time = time.time()

    fitness_history = []
    time_history = []

    # 1. Initialisation Complète
    # On fait un premier rendu et un premier calcul total
    best_arr = rerender.render_phenotype(genome, w, h)
    
    # On travaille avec l'erreur SSE positive (à minimiser)
    current_sse = fitness_calc(target_img_arr, best_arr)
    
    # On garde la compatibilité avec ton système (fitness négative) pour l'affichage
    fitness_history.append(-current_sse) 
    time_history.append(0.0)

    # Variables pour suivre la meilleure solution
    best_genome = genome

    #On save le premier step
    rerender.save_to_svg(
    best_genome, w, h,
    os.path.join(steps_dir, f"step_{step_index}.svg")
    )
    step_index += 1
    next_step_time += step_interval
    ######################

    print(f"🚀 Démarrage de l'optimisation. Erreur initiale : {current_sse}")

    main_loop = 0
    for main_loop in range(iteration):

        # 2. VÉRIFICATION DU TIMER
        if time_limit is not None:
            elapsed_time = time.time() - current_time

            # ⏺️ Time-based SVG capture
            while (
                step_index < n_steps and
                elapsed_time >= next_step_time
            ):
                rerender.save_to_svg(
                    best_genome, w, h,
                    os.path.join(steps_dir, f"step_{step_index}.svg")
                )
                print(f"📸 Saved step_{step_index}.svg at t={elapsed_time:.2f}s")

                step_index += 1
                next_step_time += step_interval
                #############################################
                        
            if elapsed_time > time_limit:
                print(f"⏱️ Temps écoulé ({elapsed_time:.1f}s) ! Arrêt prématuré à l'itération {main_loop}.")
                break

        # Boucle sur chaque gène (forme)
        for i in range(len(genome)):

            renouncement_counter = 0

            # --- OPTIMISATION : On récupère la bbox AVANT mutation ---
            old_shape = genome[i]
            # Attention : Assure-toi d'avoir ajouté get_bounds() dans shapes.py !
            old_bounds = old_shape.get_bounds() 

            while renouncement_counter < renouncement_deg:
                
                # A. Mutation
                tester_gen = mutate_targeted_gene(genome, w, h, i)
                
                # --- OPTIMISATION : On récupère la bbox APRÈS mutation ---
                new_shape = tester_gen[i]
                new_bounds = new_shape.get_bounds()

                # B. Calcul de la zone sale (Dirty Rect)
                # C'est l'union de l'ancienne position et de la nouvelle
                dirty_rect = get_union_bounds(old_bounds, new_bounds)

                # C. Rendu (Nécessaire pour avoir les pixels de la nouvelle superposition)
                tester_arr = rerender.render_phenotype(tester_gen, w, h)
                
                # D. FITNESS DIFFÉRENTIELLE (Le cœur du gain de temps)
                # Au lieu de tout recalculer, on demande : "Combien j'ai gagné/perdu sur ce rectangle ?"
                delta = fitness_local_change(target_img_arr, best_arr, tester_arr, dirty_rect)
                
                chall_sse = current_sse + delta

                # E. Comparaison (On veut MINIMISER l'erreur SSE)
                if chall_sse < current_sse:
                    # C'est mieux ! On valide
                    genome = tester_gen
                    best_genome = tester_gen
                    best_arr = tester_arr # On met à jour l'image de référence
                    current_sse = chall_sse
                    
                    # On met à jour les références pour la prochaine boucle while
                    old_shape = new_shape
                    old_bounds = new_bounds

                    # Logs (en négatif pour tes graphiques)
                    fitness_history.append(-current_sse)
                    time_history.append(time.time() - current_time)
                    
                    break # On passe à la forme suivante (i+1)

                renouncement_counter += 1

        # Fin de la boucle principale (toutes les formes testées)
        print(f"Itération {main_loop} — SSE (Erreur) : {current_sse}")

        # Sauvegardes
        rerender.save_to_svg(best_genome, w, h, outout_img_name + ".svg")
        
        if genome_save_path is not None:
            # Assure-toi que save_genome est bien importé de utils
            save_genome(best_genome, genome_save_path)


    rerender.save_to_png(best_genome, w, h, outout_img_name + ".png") #On save le .png seulement à la fin

    # 3. PLOT FINAL
    plt.figure(figsize=(8, 5))
    plt.plot(time_history, fitness_history)
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Fitness (Négative SSE)")
    plt.title("Convergence de la recherche locale")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fitness_convergence.png")
    plt.close()

    # On retourne la fitness négative pour rester cohérent avec ton ancien code
    return main_loop, best_genome, -current_sse