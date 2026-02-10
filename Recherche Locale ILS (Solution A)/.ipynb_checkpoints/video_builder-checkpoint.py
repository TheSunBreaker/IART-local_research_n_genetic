import os
import re
import cairosvg
from moviepy import ImageSequenceClip

# ⚡ Paramètres
steps_dir = "steps"
output_video = "output.mp4"
duration_seconds = 13

# 1️⃣ Récupérer tous les SVG triés numériquement
svg_files = [
    os.path.join(steps_dir, f)
    for f in os.listdir(steps_dir)
    if f.endswith(".svg")
]

def get_step_number(filename):
    m = re.search(r"step_(\d+)\.svg$", filename)
    return int(m.group(1)) if m else -1

svg_files = sorted(svg_files, key=get_step_number)

num_frames = len(svg_files)
if num_frames == 0:
    raise ValueError("Aucun SVG trouvé dans le dossier 'steps'.")

fps = float(num_frames) / duration_seconds
print(f"ℹ️ Nombre de frames : {num_frames}, FPS calculé : {fps:.3f}")

# 2️⃣ Convertir SVG en PNG
png_files = []
tmp_png_dir = os.path.join(steps_dir, "tmp_png")
os.makedirs(tmp_png_dir, exist_ok=True)

print(f"🖼️ Conversion {num_frames} SVG en PNG...")
for i, svg_path in enumerate(svg_files):
    png_path = os.path.join(tmp_png_dir, f"{i:03d}.png")
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    png_files.append(png_path)

# 3️⃣ Créer la vidéo
print(f"🎬 Création de la vidéo {output_video}...")
clip = ImageSequenceClip(png_files, fps=fps).set_duration(duration_seconds)
clip.write_videofile(output_video, codec="libx264", audio=False)

print("✅ Vidéo terminée !")
