import os
import cairosvg
from moviepy.editor import ImageSequenceClip

# ⚡ Paramètres
steps_dir = "steps"        # dossier contenant les SVG
output_video = "output.mp4"  # nom du fichier final
fps = 5                    # images par seconde (ajuste selon besoin)
w, h = 800, 600            # résolution de sortie (tu peux mettre ton w/h)

# 1️⃣ Récupérer tous les SVG triés
svg_files = sorted([
    os.path.join(steps_dir, f)
    for f in os.listdir(steps_dir)
    if f.endswith(".svg")
])

# 2️⃣ Convertir SVG en PNG (en mémoire)
png_files = []
tmp_png_dir = os.path.join(steps_dir, "tmp_png")
os.makedirs(tmp_png_dir, exist_ok=True)

print(f"🖼️ Conversion {len(svg_files)} SVG en PNG...")
for i, svg_path in enumerate(svg_files):
    png_path = os.path.join(tmp_png_dir, f"{i:03d}.png")
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=w, output_height=h)
    png_files.append(png_path)

# 3️⃣ Créer la vidéo
print(f"🎬 Création de la vidéo {output_video} à {fps} fps...")
clip = ImageSequenceClip(png_files, fps=fps)
clip.write_videofile(output_video, codec="libx264", audio=False)

print("✅ Vidéo terminée !")
