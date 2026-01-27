import cv2
import numpy as np
import random  # Nécessaire pour la génération aléatoire

class Shape:
    """Classe abstraite pour toutes les formes"""
    def __init__(self, r, g, b, alpha):
        self.r = r
        self.g = g
        self.b = b
        self.alpha = alpha

    def _apply_transparency(self, canvas, overlay):
        """Méthode interne pour appliquer le mélange Alpha (Blending)"""
        alpha_norm = self.alpha / 255.0
        cv2.addWeighted(overlay, alpha_norm, canvas, 1 - alpha_norm, 0, canvas)

    def _mutate_color(self):
        """Helper pour muter légèrement la couleur et l'alpha"""
        # On choisit aléatoirement un canal à modifier pour éviter de changer toute la teinte d'un coup
        choice = random.randint(0, 3)
        delta = random.randint(-20, 20) # Amplitude de la modification
        
        if choice == 0: # Rouge
            self.r = max(0, min(255, self.r + delta))
        elif choice == 1: # Vert
            self.g = max(0, min(255, self.g + delta))
        elif choice == 2: # Bleu
            self.b = max(0, min(255, self.b + delta))
        elif choice == 3: # Alpha
            self.alpha = max(10, min(255, self.alpha + delta)) # On évite alpha=0

    def draw(self, canvas):
        raise NotImplementedError("Chaque forme doit implémenter sa méthode draw")

    def get_svg_tag(self):
        raise NotImplementedError("Chaque forme doit implémenter sa méthode SVG")

    def copy(self):
        """Renvoie une NOUVELLE instance indépendante (clone)"""
        raise NotImplementedError("Chaque forme doit implémenter sa méthode copy")

    @classmethod
    def random(cls, max_w, max_h):
        """Génère une instance aléatoire de la forme"""
        raise NotImplementedError("Chaque forme doit implémenter sa méthode random")
    
    def perturb(self, max_w, max_h):
        """Modifie légèrement la forme (Mutation)"""
        raise NotImplementedError


class Ellipse(Shape):
    def __init__(self, cx, cy, rx, ry, angle, r, g, b, alpha):
        super().__init__(r, g, b, alpha)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
        self.angle = angle

    def get_bounds(self):
        # Pour une ellipse tournée, on récupère la "bounding box" qui l'entoure
        # boxPoints ne marche pas directement sur ellipse, on utilise une astuce:
        # On crée un rect temporaire qui représente la boite orientée
        rect_struct = ((self.cx, self.cy), (self.rx*2, self.ry*2), self.angle)
        box = cv2.boxPoints(rect_struct)
        box = np.int32(box)
        return cv2.boundingRect(box) # Retourne (x, y, w, h)

    def copy(self):
        # On retourne une nouvelle instance avec les MÊMES valeurs
        return Ellipse(self.cx, self.cy, self.rx, self.ry, self.angle, 
                       self.r, self.g, self.b, self.alpha)

    @classmethod
    def random(cls, max_w, max_h):
        cx = random.randint(0, max_w)
        cy = random.randint(0, max_h)
        # Rayons limités arbitrairement à 1/4 de l'image pour éviter des formes géantes
        rx = random.randint(1, max_w // 4) 
        ry = random.randint(1, max_h // 4)
        angle = random.randint(0, 360)
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        alpha = random.randint(30, 255) # On évite 0 (invisible)
        
        return cls(cx, cy, rx, ry, angle, r, g, b, alpha)

    def draw(self, canvas):
        overlay = canvas.copy()
        # OpenCV utilise BGR, donc on inverse (self.b, self.g, self.r)
        color_bgr = (self.b, self.g, self.r)
        
        cv2.ellipse(overlay, (self.cx, self.cy), (self.rx, self.ry), 
                    self.angle, 0, 360, color_bgr, -1)
        self._apply_transparency(canvas, overlay)

    def get_svg_tag(self):
        opacity = self.alpha / 255.0
        transform = f'transform="rotate({self.angle}, {self.cx}, {self.cy})"'
        return (f'<ellipse cx="{self.cx}" cy="{self.cy}" rx="{self.rx}" ry="{self.ry}" '
                f'fill="rgb({self.r},{self.g},{self.b})" fill-opacity="{opacity:.2f}" '
                f'{transform} />')
    
    def perturb(self, max_w, max_h):
        if random.random() < 0.5:
            # 50% de chance : Mutation Géométrique
            choice = random.randint(0, 2)
            delta = random.randint(-10, 10)
            
            if choice == 0: # Position
                self.cx = max(0, min(max_w, self.cx + delta))
                self.cy = max(0, min(max_h, self.cy + delta))
            elif choice == 1: # Taille
                self.rx = max(1, self.rx + delta) # Min rayon = 1
                self.ry = max(1, self.ry + delta)
            elif choice == 2: # Rotation
                self.angle = (self.angle + delta * 2) % 360 # Rotation un peu plus forte
        else:
            # 50% de chance : Mutation Couleur
            self._mutate_color()


class Rectangle(Shape):
    def __init__(self, cx, cy, width, height, angle, r, g, b, alpha):
        super().__init__(r, g, b, alpha)
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.angle = angle

    def get_bounds(self):
        rect_struct = ((self.cx, self.cy), (self.width, self.height), self.angle)
        box = cv2.boxPoints(rect_struct)
        box = np.int32(box)
        return cv2.boundingRect(box)

    def copy(self):
        return Rectangle(self.cx, self.cy, self.width, self.height, self.angle,
                         self.r, self.g, self.b, self.alpha)

    @classmethod
    def random(cls, max_w, max_h):
        cx = random.randint(0, max_w)
        cy = random.randint(0, max_h)
        w = random.randint(1, max_w // 2)
        h = random.randint(1, max_h // 2)
        angle = random.randint(0, 360)
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        alpha = random.randint(30, 255)
        
        return cls(cx, cy, w, h, angle, r, g, b, alpha)

    def draw(self, canvas):
        overlay = canvas.copy()
        color_bgr = (self.b, self.g, self.r)
        
        # Calcul des 4 coins avec rotation
        rect_struct = ((self.cx, self.cy), (self.width, self.height), self.angle)
        box = cv2.boxPoints(rect_struct)
        box = np.int32(box)
        
        cv2.fillPoly(overlay, [box], color_bgr)
        self._apply_transparency(canvas, overlay)

    def get_svg_tag(self):
        opacity = self.alpha / 255.0
        # SVG dessine depuis le coin haut-gauche, on doit décaler
        x_top_left = self.cx - (self.width / 2)
        y_top_left = self.cy - (self.height / 2)
        
        transform = f'transform="rotate({self.angle}, {self.cx}, {self.cy})"'
        return (f'<rect x="{x_top_left}" y="{y_top_left}" '
                f'width="{self.width}" height="{self.height}" '
                f'fill="rgb({self.r},{self.g},{self.b})" fill-opacity="{opacity:.2f}" '
                f'{transform} />')
    
    def perturb(self, max_w, max_h):
        if random.random() < 0.5:
            # Mutation Géométrique
            choice = random.randint(0, 2)
            delta = random.randint(-10, 10)
            if choice == 0: # Position Centre
                self.cx = max(0, min(max_w, self.cx + delta))
                self.cy = max(0, min(max_h, self.cy + delta))
            elif choice == 1: # Dimensions
                self.width = max(1, self.width + delta)
                self.height = max(1, self.height + delta)
            elif choice == 2: # Rotation
                self.angle = (self.angle + delta * 2) % 360
        else:
            # Mutation Couleur
            self._mutate_color()


class Triangle(Shape):
    def __init__(self, x1, y1, x2, y2, x3, y3, r, g, b, alpha):
        super().__init__(r, g, b, alpha)
        self.points = [(x1, y1), (x2, y2), (x3, y3)]

    def copy(self):
        p = self.points
        # On passe les valeurs brutes pour créer un nouvel objet
        return Triangle(p[0][0], p[0][1], p[1][0], p[1][1], p[2][0], p[2][1],
                        self.r, self.g, self.b, self.alpha)

    @classmethod
    def random(cls, max_w, max_h):
        # 3 points aléatoires
        x1, y1 = random.randint(0, max_w), random.randint(0, max_h)
        x2, y2 = random.randint(0, max_w), random.randint(0, max_h)
        x3, y3 = random.randint(0, max_w), random.randint(0, max_h)
        
        r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        alpha = random.randint(30, 255)
        
        return cls(x1, y1, x2, y2, x3, y3, r, g, b, alpha)

    def draw(self, canvas):
        overlay = canvas.copy()
        color_bgr = (self.b, self.g, self.r)
        
        pts = np.array(self.points, np.int32)
        cv2.fillPoly(overlay, [pts], color_bgr)
        self._apply_transparency(canvas, overlay)

    def get_svg_tag(self):
        opacity = self.alpha / 255.0
        p_str = f"{self.points[0][0]},{self.points[0][1]} {self.points[1][0]},{self.points[1][1]} {self.points[2][0]},{self.points[2][1]}"
        return (f'<polygon points="{p_str}" '
                f'fill="rgb({self.r},{self.g},{self.b})" fill-opacity="{opacity:.2f}" />')
    
    def perturb(self, max_w, max_h):
        if random.random() < 0.5:
            # Mutation Géométrique : On bouge UN seul point du triangle au hasard
            # C'est souvent plus efficace que de bouger tout le triangle
            pt_idx = random.randint(0, 2)
            delta_x = random.randint(-15, 15)
            delta_y = random.randint(-15, 15)
            
            # On récupère le point, on modifie, on remet dans la liste
            x, y = self.points[pt_idx]
            new_x = max(0, min(max_w, x + delta_x))
            new_y = max(0, min(max_h, y + delta_y))
            self.points[pt_idx] = (new_x, new_y)
        else:
            # Mutation Couleur
            self._mutate_color()
            

def get_random_shape(max_w, max_h):
    """Fonction utilitaire pour retourner une Ellipse, un Rectangle ou un Triangle au hasard"""
    classes = [Ellipse, Rectangle, Triangle]
    ChosenClass = random.choice(classes)
    return ChosenClass.random(max_w, max_h)