import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection, PatchCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import sys
import os
import threading
from datetime import datetime
import glob  # Ajout nécessaire pour l'analyse

from core.solver import run_full_simulation

# --- IMPORT DU LOGGER ---
from utils.logger import get_logger

logger = get_logger(__name__)
# ------------------------

# Configuration du thème sombre strict pour Matplotlib
plt.style.use("dark_background")

# Ajout du dossier racine au PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from physics.geometry import ReactorGeometry, ReactorMeshGenerator

# Import de la base de données des matériaux
try:
    from physics.materials import MATERIAL_DB
except ImportError:
    # Fallback au cas où le fichier serait à la racine ou dans un autre module
    from physics.materials import MATERIAL_DB


class ReactorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        logger.info("Démarrage de l'interface graphique ReactorGUI.")
        self.title("Générateur de Géométrie - Cœur de Réacteur")
        # Fenêtre agrandie pour intégrer le panneau des matériaux
        self.geometry("550x875")
        self.minsize(500, 850)
        self.resizable(True, True)

        # Raccourcis Plein Écran
        self.bind(
            "<F11>",
            lambda event: self.attributes(
                "-fullscreen", not self.attributes("-fullscreen")
            ),
        )
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        # --- Thème VS Code (Dark+) ---
        self.BG_COLOR = "#1e1e1e"
        self.PANEL_BG = "#252526"
        self.FG_COLOR = "#cccccc"
        self.TITLE_COLOR = "#d4d4d4"
        self.ENTRY_BG = "#3c3c3c"
        self.BORDER_COLOR = "#2d2d2d"
        self.ACCENT_BLUE = "#0e639c"
        self.ACCENT_HOVER = "#1177bb"

        self.configure(bg=self.BG_COLOR)

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        default_font = ("Segoe UI", 10)
        bold_font = ("Segoe UI", 10, "bold")

        style.configure(
            ".", font=default_font, background=self.BG_COLOR, foreground=self.FG_COLOR
        )
        style.configure("TFrame", background=self.BG_COLOR)
        style.configure(
            "TEntry",
            fieldbackground=self.ENTRY_BG,
            foreground=self.FG_COLOR,
            bordercolor=self.BORDER_COLOR,
            lightcolor=self.ENTRY_BG,
            darkcolor=self.ENTRY_BG,
        )
        style.configure(
            "TSpinbox",
            fieldbackground=self.ENTRY_BG,
            foreground=self.FG_COLOR,
            bordercolor=self.BORDER_COLOR,
            arrowcolor=self.FG_COLOR,
            lightcolor=self.ENTRY_BG,
            darkcolor=self.ENTRY_BG,
        )
        style.configure(
            "TLabelframe",
            background=self.PANEL_BG,
            bordercolor=self.BORDER_COLOR,
            lightcolor=self.PANEL_BG,
            darkcolor=self.BORDER_COLOR,
        )
        style.configure(
            "TLabelframe.Label",
            font=bold_font,
            foreground=self.TITLE_COLOR,
            background=self.PANEL_BG,
        )
        style.configure("TSeparator", background=self.BORDER_COLOR)
        style.configure(
            "TButton",
            font=default_font,
            padding=6,
            background="#4d4d4d",
            foreground=self.FG_COLOR,
            bordercolor=self.BORDER_COLOR,
            lightcolor="#4d4d4d",
            darkcolor="#4d4d4d",
        )
        style.map("TButton", background=[("active", "#5a5a5a")])

        style.configure(
            "Accent.TButton",
            font=bold_font,
            background=self.ACCENT_BLUE,
            foreground="white",
            bordercolor=self.BORDER_COLOR,
        )
        style.map("Accent.TButton", background=[("active", self.ACCENT_HOVER)])

        style.configure(
            "Success.TButton",
            font=bold_font,
            background="#125e2a",
            foreground="white",
            bordercolor=self.BORDER_COLOR,
        )
        style.map("Success.TButton", background=[("active", "#187a37")])

        style.configure(
            "Warning.TButton",
            font=bold_font,
            background="#9e5a0e",
            foreground="white",
            bordercolor=self.BORDER_COLOR,
        )
        style.map("Warning.TButton", background=[("active", "#bf6d11")])

        # --- Style spécifique pour les Combobox ---
        style.configure(
            "TCombobox",
            fieldbackground=self.ENTRY_BG,
            foreground=self.FG_COLOR,
            background=self.BORDER_COLOR,
            bordercolor=self.BORDER_COLOR,
            lightcolor=self.ENTRY_BG,
            darkcolor=self.ENTRY_BG,
            arrowcolor=self.FG_COLOR,
        )

        # Forcer les couleurs pour l'état 'readonly'
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.ENTRY_BG)],
            foreground=[("readonly", self.FG_COLOR)],
            selectbackground=[("readonly", self.ACCENT_BLUE)],
            selectforeground=[("readonly", "white")],
        )

        # Correction de la liste déroulante (Listbox) générée par Tkinter de base
        self.option_add("*TCombobox*Listbox.background", self.ENTRY_BG)
        self.option_add("*TCombobox*Listbox.foreground", self.FG_COLOR)
        self.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT_BLUE)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")
        self.option_add("*TCombobox*Listbox.font", default_font)

        # --- Variables Tkinter ---
        self.var_R_hex = tk.DoubleVar(value=2.0)
        self.var_R_noyau = tk.DoubleVar(value=12.0)
        self.var_epaisseur_reflec = tk.DoubleVar(value=4.0)

        self.var_pins_fuel = tk.IntVar(value=4)
        self.var_pins_cr = tk.IntVar(value=3)

        self.var_cr_rings = tk.IntVar(value=1)
        self.var_cr_density = tk.DoubleVar(value=0.5)

        # Variables pour les matériaux
        available_materials = list(MATERIAL_DB.keys()) if MATERIAL_DB else []

        self.var_mat_fuel = tk.StringVar(
            value="Fuel_Uranium" if "Fuel_Uranium" in available_materials else ""
        )
        self.var_mat_mod = tk.StringVar(
            value="Water_Moderator" if "Water_Moderator" in available_materials else ""
        )
        self.var_mat_ref = tk.StringVar(
            value="Graphite_Moderator"
            if "Graphite_Moderator" in available_materials
            else ""
        )
        self.var_mat_cr = tk.StringVar(
            value="Boral_ControlRod"
            if "Boral_ControlRod" in available_materials
            else ""
        )

        self.available_materials = available_materials

        self.create_widgets()

    def action_solve(self):
        """Ouvre un sélecteur de fichier pour choisir le maillage à simuler."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        initial_dir = os.path.join(base_dir, "output", "meshes")
        os.makedirs(initial_dir, exist_ok=True)

        output_file = filedialog.askopenfilename(
            title="Sélectionner un maillage à simuler",
            initialdir=initial_dir,
            filetypes=[("Fichiers Gmsh", "*.msh"), ("Tous les fichiers", "*.*")],
        )

        if not output_file:
            return  # L'utilisateur a annulé la sélection

        logger.info(
            f"Fichier de maillage sélectionné pour la simulation : {output_file}"
        )

        # Récupération de la configuration personnalisée des matériaux
        # MODIFICATION STRICTEMENT NÉCESSAIRE : Ajout de la valeur par défaut pour éviter le KeyError
        user_materials = {
            "Fuel": self.var_mat_fuel.get() or "Fuel_Uranium",
            "Moderator": self.var_mat_mod.get() or "Water_Moderator",
            "Reflector": self.var_mat_ref.get() or "Graphite_Moderator",
            "ControlRods": self.var_mat_cr.get() or "Boral_ControlRod",
        }

        try:
            # Injection de la configuration dans la fonction du solveur
            run_full_simulation(output_file, user_mapping=user_materials)
        except Exception as e:
            logger.error(
                f"Le solver a échoué lors de l'exécution unitaire : {e}", exc_info=True
            )
            messagebox.showerror("Erreur de calcul", f"Le solver a échoué :\n{e}")

    def action_view_mesh(self):
        """Ouvre la visionneuse native de Gmsh pour inspecter un maillage généré."""
        import gmsh

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        initial_dir = os.path.join(base_dir, "output", "meshes")
        os.makedirs(initial_dir, exist_ok=True)

        mesh_file = filedialog.askopenfilename(
            title="Sélectionner un maillage à visualiser",
            initialdir=initial_dir,
            filetypes=[("Fichiers Gmsh", "*.msh"), ("Tous les fichiers", "*.*")],
        )

        if not mesh_file:
            return

        logger.info(f"Ouverture de Gmsh pour visualiser : {mesh_file}")
        try:
            if gmsh.isInitialized():
                gmsh.finalize()

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(mesh_file)
            gmsh.fltk.run()
            gmsh.finalize()
        except Exception as e:
            logger.error(
                f"Impossible d'afficher le maillage 2D avec Gmsh : {e}", exc_info=True
            )
            messagebox.showerror(
                "Erreur Gmsh", f"Impossible d'afficher le maillage :\n{e}"
            )

    def create_widgets(self):
        main_container = ttk.Frame(self, padding=(20, 20))
        main_container.pack(fill="both", expand=True)

        lbl_hint = tk.Label(
            main_container,
            text="F11 : Plein écran | Échap : Quitter plein écran",
            bg=self.BG_COLOR,
            fg="#666666",
            font=("Segoe UI", 8, "italic"),
        )
        lbl_hint.pack(side="top", anchor="e", pady=(0, 10))

        def create_panel(parent, text):
            frame = ttk.Frame(parent, style="TFrame")
            frame.pack(fill="x", pady=(0, 10))
            lf = ttk.LabelFrame(frame, text=text, padding=(15, 10))
            lf.pack(fill="both", expand=True)
            return lf

        # --- Dimensions Globales ---
        frame_dim = create_panel(main_container, "Dimensions Globales (cm)")
        ttk.Label(
            frame_dim, text="Rayon d'un hexagone :", background=self.PANEL_BG
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(frame_dim, textvariable=self.var_R_hex, width=12).grid(
            row=0, column=1, sticky="e", pady=2
        )

        ttk.Label(
            frame_dim, text="Rayon limite du cœur :", background=self.PANEL_BG
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(frame_dim, textvariable=self.var_R_noyau, width=12).grid(
            row=1, column=1, sticky="e", pady=2
        )

        ttk.Label(
            frame_dim, text="Épaisseur réflecteur :", background=self.PANEL_BG
        ).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(frame_dim, textvariable=self.var_epaisseur_reflec, width=12).grid(
            row=2, column=1, sticky="e", pady=2
        )
        frame_dim.grid_columnconfigure(0, weight=1)

        # --- Sélection des Matériaux ---
        frame_materials = create_panel(main_container, "Sélection des Matériaux")

        ttk.Label(frame_materials, text="Combustible :", background=self.PANEL_BG).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Combobox(
            frame_materials,
            textvariable=self.var_mat_fuel,
            values=self.available_materials,
            state="readonly",
            width=25,
        ).grid(row=0, column=1, sticky="e", pady=2)

        ttk.Label(
            frame_materials, text="Modérateur (Cœur) :", background=self.PANEL_BG
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frame_materials,
            textvariable=self.var_mat_mod,
            values=self.available_materials,
            state="readonly",
            width=25,
        ).grid(row=1, column=1, sticky="e", pady=2)

        ttk.Label(frame_materials, text="Réflecteur :", background=self.PANEL_BG).grid(
            row=2, column=0, sticky="w", pady=2
        )
        ttk.Combobox(
            frame_materials,
            textvariable=self.var_mat_ref,
            values=self.available_materials,
            state="readonly",
            width=25,
        ).grid(row=2, column=1, sticky="e", pady=2)

        ttk.Label(
            frame_materials, text="Barres de contrôle :", background=self.PANEL_BG
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Combobox(
            frame_materials,
            textvariable=self.var_mat_cr,
            values=self.available_materials,
            state="readonly",
            width=25,
        ).grid(row=3, column=1, sticky="e", pady=2)

        frame_materials.grid_columnconfigure(0, weight=1)

        # --- Structure Interne ---
        frame_pins = create_panel(
            main_container, "Structure Interne (Couronnes de crayons)"
        )
        ttk.Label(
            frame_pins, text="Combustible (FUEL) :", background=self.PANEL_BG
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            frame_pins, from_=1, to=10, textvariable=self.var_pins_fuel, width=10
        ).grid(row=0, column=1, sticky="e", pady=2)

        ttk.Label(
            frame_pins, text="Barre de contrôle (CR) :", background=self.PANEL_BG
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            frame_pins, from_=1, to=10, textvariable=self.var_pins_cr, width=10
        ).grid(row=1, column=1, sticky="e", pady=2)
        frame_pins.grid_columnconfigure(0, weight=1)

        # --- Répartition CR ---
        frame_cr = create_panel(main_container, "Répartition Automatique des CR")
        ttk.Label(
            frame_cr, text="Nombre d'anneaux CR :", background=self.PANEL_BG
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Spinbox(
            frame_cr, from_=0, to=5, textvariable=self.var_cr_rings, width=10
        ).grid(row=0, column=1, sticky="e", pady=2)

        ttk.Label(
            frame_cr,
            text="Densité dans l'anneau (0.0 - 1.0) :",
            background=self.PANEL_BG,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(frame_cr, textvariable=self.var_cr_density, width=12).grid(
            row=1, column=1, sticky="e", pady=2
        )
        frame_cr.grid_columnconfigure(0, weight=1)

        # --- Boutons d'Action ---
        frame_actions = ttk.Frame(main_container)
        frame_actions.pack(fill="both", side="bottom", pady=5, expand=True)

        btn_container = ttk.Frame(frame_actions)
        btn_container.pack(expand=True, anchor="s")

        btn_width = 45

        ttk.Button(
            btn_container,
            text="Aperçu Visuel Rapide",
            width=btn_width,
            command=self.action_preview,
        ).pack(pady=(0, 5))
        ttk.Button(
            btn_container,
            text="Générer Maillage (Auto)",
            width=btn_width,
            command=self.action_gmsh_auto,
            style="Accent.TButton",
        ).pack(pady=(0, 5))
        ttk.Button(
            btn_container,
            text="Sélection Manuelle (Pinceau)",
            width=btn_width,
            command=self.action_manual_selection,
            style="Success.TButton",
        ).pack(pady=(0, 5))
        ttk.Button(
            btn_container,
            text="🔎 Afficher le Maillage 2D (Gmsh)",
            width=btn_width,
            command=self.action_view_mesh,
        ).pack(pady=(0, 15))

        btn_solve = ttk.Button(
            btn_container,
            text="🚀 Lancer la Simulation",
            width=btn_width,
            command=self.action_solve,
            style="Accent.TButton",
        )
        btn_solve.pack(pady=(5, 0))

        ttk.Separator(btn_container, orient="horizontal").pack(fill="x", pady=5)
        ttk.Button(
            btn_container,
            text="🔬 Cas A : Étude de Pilotabilité (Statique)",
            width=btn_width,
            command=self.action_run_case_A,
            style="Accent.TButton",
        ).pack(pady=(5, 0))

        ttk.Button(
            btn_container,
            text="🔬 Cas 3 : Optimisation du Réflecteur",
            width=btn_width,
            command=self.action_run_case_3,
            style="Accent.TButton",
        ).pack(pady=(5, 0))

        ttk.Button(
            btn_container,
            text="Étude Paramétrique (Batch)",
            width=btn_width,
            command=self.action_parametric_setup,
            style="Warning.TButton",
        ).pack(pady=(5, 0))

        ttk.Button(
            btn_container,
            text="📈 Analyser l'Étude Paramétrique",
            width=btn_width,
            command=self.action_analyze_parametric,
            style="Success.TButton",
        ).pack(pady=(5, 0))

    def get_current_params(self):
        try:
            r_n = self.var_R_noyau.get()
            epaisseur = self.var_epaisseur_reflec.get()
            return {
                "R_hex": self.var_R_hex.get(),
                "R_noyau": r_n,
                "epaisseur_reflec": epaisseur,
                "R_reflec": r_n + epaisseur,
                "pins_fuel": self.var_pins_fuel.get(),
                "pins_cr": self.var_pins_cr.get(),
                "cr_rings": self.var_cr_rings.get(),
                "cr_density": self.var_cr_density.get(),
            }
        except tk.TclError:
            messagebox.showerror(
                "Erreur de saisie", "Veuillez entrer des valeurs numériques valides."
            )
            return None

    def _check_reflector_logic(self, p):
        if p["epaisseur_reflec"] <= 0:
            messagebox.showwarning(
                "Incohérence",
                "L'épaisseur du réflecteur doit être strictement positive.",
            )
            return False
        return True

    def action_preview(self):
        p = self.get_current_params()
        if not p or not self._check_reflector_logic(p):
            return

        geom = ReactorGeometry(R_n=p["R_noyau"], R_hex=p["R_hex"])
        centers, tags, _ = geom.get_tagged_assemblies(
            n_cr_rings=p["cr_rings"], cr_density=p["cr_density"]
        )

        if len(centers) == 0:
            messagebox.showinfo("Vide", "Aucun assemblage ne rentre dans dimensions.")
            return

        self._plot_static_preview(p, geom, centers, tags)

    def _plot_static_preview(self, p, geom, centers, tags):
        plt.close("Aperçu Matplotlib - Cœur de Réacteur")

        mask_fuel, mask_cr = tags == "FUEL", tags == "CR"
        fuel_centers, cr_centers = centers[mask_fuel], centers[mask_cr]

        angles = np.pi / 6 + np.arange(6) * (np.pi / 3)
        base_hex = np.column_stack((np.cos(angles), np.sin(angles))) * p["R_hex"]

        fig, ax = plt.subplots(
            num="Aperçu Matplotlib - Cœur de Réacteur", figsize=(8, 8)
        )
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.BG_COLOR)

        try:
            reflector = geom.get_reflector_patch(R_reflector=p["R_reflec"])
            reflector.set_facecolor(self.PANEL_BG)
            reflector.set_edgecolor(self.BORDER_COLOR)
            ax.add_patch(reflector)
        except Exception as e:
            messagebox.showerror("Erreur Réflecteur", str(e))
            return

        ax.add_collection(
            PolyCollection(
                fuel_centers[:, None, :] + base_hex[None, :, :],
                facecolors="#264f78",
                edgecolors=self.BG_COLOR,
                linewidths=1.5,
                alpha=0.9,
                zorder=1,
            )
        )
        if len(cr_centers) > 0:
            ax.add_collection(
                PolyCollection(
                    cr_centers[:, None, :] + base_hex[None, :, :],
                    facecolors="#842029",
                    edgecolors=self.BG_COLOR,
                    linewidths=1.5,
                    alpha=0.9,
                    zorder=1,
                )
            )

        offsets_fuel, r_fuel = geom.get_local_pin_offsets(p["pins_fuel"])
        offsets_cr, r_cr = geom.get_local_pin_offsets(p["pins_cr"])

        if len(fuel_centers) > 0 and len(offsets_fuel) > 0:
            ax.add_collection(
                PatchCollection(
                    [
                        mpatches.Circle(xy, r_fuel)
                        for xy in (
                            fuel_centers[:, None, :] + offsets_fuel[None, :, :]
                        ).reshape(-1, 2)
                    ],
                    facecolor="#3a82c4",
                    alpha=0.9,
                    zorder=3,
                )
            )
        if len(cr_centers) > 0 and len(offsets_cr) > 0:
            ax.add_collection(
                PatchCollection(
                    [
                        mpatches.Circle(xy, r_cr)
                        for xy in (
                            cr_centers[:, None, :] + offsets_cr[None, :, :]
                        ).reshape(-1, 2)
                    ],
                    facecolor="#c43a46",
                    alpha=0.9,
                    zorder=3,
                )
            )

        ax.set_aspect("equal")
        limit = p["R_reflec"] * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(
            f"Aperçu : {len(centers)} assemblages",
            fontsize=12,
            fontweight="bold",
            color=self.FG_COLOR,
        )

        ax.axis("off")
        plt.tight_layout()
        plt.show()

    def action_gmsh_auto(self):
        p = self.get_current_params()
        if not p or not self._check_reflector_logic(p):
            return
        geom = ReactorGeometry(R_n=p["R_noyau"], R_hex=p["R_hex"])

        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(base_dir, "output", "meshes")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reactor_Rn{p['R_noyau']}_Rhex{p['R_hex']}_{timestamp}.msh"
        output_file = os.path.join(output_dir, filename)

        logger.info(f"Début de la génération automatique (Gmsh) : {output_file}")
        self._run_gmsh(geom, p, output_file, show_popup=True)

    def action_manual_selection(self):
        p = self.get_current_params()
        if not p or not self._check_reflector_logic(p):
            return

        geom = ReactorGeometry(R_n=p["R_noyau"], R_hex=p["R_hex"])
        centers = geom.hex_centers()
        if len(centers) == 0:
            messagebox.showinfo("Vide", "Aucun assemblage à afficher.")
            return

        _, auto_tags, _ = geom.get_tagged_assemblies(
            n_cr_rings=p["cr_rings"], cr_density=p["cr_density"]
        )
        if len(auto_tags) == len(centers):
            tags = auto_tags
        else:
            tags = np.full(len(centers), "FUEL", dtype=object)

        top = tk.Toplevel(self)
        top.title("Peinture Manuelle du Cœur")
        top.geometry("750x850")
        top.configure(bg=self.BG_COLOR)

        lbl_info = tk.Label(
            top,
            text="Cliquez et glissez (peinture) sur les hexagones pour basculer :\nBleu (Combustible) ↔ Rouge (Barre de Contrôle)",
            font=("Segoe UI", 10, "italic"),
            bg=self.BG_COLOR,
            fg="#888888",
            justify="center",
        )
        lbl_info.pack(pady=15)

        fig = Figure(figsize=(6, 6))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.BG_COLOR)

        angles = np.pi / 6 + np.arange(6) * (np.pi / 3)
        base_hex = np.column_stack((np.cos(angles), np.sin(angles))) * p["R_hex"]
        verts = centers[:, None, :] + base_hex[None, :, :]

        color_fuel = "#264f78"
        color_cr = "#842029"

        colors = np.where(tags == "FUEL", color_fuel, color_cr)
        collection = PolyCollection(
            verts, facecolors=colors, edgecolors=self.BG_COLOR, linewidths=1.5
        )
        ax.add_collection(collection)

        ax.set_aspect("equal")
        limit = p["R_noyau"] * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axis("off")

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=20)

        apotheme = p["R_hex"] * np.sqrt(3) / 2

        paint_state = {
            "is_dragging": False,
            "target_tag": None,
            "last_painted_idx": None,
        }

        def update_plot():
            new_colors = np.where(tags == "FUEL", color_fuel, color_cr)
            collection.set_facecolors(new_colors)
            canvas.draw_idle()

        def get_closest_hex(event):
            if event.inaxes != ax:
                return None
            dist = np.sqrt(
                (centers[:, 0] - event.xdata) ** 2 + (centers[:, 1] - event.ydata) ** 2
            )
            idx = np.argmin(dist)
            if dist[idx] <= apotheme:
                return idx
            return None

        def on_press(event):
            idx = get_closest_hex(event)
            if idx is not None:
                paint_state["is_dragging"] = True
                paint_state["target_tag"] = "CR" if tags[idx] == "FUEL" else "FUEL"
                paint_state["last_painted_idx"] = idx
                tags[idx] = paint_state["target_tag"]
                update_plot()

        def on_motion(event):
            if not paint_state["is_dragging"]:
                return
            idx = get_closest_hex(event)
            if idx is not None and idx != paint_state["last_painted_idx"]:
                tags[idx] = paint_state["target_tag"]
                paint_state["last_painted_idx"] = idx
                update_plot()

        def on_release(event):
            paint_state["is_dragging"] = False
            paint_state["last_painted_idx"] = None

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)

        frame_btns = ttk.Frame(top)
        frame_btns.pack(fill="x", pady=20, padx=20)

        def preview_custom():
            original_func = geom.get_tagged_assemblies
            geom.get_tagged_assemblies = lambda **kwargs: (centers, tags, None)
            self._plot_static_preview(p, geom, centers, tags)
            geom.get_tagged_assemblies = original_func

        def generate_custom_mesh():
            geom.get_tagged_assemblies = lambda **kwargs: (centers, tags, None)
            top.destroy()
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            output_dir = os.path.join(base_dir, "output", "meshes")
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"reactor_manual_Rn{p['R_noyau']}_Rhex{p['R_hex']}_{timestamp}.msh"
            )
            output_file = os.path.join(output_dir, filename)

            logger.info(f"Début de la génération manuelle (Gmsh) : {output_file}")
            self._run_gmsh(geom, p, output_file, show_popup=True)

        ttk.Button(
            frame_btns, text="👁 Aperçu Détaillé (Crayons)", command=preview_custom
        ).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(
            frame_btns,
            text="✔ Valider & Générer Maillage",
            command=generate_custom_mesh,
            style="Success.TButton",
        ).pack(side="right", expand=True, fill="x", padx=5)

    def _run_gmsh(self, geom, p, output_file, show_popup=False):
        try:
            mesh_gen = ReactorMeshGenerator(geometry=geom, params=p)
            mesh_gen.generate(output_filename=output_file)
            if show_popup:
                messagebox.showinfo(
                    "Maillage Terminé", f"Sauvegardé dans :\n{output_file}"
                )
        except Exception as e:
            logger.error(f"Erreur Gmsh lors du maillage : {e}", exc_info=True)
            if show_popup:
                messagebox.showerror(
                    "Erreur Gmsh", f"Erreur lors du maillage :\n\n{str(e)}"
                )
            else:
                print(f"Erreur silencieuse Gmsh : {e}")

    def action_parametric_setup(self):
        p_base = self.get_current_params()
        if not p_base:
            return

        top = tk.Toplevel(self)
        top.title("Configuration - Étude Paramétrique")
        top.geometry("850x450")
        top.minsize(800, 450)
        top.configure(bg=self.BG_COLOR)

        top.is_running = True

        def on_closing():
            top.is_running = False
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_closing)

        frame_left = ttk.Frame(top, padding=10)
        frame_left.pack(side="left", fill="y")

        frame_right = ttk.Frame(top, padding=10)
        frame_right.pack(side="right", fill="both", expand=True)

        var_rmin = tk.DoubleVar(value=10.0)
        var_rmax = tk.DoubleVar(value=30.0)
        var_rstep = tk.DoubleVar(value=2.0)
        var_epaisseur = tk.DoubleVar(value=p_base.get("epaisseur_reflec", 5.0))

        frame_inputs = ttk.LabelFrame(
            frame_left, text="Paramètres de boucle", padding=15
        )
        frame_inputs.pack(fill="x", pady=10)

        ttk.Label(frame_inputs, text="Rn Minimum :", background=self.PANEL_BG).grid(
            row=0, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame_inputs, textvariable=var_rmin, width=8).grid(
            row=0, column=1, sticky="e"
        )

        ttk.Label(frame_inputs, text="Rn Maximum :", background=self.PANEL_BG).grid(
            row=1, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame_inputs, textvariable=var_rmax, width=8).grid(
            row=1, column=1, sticky="e"
        )

        ttk.Label(frame_inputs, text="Pas (ΔRn) :", background=self.PANEL_BG).grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame_inputs, textvariable=var_rstep, width=8).grid(
            row=2, column=1, sticky="e"
        )

        ttk.Separator(frame_inputs, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=10
        )

        ttk.Label(
            frame_inputs, text="Épaisseur Réflec. :", background=self.PANEL_BG
        ).grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame_inputs, textvariable=var_epaisseur, width=8).grid(
            row=4, column=1, sticky="e"
        )

        progress_var = tk.DoubleVar(value=0.0)
        pb = ttk.Progressbar(frame_left, variable=progress_var, maximum=100)
        pb.pack(fill="x", pady=(15, 5))

        lbl_status = tk.Label(
            frame_left,
            text="Prêt à lancer...",
            font=("Segoe UI", 9, "italic"),
            bg=self.BG_COLOR,
            fg="#888888",
        )
        lbl_status.pack(pady=5)

        fig = Figure(figsize=(5, 5))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.BG_COLOR)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Aperçu en direct", fontsize=10, color=self.FG_COLOR)

        canvas = FigureCanvasTkAgg(fig, master=frame_right)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        top.safe_refs = [
            var_rmin,
            var_rmax,
            var_rstep,
            var_epaisseur,
            progress_var,
            fig,
            canvas,
            ax,
        ]

        def start_thread():
            r_min, r_max, r_step = var_rmin.get(), var_rmax.get(), var_rstep.get()
            epaisseur = var_epaisseur.get()

            # --- Capture des matériaux sélectionnés pour le solver en batch ---
            # MODIFICATION STRICTEMENT NÉCESSAIRE : Ajout de la valeur par défaut pour éviter le KeyError
            user_materials = {
                "Fuel": self.var_mat_fuel.get() or "Fuel_Uranium",
                "Moderator": self.var_mat_mod.get() or "Water_Moderator",
                "Reflector": self.var_mat_ref.get() or "Graphite_Moderator",
                "ControlRods": self.var_mat_cr.get() or "Boral_ControlRod",
            }

            if r_min >= r_max or r_step <= 0:
                messagebox.showerror("Erreur", "Paramètres invalides.", parent=top)
                return

            logger.info(
                f"Démarrage de l'étude paramétrique : R_n de {r_min} à {r_max} par pas de {r_step}"
            )
            btn_start.state(["disabled"])
            lbl_status.config(text="Initialisation...")

            max_limit = r_max + epaisseur
            ax.set_xlim(-max_limit, max_limit)
            ax.set_ylim(-max_limit, max_limit)

            thread = threading.Thread(
                target=self._parametric_worker,
                args=(
                    r_min,
                    r_max,
                    r_step,
                    epaisseur,
                    p_base,
                    user_materials,  # Passage du mapping au thread
                    top,
                    progress_var,
                    lbl_status,
                    ax,
                    canvas,
                ),
            )
            thread.daemon = True
            thread.start()

        btn_start = ttk.Button(
            frame_left,
            text="Démarrer le Batch",
            command=start_thread,
            style="Warning.TButton",
        )
        btn_start.pack(pady=(20, 0), fill="x")

    def _parametric_worker(
        self,
        r_min,
        r_max,
        r_step,
        epaisseur,
        p_base,
        user_materials,
        top_window,
        progress_var,
        lbl_status,
        ax,
        canvas,
    ):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            base_dir, "output", "meshes", f"parametric_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Création du dossier de l'étude paramétrique : {output_dir}")

        r_vals = np.arange(r_min, r_max + 1e-9, r_step)
        total_steps = len(r_vals)
        seen_assemblies = set()
        generated_count = 0

        for i, r_n in enumerate(r_vals):
            if not getattr(top_window, "is_running", False):
                logger.warning("Thread paramétrique interrompu (Fenêtre fermée).")
                print("Thread paramétrique interrompu (Fenêtre fermée).")
                return

            logger.debug(f"Traitement du rayon R_n = {r_n}...")
            geom = ReactorGeometry(R_n=r_n, R_hex=p_base["R_hex"])
            centers, tags, _ = geom.get_tagged_assemblies(
                n_cr_rings=p_base["cr_rings"], cr_density=p_base["cr_density"]
            )
            n_assemblies = len(centers)

            if n_assemblies in seen_assemblies or n_assemblies == 0:
                self.after(
                    0,
                    self._update_ui_progress,
                    progress_var,
                    lbl_status,
                    i + 1,
                    total_steps,
                    "Saut (topologie identique)",
                    top_window,
                )
                continue

            seen_assemblies.add(n_assemblies)

            p_current = p_base.copy()
            p_current["R_noyau"] = r_n
            p_current["R_reflec"] = r_n + epaisseur

            self.after(
                0,
                self._update_live_plot,
                ax,
                canvas,
                p_current,
                geom,
                centers,
                tags,
                top_window,
            )

            prefix = f"{generated_count + 1:02d}"
            filename = os.path.join(
                output_dir, f"{prefix}_mesh_N{n_assemblies}_Rn{r_n:.1f}.msh"
            )

            # Génération du maillage
            self._run_gmsh(geom, p_current, filename, show_popup=False)

            # Exécution de la simulation en mode headless et sauvegarde CSV
            self.after(
                0,
                self._update_ui_progress,
                progress_var,
                lbl_status,
                i + 1,
                total_steps,
                f"Simulation Rn={r_n}...",
                top_window,
            )

            csv_filename = filename.replace(".msh", "_results.csv")
            try:
                run_full_simulation(
                    filename,
                    user_mapping=user_materials,
                    headless=True,
                    save_csv=csv_filename,
                )
            except Exception as e:
                logger.error(
                    f"!!! ERREUR LORS DE LA SIMULATION (Rn={r_n}) !!!\nDétails : {e}",
                    exc_info=True,
                )
                print(f"Erreur simulation Rn={r_n}: {e}")

            generated_count += 1

        logger.info("Étude paramétrique terminée.")
        self.after(0, self._finish_parametric, top_window, generated_count, output_dir)

    def action_analyze_parametric(self):
        """Ouvre un dossier, lit les CSV, affiche les courbes et calcule le meilleur rayon."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        initial_dir = os.path.join(base_dir, "output", "meshes")

        study_dir = filedialog.askdirectory(
            title="Sélectionner le dossier de l'étude", initialdir=initial_dir
        )
        if not study_dir:
            return

        logger.info(
            f"Lancement de l'analyse de l'étude paramétrique sur le dossier : {study_dir}"
        )

        csv_files = glob.glob(os.path.join(study_dir, "*_results.csv"))

        if not csv_files:
            logger.warning("Aucun fichier CSV trouvé dans le dossier sélectionné.")
            messagebox.showwarning(
                "Dossier vide", "Aucun fichier de résultat CSV trouvé dans ce dossier."
            )
            return

        # Puissance cible (Doit correspondre à PUISSANCE_CIBLE de solver.py)
        P_CIBLE = 50000000000.0

        best_rn = None
        best_score = float("inf")
        best_times = None
        best_power = None

        all_data = []

        # 1. Lecture et évaluation de la stabilité
        for f in csv_files:
            logger.debug(f"Analyse du fichier CSV : {f}")
            try:
                rn_str = os.path.basename(f).split("_Rn")[1].split("_results.csv")[0]
                rn_val = float(rn_str)
            except Exception as e:
                logger.warning(
                    f"Erreur d'extraction du rayon pour le nom de fichier {f}: {e}"
                )
                print(f"Erreur d'extraction du rayon pour {f}: {e}")
                rn_val = "Inconnu"

            try:
                data = np.loadtxt(f, delimiter=",", skiprows=1)
                times = data[:, 0]
                power = data[:, 1]
                all_data.append((rn_val, times, power))

                # Formule de Stabilité : Intégrale de l'erreur absolue sur la 2ème moitié du temps
                demi_idx = len(power) // 2
                erreur_absolue = np.sum(np.abs(power[demi_idx:] - P_CIBLE))

                if erreur_absolue < best_score:
                    best_score = erreur_absolue
                    best_rn = rn_val
                    best_times = times
                    best_power = power
            except Exception as e:
                logger.error(
                    f"Impossible de lire les données dans {f}: {e}", exc_info=True
                )
                print(f"Impossible de lire {f}: {e}")

        if not all_data:
            logger.error("Échec de la lecture de tous les fichiers CSV.")
            messagebox.showwarning(
                "Erreur de lecture", "Les fichiers CSV n'ont pas pu être lus."
            )
            return

        logger.info(f"Analyse terminée avec succès. Meilleur rayon trouvé : {best_rn}")

        # 2. Affichage Graphique
        plt.close("Analyse de Stabilité")
        fig, ax = plt.subplots(num="Analyse de Stabilité", figsize=(10, 6))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.PANEL_BG)

        ax.axhline(
            P_CIBLE,
            color="white",
            linestyle="--",
            linewidth=2,
            label="Cible (50 GW)",
            zorder=3,
        )

        for rn_val, times, power in all_data:
            if rn_val == best_rn:
                continue
            ax.plot(times, power, color=self.FG_COLOR, alpha=0.3, linewidth=1)

        # Tracé du champion
        ax.plot(
            best_times,
            best_power,
            color="#00ff00",
            linewidth=3,
            label=f"Meilleur (Rn={best_rn} cm)",
            zorder=4,
        )

        ax.set_title(
            "Stabilité du Pilotage Automatique selon le Rayon (Rn)",
            color=self.FG_COLOR,
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Temps (s)", color=self.FG_COLOR)
        ax.set_ylabel("Puissance Totale", color=self.FG_COLOR)
        ax.tick_params(colors=self.FG_COLOR)

        ax.grid(True, color=self.BORDER_COLOR, linestyle=":")
        legend = ax.legend(
            facecolor=self.ENTRY_BG,
            edgecolor=self.BORDER_COLOR,
            labelcolor=self.FG_COLOR,
        )

        plt.tight_layout()
        plt.show()

        messagebox.showinfo(
            "Résultat de l'analyse",
            f"Le rayon offrant la meilleure stabilité pour ces matériaux est :\n\nRn = {best_rn} cm\n\n(Ce rayon minimise les oscillations autour de la cible).",
        )

    def _update_live_plot(self, ax, canvas, p, geom, centers, tags, top_window):
        try:
            if not top_window.winfo_exists() or not getattr(
                top_window, "is_running", False
            ):
                return
        except tk.TclError:
            return

        ax.clear()
        ax.axis("off")
        max_limit = p["R_reflec"] * 1.1
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.set_title(
            f"Rn = {p['R_noyau']:.1f} cm | {len(centers)} Assemblages",
            fontsize=10,
            color=self.FG_COLOR,
        )

        try:
            reflector = geom.get_reflector_patch(R_reflector=p["R_reflec"])
            reflector.set_facecolor(self.PANEL_BG)
            reflector.set_edgecolor(self.BORDER_COLOR)
            ax.add_patch(reflector)
        except Exception:
            pass

        if len(centers) > 0:
            mask_fuel, mask_cr = tags == "FUEL", tags == "CR"
            angles = np.pi / 6 + np.arange(6) * (np.pi / 3)
            base_hex = np.column_stack((np.cos(angles), np.sin(angles))) * p["R_hex"]

            if np.any(mask_fuel):
                fuel_verts = centers[mask_fuel][:, None, :] + base_hex[None, :, :]
                ax.add_collection(
                    PolyCollection(
                        fuel_verts,
                        facecolors="#264f78",
                        edgecolors=self.BG_COLOR,
                        linewidths=1.5,
                        alpha=0.8,
                    )
                )

            if np.any(mask_cr):
                cr_verts = centers[mask_cr][:, None, :] + base_hex[None, :, :]
                ax.add_collection(
                    PolyCollection(
                        cr_verts,
                        facecolors="#842029",
                        edgecolors=self.BG_COLOR,
                        linewidths=1.5,
                        alpha=0.8,
                    )
                )

        canvas.draw_idle()

    def _update_ui_progress(
        self, progress_var, lbl_status, current, total, msg, top_window
    ):
        try:
            if not top_window.winfo_exists() or not getattr(
                top_window, "is_running", False
            ):
                return
            percent = (current / total) * 100
            progress_var.set(percent)
            lbl_status.config(text=f"Étape {current}/{total} : {msg}")
        except tk.TclError:
            pass

    def _finish_parametric(self, top_window, generated_count, output_dir):
        try:
            if not top_window.winfo_exists() or not getattr(
                top_window, "is_running", False
            ):
                return
            top_window.is_running = False
            messagebox.showinfo(
                "Batch Terminé",
                f"Génération terminée !\n\n{generated_count} maillages uniques générés.\nDossier : {output_dir}",
                parent=top_window,
            )
            top_window.destroy()
        except tk.TclError:
            pass

    def action_run_case_A(self):
        """Lance l'étude Cas A par force brute sur des configurations discrètes."""
        p_ui = self.get_current_params()
        if not p_ui: return

        # Fenêtre de dialogue pour configurer la force brute
        diag = tk.Toplevel(self)
        diag.title("Cas A : Force Brute Configurations")
        diag.geometry("350x220")
        diag.configure(bg=self.BG_COLOR)
        diag.transient(self)

        tk.Label(diag, text=f"Rayon actuel ciblé : Rn = {p_ui['R_noyau']} cm", 
                 bg=self.BG_COLOR, fg="#00d2ff", font=("Segoe UI", 10, "bold")).pack(pady=10)

        tk.Label(diag, text="Nombre MAX de couronnes de contrôle à tester :", 
                 bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=5)
        
        var_max_rings = tk.IntVar(value=3)
        ttk.Spinbox(diag, from_=1, to=10, textvariable=var_max_rings, width=10).pack()

        tk.Label(diag, text="(L'algorithme testera 50% et 100% pour chaque couronne)", 
                 bg=self.BG_COLOR, fg="#888888", font=("Segoe UI", 8, "italic")).pack(pady=5)

        def launch():
            p_ui.update({
                "max_rings": var_max_rings.get(),
                "mat_mod": self.var_mat_mod.get(),
                "mat_ref": self.var_mat_ref.get(),
                "mat_cr": self.var_mat_cr.get()
            })
            diag.destroy()
            self._execute_case_A_worker(p_ui)

        ttk.Button(diag, text="🚀 Lancer la Force Brute", command=launch, style="Accent.TButton").pack(pady=15)

    def _execute_case_A_worker(self, params):
        top = tk.Toplevel(self)
        top.title("Calcul en cours...")
        top.geometry("400x120")
        top.configure(bg=self.BG_COLOR)
        top.attributes("-topmost", True)
        
        # SÉCURITÉ 1 : On désactive la croix rouge de la fenêtre de chargement.
        # Si l'utilisateur la ferme pendant que Gmsh tourne en fond, ça fait crasher Tkinter.
        top.protocol("WM_DELETE_WINDOW", lambda: None)

        lbl = tk.Label(top, text="Maillage et Analyse...", bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl.pack(pady=15)
        
        # SÉCURITÉ 2 : On stocke la variable directement dans l'objet 'top'.
        # Cela empêche le Garbage Collector du thread de détruire la variable.
        top.pv = tk.DoubleVar()
        pb = ttk.Progressbar(top, variable=top.pv, maximum=100)
        pb.pack(fill="x", padx=20)

        # Import du script
        from cases.case1_statique import run_discrete_brute_force

        # SÉCURITÉ 3 : Fonction encapsulée proprement pour le main thread
        def update_ui(c, t, m):
            try:
                if top.winfo_exists():
                    top.pv.set((c/t)*100)
                    lbl.config(text=m)
            except tk.TclError:
                pass

        # Le callback ne stocke rien localement, il transfère juste au main thread
        def cb(c, t, m):
            self.after(0, update_ui, c, t, m)

        def worker():
            try:
                # Lancement du calcul lourd
                labels, results = run_discrete_brute_force(params, cb)
                
                # Une fois terminé, on délègue l'affichage graphique au main thread
                self.after(0, self._plot_case_A_discrete, labels, results, params["R_noyau"], top)
                
            except Exception as e:
                logger.error(f"Erreur thread: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Erreur de Calcul", str(e)) if top.winfo_exists() else None)
                self.after(0, top.destroy)

        # Lancement du thread en tâche de fond
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def _plot_case_A_discrete(self, labels, results, r_noyau, top_window):
        try:
            if top_window.winfo_exists():
                top_window.destroy()
        except tk.TclError:
            pass

        plt.close("Cas A - Pilotabilité (Discret)")
        fig, ax = plt.subplots(num="Cas A - Pilotabilité (Discret)", figsize=(10, 6))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.PANEL_BG)

        # Tracé des points catégoriels
        x_positions = np.arange(len(labels))
        
        ax.plot(x_positions, results["Fuel_Uranium"], marker='o', markersize=8, linestyle='-', linewidth=2, color="#00d2ff", label="UOX")
        ax.plot(x_positions, results["Fuel_MOX"], marker='s', markersize=8, linestyle='-', linewidth=2, color="#ff4500", label="MOX")

        # Zone optimale
        ax.axhspan(0.3, 0.7, color='#125e2a', alpha=0.3, label='Plage de Pilotage Optimale [0.3 - 0.7]')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=1)
        ax.axhline(0.0, color='grey', linestyle='--', linewidth=1)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, color=self.FG_COLOR)
        
        ax.set_xlabel("Configuration (Anneaux & Densité)", color=self.FG_COLOR, fontweight="bold")
        ax.set_ylabel("Position d'équilibre (Z_eq)", color=self.FG_COLOR, fontweight="bold")
        ax.set_title(f"Cas A : Configurations Constructibles pour Rn = {r_noyau} cm", color=self.FG_COLOR, fontsize=12, fontweight="bold")
        
        ax.tick_params(colors=self.FG_COLOR)
        ax.grid(True, color=self.BORDER_COLOR, linestyle=":", axis='y')
        ax.legend(facecolor=self.ENTRY_BG, edgecolor=self.BORDER_COLOR, labelcolor=self.FG_COLOR)

        plt.tight_layout()
        plt.show()
    
    def action_run_case_3(self):
        """Lance l'étude Cas 3 d'optimisation du réflecteur via une boîte de dialogue paramétrable."""
        p_ui = self.get_current_params()
        if not p_ui: return

        diag = tk.Toplevel(self)
        diag.title("Cas 3 : Optimisation Réflecteur")
        # On agrandit la fenêtre pour accueillir les nouveaux paramètres
        diag.geometry("400x380")
        diag.configure(bg=self.BG_COLOR)
        diag.transient(self)

        # --- 1. Taille du cœur ---
        tk.Label(diag, text="Rayon du cœur (Rn) en cm :", 
                 bg=self.BG_COLOR, fg="#00d2ff", font=("Segoe UI", 9, "bold")).pack(pady=(10, 0))
        var_rn = tk.DoubleVar(value=p_ui.get("R_noyau", 12.0))
        ttk.Spinbox(diag, from_=5.0, to=50.0, increment=2.0, textvariable=var_rn, width=12).pack()

        # --- 2. Barres de contrôle (Anti-explosion) ---
        tk.Label(diag, text="Couronnes de contrôle (CR Rings) :", 
                 bg=self.BG_COLOR, fg="#ff4500", font=("Segoe UI", 9, "bold")).pack(pady=(10, 0))
        # On propose 2 couronnes par défaut pour plus de sécurité
        var_cr_rings = tk.IntVar(value=max(2, p_ui.get("cr_rings", 2))) 
        ttk.Spinbox(diag, from_=1, to=10, increment=1, textvariable=var_cr_rings, width=12).pack()

        tk.Label(diag, text="Densité des barres (0.0 à 1.0) :", 
                 bg=self.BG_COLOR, fg="#ff4500", font=("Segoe UI", 9, "bold")).pack(pady=(5, 0))
        # On met 1.0 (100%) par défaut pour maximiser l'absorption
        var_cr_density = tk.DoubleVar(value=1.0) 
        ttk.Spinbox(diag, from_=0.1, to=1.0, increment=0.1, textvariable=var_cr_density, width=12).pack()

        # --- 3. Paramètres du Réflecteur ---
        tk.Label(diag, text="Épaisseur MAX du réflecteur (cm) :", 
                 bg=self.BG_COLOR, fg="#2ca02c", font=("Segoe UI", 9, "bold")).pack(pady=(10, 0))
        var_max_ep = tk.DoubleVar(value=16.0)
        ttk.Spinbox(diag, from_=4.0, to=40.0, increment=4.0, textvariable=var_max_ep, width=12).pack()

        tk.Label(diag, text="(Générera 5 points de test progressifs)", 
                 bg=self.BG_COLOR, fg="#888888", font=("Segoe UI", 8, "italic")).pack(pady=2)

        def launch():
            # On met à jour p_ui avec les valeurs saisies dans la pop-up
            p_ui.update({
                "R_noyau": var_rn.get(),
                "cr_rings": var_cr_rings.get(),
                "cr_density": var_cr_density.get(),
                "max_epaisseur": var_max_ep.get(),
                "mat_mod": self.var_mat_mod.get(),
                "mat_ref": self.var_mat_ref.get(),
                "mat_cr": self.var_mat_cr.get(),
                "mat_fuel": self.var_mat_fuel.get()
            })
            diag.destroy()
            self._execute_case_3_worker(p_ui)

        ttk.Button(diag, text="🚀 Lancer l'Optimisation", command=launch, style="Success.TButton").pack(pady=15)

    def _execute_case_3_worker(self, params):
        top = tk.Toplevel(self)
        top.title("Calcul en cours...")
        top.geometry("400x120")
        top.configure(bg=self.BG_COLOR)
        top.attributes("-topmost", True)
        
        # Sécurité : empêcher la fermeture manuelle de la fenêtre
        top.protocol("WM_DELETE_WINDOW", lambda: None)

        lbl = tk.Label(top, text="Génération et calculs de diffusion...", bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl.pack(pady=15)
        
        top.pv = tk.DoubleVar()
        pb = ttk.Progressbar(top, variable=top.pv, maximum=100)
        pb.pack(fill="x", padx=20)

        # Import du script refactorisé
        from cases.case3_optimisationReflecteur import run_reflector_optimization

        def update_ui(c, t, m):
            try:
                if top.winfo_exists():
                    top.pv.set((c/t)*100)
                    lbl.config(text=m)
            except tk.TclError:
                pass

        def cb(c, t, m):
            self.after(0, update_ui, c, t, m)

        def worker():
            try:
                # Exécution du Cas 3 en tâche de fond
                epaisseurs, fq_values = run_reflector_optimization(params, cb)
                
                # Délégation de l'affichage au main thread Tkinter
                self.after(0, self._plot_case_3_results, epaisseurs, fq_values, params["R_noyau"], top)
                
            except Exception as e:
                logger.error(f"Erreur thread (Cas 3): {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Erreur de Calcul", str(e)) if top.winfo_exists() else None)
                self.after(0, top.destroy)

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def _plot_case_3_results(self, epaisseurs, fq_values, r_noyau, top_window):
        try:
            if top_window.winfo_exists():
                top_window.destroy()
        except tk.TclError:
            pass

        # Recherche de l'optimum (Fq le plus petit)
        valid_indices = [i for i, fq in enumerate(fq_values) if fq != float('inf')]
        if not valid_indices:
            messagebox.showerror("Échec", "Aucun calcul n'a convergé.")
            return

        best_idx = min(valid_indices, key=lambda i: fq_values[i])
        best_ep = epaisseurs[best_idx]
        best_fq = fq_values[best_idx]

        plt.close("Cas 3 - Optimisation Réflecteur")
        fig, ax = plt.subplots(num="Cas 3 - Optimisation Réflecteur", figsize=(9, 6))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.PANEL_BG)

        ax.plot(epaisseurs, fq_values, marker='s', markersize=8, linestyle='-', linewidth=2, color="#2ca02c", label="Facteur de Forme ($F_q$)")

        # Mise en évidence de l'optimum
        ax.axvline(best_ep, color='#ffcc00', linestyle='--', linewidth=2, 
                   label=f"Optimum calculé (~{best_ep:.1f} cm)")
        ax.plot(best_ep, best_fq, marker='*', markersize=15, color='#ffcc00')

        ax.set_xlabel("Épaisseur du réflecteur (cm)", color=self.FG_COLOR, fontweight="bold")
        ax.set_ylabel("Facteur de Forme $F_q$ (Max/Moyen)", color=self.FG_COLOR, fontweight="bold")
        ax.set_title(f"Cas 3 : Aplatissement du Flux (Rn = {r_noyau} cm)", color=self.FG_COLOR, fontsize=12, fontweight="bold")
        
        ax.tick_params(colors=self.FG_COLOR)
        ax.grid(True, color=self.BORDER_COLOR, linestyle=":")
        ax.legend(facecolor=self.ENTRY_BG, edgecolor=self.BORDER_COLOR, labelcolor=self.FG_COLOR)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    app = ReactorGUI()
    app.mainloop()
