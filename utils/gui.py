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
    from physics.materials import MATERIAL_DB


class ReactorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        logger.info("Démarrage de l'interface graphique ReactorGUI.")
        self.title("Générateur de Géométrie - Cœur de Réacteur")
        self.geometry("750x900")
        self.minsize(650, 850)
        self.resizable(True, True)

        self.bind(
            "<F11>",
            lambda event: self.attributes(
                "-fullscreen", not self.attributes("-fullscreen")
            ),
        )
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

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

        style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.ENTRY_BG)],
            foreground=[("readonly", self.FG_COLOR)],
            selectbackground=[("readonly", self.ACCENT_BLUE)],
            selectforeground=[("readonly", "white")],
        )

        self.option_add("*TCombobox*Listbox.background", self.ENTRY_BG)
        self.option_add("*TCombobox*Listbox.foreground", self.FG_COLOR)
        self.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT_BLUE)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")
        self.option_add("*TCombobox*Listbox.font", default_font)

        self.var_R_hex = tk.DoubleVar(value=2.0)
        self.var_R_noyau = tk.DoubleVar(value=12.0)
        self.var_epaisseur_reflec = tk.DoubleVar(value=4.0)

        self.var_pins_fuel = tk.IntVar(value=4)
        self.var_pins_cr = tk.IntVar(value=3)

        self.var_cr_rings = tk.IntVar(value=1)
        self.var_cr_density = tk.DoubleVar(value=0.5)

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
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        initial_dir = os.path.join(base_dir, "output", "meshes")
        os.makedirs(initial_dir, exist_ok=True)

        output_file = filedialog.askopenfilename(
            title="Sélectionner un maillage à simuler",
            initialdir=initial_dir,
            filetypes=[("Fichiers Gmsh", "*.msh"), ("Tous les fichiers", "*.*")],
        )

        if not output_file:
            return

        logger.info(
            f"Fichier de maillage sélectionné pour la simulation : {output_file}"
        )

        user_materials = {
            "Fuel": self.var_mat_fuel.get() or "Fuel_Uranium",
            "Moderator": self.var_mat_mod.get() or "Water_Moderator",
            "Reflector": self.var_mat_ref.get() or "Graphite_Moderator",
            "ControlRods": self.var_mat_cr.get() or "Boral_ControlRod",
        }

        try:
            run_full_simulation(output_file, user_mapping=user_materials)
        except Exception as e:
            logger.error(
                f"Le solver a échoué lors de l'exécution unitaire : {e}", exc_info=True
            )
            messagebox.showerror("Erreur de calcul", f"Le solver a échoué :\n{e}")

    def action_view_mesh(self):
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
            text="🔬 Cas 1 : Étude de Pilotabilité (Statique)",
            width=btn_width,
            command=self.action_run_case_1,
            style="Accent.TButton",
        ).pack(pady=(5, 0))

        ttk.Button(
            btn_container,
            text="📉 Cas 2 : Cartographie Overshoot",
            width=btn_width,
            command=self.action_run_case_2,
            style="Accent.TButton",
        ).pack(pady=(5, 0))
        ttk.Button(
            btn_container,
            text="✅ Cas 3 : Validation Théorique (V&V)",
            width=btn_width,
            command=self.action_run_case_3,
            style="Accent.TButton",
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

        logger.info(f"Début de la generation automatique (Gmsh) : {output_file}")
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

            logger.info(f"Début de la generation manuelle (Gmsh) : {output_file}")
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

    def action_run_case_1(self):
        p_ui = self.get_current_params()
        if not p_ui:
            return

        diag = tk.Toplevel(self)
        diag.title("Cas 1 : Force Brute Configurations")
        diag.geometry("350x220")
        diag.configure(bg=self.BG_COLOR)
        diag.transient(self)

        tk.Label(
            diag,
            text=f"Rayon actuel ciblé : Rn = {p_ui['R_noyau']} cm",
            bg=self.BG_COLOR,
            fg="#00d2ff",
            font=("Segoe UI", 10, "bold"),
        ).pack(pady=10)

        tk.Label(
            diag,
            text="Nombre MAX de couronnes de contrôle à tester :",
            bg=self.BG_COLOR,
            fg=self.FG_COLOR,
        ).pack(pady=5)

        var_max_rings = tk.IntVar(value=3)
        ttk.Spinbox(diag, from_=1, to=10, textvariable=var_max_rings, width=10).pack()

        tk.Label(
            diag,
            text="(L'algorithme testera 50% et 100% pour chaque couronne)",
            bg=self.BG_COLOR,
            fg="#888888",
            font=("Segoe UI", 8, "italic"),
        ).pack(pady=5)

        def launch():
            p_ui.update(
                {
                    "max_rings": var_max_rings.get(),
                    "mat_mod": self.var_mat_mod.get(),
                    "mat_ref": self.var_mat_ref.get(),
                    "mat_cr": self.var_mat_cr.get(),
                }
            )
            diag.destroy()
            self._execute_case_1_worker(p_ui)

        ttk.Button(
            diag,
            text="🚀 Lancer la Force Brute",
            command=launch,
            style="Accent.TButton",
        ).pack(pady=15)

    def _execute_case_1_worker(self, params):
        top = tk.Toplevel(self)
        top.title("Calcul en cours...")
        top.geometry("400x120")
        top.configure(bg=self.BG_COLOR)
        top.attributes("-topmost", True)
        top.protocol("WM_DELETE_WINDOW", lambda: None)

        lbl = tk.Label(
            top, text="Maillage et Analyse...", bg=self.BG_COLOR, fg=self.FG_COLOR
        )
        lbl.pack(pady=15)

        top.pv = tk.DoubleVar()
        pb = ttk.Progressbar(top, variable=top.pv, maximum=100)
        pb.pack(fill="x", padx=20)

        from cases.case1_statique import run_discrete_brute_force

        def update_ui(c, t, m):
            try:
                if top.winfo_exists():
                    top.pv.set((c / t) * 100)
                    lbl.config(text=m)
            except tk.TclError:
                pass

        def cb(c, t, m):
            self.after(0, update_ui, c, t, m)

        def worker():
            try:
                labels, results = run_discrete_brute_force(params, cb)

                self.after(
                    0,
                    self._plot_case_1_discrete,
                    labels,
                    results,
                    params["R_noyau"],
                    top,
                )

            except Exception as e:
                logger.error(f"Erreur thread: {e}", exc_info=True)
                self.after(
                    0,
                    lambda: messagebox.showerror("Erreur de Calcul", str(e))
                    if top.winfo_exists()
                    else None,
                )
                self.after(0, top.destroy)

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()

    def _plot_case_1_discrete(self, labels, results, r_noyau, top_window):
        try:
            if top_window.winfo_exists():
                top_window.destroy()
        except tk.TclError:
            pass

        plt.close("Cas 1 - Pilotabilité (Discret)")
        fig, ax = plt.subplots(num="Cas 1 - Pilotabilité (Discret)", figsize=(10, 6))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.PANEL_BG)

        x_positions = np.arange(len(labels))

        ax.plot(
            x_positions,
            results["Fuel_Uranium"],
            marker="o",
            markersize=8,
            linestyle="-",
            linewidth=2,
            color="#00d2ff",
            label="UOX",
        )
        ax.plot(
            x_positions,
            results["Fuel_MOX"],
            marker="s",
            markersize=8,
            linestyle="-",
            linewidth=2,
            color="#ff4500",
            label="MOX",
        )

        ax.axhspan(
            0.3,
            0.7,
            color="#125e2a",
            alpha=0.3,
            label="Plage de Pilotage Optimale [0.3 - 0.7]",
        )
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, color=self.FG_COLOR)

        ax.set_xlabel(
            "Configuration (Anneaux & Densité)", color=self.FG_COLOR, fontweight="bold"
        )
        ax.set_ylabel(
            "Position d'équilibre (Z_eq)", color=self.FG_COLOR, fontweight="bold"
        )
        ax.set_title(
            f"Cas 1 : Configurations Constructibles pour Rn = {r_noyau} cm",
            color=self.FG_COLOR,
            fontsize=12,
            fontweight="bold",
        )

        ax.tick_params(colors=self.FG_COLOR)
        ax.grid(True, color=self.BORDER_COLOR, linestyle=":", axis="y")
        ax.legend(
            facecolor=self.ENTRY_BG,
            edgecolor=self.BORDER_COLOR,
            labelcolor=self.FG_COLOR,
        )

        plt.tight_layout()
        plt.show()

    def action_run_case_2(self):
        """Ouvre le dialogue de configuration pour la Heatmap d'Overshoot."""
        p_ui = self.get_current_params()
        if not p_ui: return

        diag = tk.Toplevel(self)
        diag.title("Cas 2 : Configuration Heatmap")
        diag.geometry("450x350")
        diag.configure(bg=self.BG_COLOR)
        diag.transient(self)
        diag.grab_set()

        tk.Label(diag, text="Optimisation Spatiale de l'Overshoot", 
                 bg=self.BG_COLOR, fg="#00d2ff", font=("Segoe UI", 12, "bold")).pack(pady=15)

        confirm_frame = tk.LabelFrame(diag, text="Matériaux", bg=self.PANEL_BG, fg=self.FG_COLOR, padx=10, pady=10)
        confirm_frame.pack(fill="x", padx=30, pady=10)
        
        tk.Label(confirm_frame, text=f"• Combustible : {self.var_mat_fuel.get()}", bg=self.PANEL_BG, fg=self.FG_COLOR).pack(anchor="w")
        tk.Label(confirm_frame, text=f"• Modérateur : {self.var_mat_mod.get()}", bg=self.PANEL_BG, fg=self.FG_COLOR).pack(anchor="w")

        input_frame = tk.Frame(diag, bg=self.BG_COLOR)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Épaisseur max réflecteur (cm) :", bg=self.BG_COLOR, fg=self.FG_COLOR).grid(row=0, column=0, padx=5, pady=5)
        var_max_thick = tk.DoubleVar(value=12.0)
        ttk.Entry(input_frame, textvariable=var_max_thick, width=10).grid(row=0, column=1, pady=5)

        def launch():
            params = p_ui.copy()
            params.update({
                "max_thickness": var_max_thick.get(),
                "mat_fuel": self.var_mat_fuel.get(),
                "mat_mod": self.var_mat_mod.get(),
                "mat_ref": self.var_mat_ref.get(),
                "mat_cr": self.var_mat_cr.get()
            })
            diag.destroy()
            self._execute_case_2_worker(params)

        ttk.Button(diag, text="🚀 Générer la Heatmap", command=launch, style="Accent.TButton").pack(pady=10)
    def _execute_case_2_worker(self, params):
        """Gère l'exécution asynchrone et le déballage des 3 variables de la Heatmap."""
        top = tk.Toplevel(self)
        top.title("Calcul de la Heatmap...")
        top.geometry("450x150")
        top.configure(bg=self.BG_COLOR)
        top.attributes("-topmost", True)
        top.protocol("WM_DELETE_WINDOW", lambda: None)

        lbl = tk.Label(top, text="Analyse matricielle en cours...", bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl.pack(pady=20)
        
        progress_var = tk.DoubleVar()
        pb = ttk.Progressbar(top, variable=progress_var, maximum=100)
        pb.pack(fill="x", padx=40)

        from cases.case2_overshootopt import run_overshoot_analysis

        def update_ui(current, total, msg):
            try:
                if top.winfo_exists():
                    progress_var.set((current / total) * 100)
                    lbl.config(text=msg)
            except tk.TclError: pass

        def worker():
            try:
                # Déballage des 3 variables attendues
                thicknesses, target_rings, matrix = run_overshoot_analysis(
                    params, 
                    progress_callback=lambda c, t, m: self.after(0, update_ui, c, t, m)
                )
                
                self.after(0, lambda: (top.destroy(), self.plot_overshoot_heatmap(thicknesses, target_rings, matrix)))
                
            except Exception as e:
                logger.error(f"Erreur Cas 2 : {e}", exc_info=True)
                err_txt = str(e)
                self.after(0, lambda m=err_txt: (messagebox.showerror("Erreur", m), top.destroy()))

        threading.Thread(target=worker, daemon=True).start()

    def plot_overshoot_heatmap(self, thicknesses, target_rings, overshoot_matrix):
        """Affiche la carte thermique 2D de l'overshoot."""
        plt.close("Cas 2 - Heatmap Overshoot")
        fig, ax = plt.subplots(num="Cas 2 - Heatmap Overshoot", figsize=(10, 7))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.PANEL_BG)

        # Création de la grille pour le tracé
        X, Y = np.meshgrid(target_rings, thicknesses)
        
        # Tracé des contours remplis (Heatmap)
        cp = ax.contourf(X, Y, overshoot_matrix, levels=20, cmap='magma')
        
        # Barre d'échelle
        cbar = fig.colorbar(cp, ax=ax)
        cbar.set_label('Overshoot (%)', color=self.FG_COLOR, fontweight='bold')
        cbar.ax.yaxis.set_tick_params(color=self.FG_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=self.FG_COLOR)

        # Recherche et marquage de l'optimum (valeur minimale)
        if not np.all(np.isnan(overshoot_matrix)):
            idx = np.unravel_index(np.nanargmin(overshoot_matrix), overshoot_matrix.shape)
            opt_t, opt_r = thicknesses[idx[0]], target_rings[idx[1]]
            
            ax.scatter(opt_r, opt_t, color='#00ff00', marker='*', s=250, 
                       edgecolor='white', label='Optimum Physique', zorder=5)
            
            ax.annotate(f"Idéal: {overshoot_matrix[idx]:.1f}%", 
                        (opt_r, opt_t), xytext=(15, 15), 
                        textcoords='offset points', color='#00ff00', fontweight='bold',
                        arrowprops=dict(arrowstyle="->", color="#00ff00"))

        # Cosmétique scientifique
        ax.set_xlabel("Position radiale de l'anneau (Indice)", color=self.FG_COLOR, fontweight="bold")
        ax.set_ylabel("Épaisseur du réflecteur (cm)", color=self.FG_COLOR, fontweight="bold")
        ax.set_title("Optimisation Spatiale du Pilotage (Overshoot)", 
                     color=self.FG_COLOR, fontsize=12, fontweight="bold", pad=20)
        
        ax.set_xticks(target_rings)
        ax.tick_params(colors=self.FG_COLOR)
        ax.grid(True, linestyle=':', alpha=0.3)
        ax.legend(facecolor=self.PANEL_BG, edgecolor=self.BORDER_COLOR, labelcolor=self.FG_COLOR)
        
        plt.tight_layout()
        plt.show()

    def action_run_case_3(self):
        """Lance l'étude de validation théorique (V&V)."""
        p_ui = self.get_current_params()
        if not p_ui: return

        top = tk.Toplevel(self)
        top.title("Analyse V&V en cours")
        top.geometry("400x150")
        top.configure(bg=self.BG_COLOR)
        
        lbl = tk.Label(top, text="Calcul des estimateurs...", bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl.pack(pady=20)
        
        progress = ttk.Progressbar(top, length=300, mode='determinate')
        progress.pack(pady=10)

        from cases.case3_validation import run_validation_test

        def worker():
            try:
                # Récupération des résultats synchronisés
                res = run_validation_test(
                    p_ui, 
                    progress_callback=lambda c, t, m: self.after(0, lambda: (progress.configure(value=c), lbl.configure(text=m)))
                )
                self.after(0, lambda: (top.destroy(), self._plot_case_3(*res)))
            except Exception as e:
                logger.error(f"Erreur Cas 3: {e}", exc_info=True)
                self.after(0, lambda: messagebox.showerror("Erreur", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _plot_case_3(self, times, num_max, theo_bound, num_min, pop_totale):
        """Affiche les graphiques de validation."""
        plt.close("Cas 3 - Validation Théorique")
        fig, axes = plt.subplots(3, 1, figsize=(10, 11), num="Cas 3 - Validation Théorique")
        fig.patch.set_facecolor(self.BG_COLOR)
        
        # 1. Population (Conservation)
        axes[0].set_facecolor(self.PANEL_BG)
        axes[0].plot(times, pop_totale, color="#00d2ff", lw=2, label=r"$\Phi(t) = \int_{\Omega} \phi d\Omega$")
        axes[0].set_title("1. Évolution de la Population Neutronique Totale", color="white", fontweight="bold")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend()

        # 2. Borne Supérieure (Growth Estimate) - Échelle LOG
        axes[1].set_facecolor(self.PANEL_BG)
        axes[1].plot(times, num_max, color="blue", lw=2, label=r"Max($\phi$) numérique")
        axes[1].plot(times, theo_bound, color="red", ls="--", lw=2, label=r"Borne $e^{vCt} \|\phi_0\|_{\infty}$")
        axes[1].set_yscale("log")
        axes[1].set_title("2. Vérification de la Borne Supérieure", color="white", fontweight="bold")
        axes[1].grid(True, which="both", alpha=0.1)
        axes[1].legend()

        # 3. Positivité (Weak Maximum Principle)
        axes[2].set_facecolor(self.PANEL_BG)
        axes[2].plot(times, num_min, color="green", lw=2, label=r"Min($\phi$) numérique")
        axes[2].axhline(0, color="red", ls=":", lw=2, label="Zéro physique")
        axes[2].set_title("3. Préservation de la Positivité", color="white", fontweight="bold")
        axes[2].set_xlabel("Temps (s)", color="white")
        axes[2].grid(True, alpha=0.2)
        axes[2].legend()

        for ax in axes:
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")

        plt.tight_layout(pad=3.0)
        plt.show()

if __name__ == "__main__":
    app = ReactorGUI()
    app.mainloop()
