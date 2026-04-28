import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection, PatchCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure  # <-- Import crucial pour éviter le bug des fenêtres multiples
import numpy as np
import sys
import os
import threading

from core.solver import run_full_simulation

# Configuration du thème sombre strict pour Matplotlib
plt.style.use('dark_background')

# Ajout du dossier racine au PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from physics.geometry import ReactorGeometry, ReactorMeshGenerator


class ReactorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Générateur de Géométrie - Cœur de Réacteur")
        self.geometry("550x750")
        self.minsize(500, 700)
        self.resizable(True, True)

        # Raccourcis Plein Écran
        self.bind("<F11>", lambda event: self.attributes("-fullscreen", not self.attributes("-fullscreen")))
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

        style.configure(".", font=default_font, background=self.BG_COLOR, foreground=self.FG_COLOR)
        style.configure("TFrame", background=self.BG_COLOR)
        style.configure("TEntry", fieldbackground=self.ENTRY_BG, foreground=self.FG_COLOR, bordercolor=self.BORDER_COLOR, lightcolor=self.ENTRY_BG, darkcolor=self.ENTRY_BG)
        style.configure("TSpinbox", fieldbackground=self.ENTRY_BG, foreground=self.FG_COLOR, bordercolor=self.BORDER_COLOR, arrowcolor=self.FG_COLOR, lightcolor=self.ENTRY_BG, darkcolor=self.ENTRY_BG)
        style.configure("TLabelframe", background=self.PANEL_BG, bordercolor=self.BORDER_COLOR, lightcolor=self.PANEL_BG, darkcolor=self.BORDER_COLOR)
        style.configure("TLabelframe.Label", font=bold_font, foreground=self.TITLE_COLOR, background=self.PANEL_BG)
        style.configure("TSeparator", background=self.BORDER_COLOR)
        style.configure("TButton", font=default_font, padding=6, background="#4d4d4d", foreground=self.FG_COLOR, bordercolor=self.BORDER_COLOR, lightcolor="#4d4d4d", darkcolor="#4d4d4d")
        style.map("TButton", background=[("active", "#5a5a5a")])

        style.configure("Accent.TButton", font=bold_font, background=self.ACCENT_BLUE, foreground="white", bordercolor=self.BORDER_COLOR)
        style.map("Accent.TButton", background=[("active", self.ACCENT_HOVER)])
        
        style.configure("Success.TButton", font=bold_font, background="#125e2a", foreground="white", bordercolor=self.BORDER_COLOR)
        style.map("Success.TButton", background=[("active", "#187a37")])

        style.configure("Warning.TButton", font=bold_font, background="#9e5a0e", foreground="white", bordercolor=self.BORDER_COLOR)
        style.map("Warning.TButton", background=[("active", "#bf6d11")])

        # --- Variables Tkinter ---
        self.var_R_hex = tk.DoubleVar(value=2.0)
        self.var_R_noyau = tk.DoubleVar(value=12.0)
        self.var_R_reflec = tk.DoubleVar(value=16.0)

        self.var_pins_fuel = tk.IntVar(value=4)
        self.var_pins_cr = tk.IntVar(value=3)

        self.var_cr_rings = tk.IntVar(value=1)
        self.var_cr_density = tk.DoubleVar(value=0.5)

        self.create_widgets()

    def action_solve(self):
        p = self.get_current_params()
        # On définit le chemin du fichier
        output_file = "output/meshes/reactor_core.msh"
        
        # 1. On s'assure que le maillage est à jour
        self.action_gmsh_auto() 
        
        # 2. On lance le calcul
        try:
            run_full_simulation(output_file)
        except Exception as e:
            messagebox.showerror("Erreur de calcul", f"Le solver a échoué :\n{e}")

    def create_widgets(self):
        main_container = ttk.Frame(self, padding=(20, 20))
        main_container.pack(fill="both", expand=True)

        lbl_hint = tk.Label(main_container, text="F11 : Plein écran | Échap : Quitter plein écran", bg=self.BG_COLOR, fg="#666666", font=("Segoe UI", 8, "italic"))
        lbl_hint.pack(side="top", anchor="e", pady=(0, 10))

        def create_panel(parent, text):
            frame = ttk.Frame(parent, style="TFrame")
            frame.pack(fill="x", pady=(0, 15))
            lf = ttk.LabelFrame(frame, text=text, padding=(15, 10))
            lf.pack(fill="both", expand=True)
            return lf

        frame_dim = create_panel(main_container, "Dimensions Globales (cm)")
        ttk.Label(frame_dim, text="Rayon d'un hexagone :", background=self.PANEL_BG).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame_dim, textvariable=self.var_R_hex, width=12).grid(row=0, column=1, sticky="e", pady=5)

        ttk.Label(frame_dim, text="Rayon limite du cœur :", background=self.PANEL_BG).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame_dim, textvariable=self.var_R_noyau, width=12).grid(row=1, column=1, sticky="e", pady=5)

        ttk.Label(frame_dim, text="Rayon extérieur réflecteur :", background=self.PANEL_BG).grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame_dim, textvariable=self.var_R_reflec, width=12).grid(row=2, column=1, sticky="e", pady=5)
        frame_dim.grid_columnconfigure(0, weight=1)

        frame_pins = create_panel(main_container, "Structure Interne (Couronnes de crayons)")
        ttk.Label(frame_pins, text="Combustible (FUEL) :", background=self.PANEL_BG).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Spinbox(frame_pins, from_=1, to=10, textvariable=self.var_pins_fuel, width=10).grid(row=0, column=1, sticky="e", pady=5)

        ttk.Label(frame_pins, text="Barre de contrôle (CR) :", background=self.PANEL_BG).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Spinbox(frame_pins, from_=1, to=10, textvariable=self.var_pins_cr, width=10).grid(row=1, column=1, sticky="e", pady=5)
        frame_pins.grid_columnconfigure(0, weight=1)

        frame_cr = create_panel(main_container, "Répartition Automatique des CR")
        ttk.Label(frame_cr, text="Nombre d'anneaux CR :", background=self.PANEL_BG).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Spinbox(frame_cr, from_=0, to=5, textvariable=self.var_cr_rings, width=10).grid(row=0, column=1, sticky="e", pady=5)

        ttk.Label(frame_cr, text="Densité dans l'anneau (0.0 - 1.0) :", background=self.PANEL_BG).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame_cr, textvariable=self.var_cr_density, width=12).grid(row=1, column=1, sticky="e", pady=5)
        frame_cr.grid_columnconfigure(0, weight=1)

        frame_actions = ttk.Frame(main_container)
        frame_actions.pack(fill="both", side="bottom", pady=5, expand=True)

        btn_container = ttk.Frame(frame_actions)
        btn_container.pack(expand=True, anchor="s")

        btn_width = 45 

        ttk.Button(btn_container, text="Aperçu Visuel Rapide", width=btn_width, command=self.action_preview).pack(pady=(0, 5))
        ttk.Button(btn_container, text="Générer Maillage (Auto)", width=btn_width, command=self.action_gmsh_auto, style="Accent.TButton").pack(pady=(0, 5))
        ttk.Button(btn_container, text="Sélection Manuelle (Pinceau)", width=btn_width, command=self.action_manual_selection, style="Success.TButton").pack(pady=(0, 15))
        
        btn_solve = ttk.Button(btn_container, text="🚀 Lancer la Simulation", 
                       width=btn_width, command=self.action_solve, 
                       style="Accent.TButton")
        btn_solve.pack(pady=(5, 0))

        ttk.Separator(btn_container, orient='horizontal').pack(fill='x', pady=5)
        ttk.Button(btn_container, text="Étude Paramétrique (Batch)", width=btn_width, command=self.action_parametric_setup, style="Warning.TButton").pack(pady=(5, 0))

    def get_current_params(self):
        try:
            return {
                "R_hex": self.var_R_hex.get(),
                "R_noyau": self.var_R_noyau.get(),
                "R_reflec": self.var_R_reflec.get(),
                "pins_fuel": self.var_pins_fuel.get(),
                "pins_cr": self.var_pins_cr.get(),
                "cr_rings": self.var_cr_rings.get(),
                "cr_density": self.var_cr_density.get(),
            }
        except tk.TclError:
            messagebox.showerror("Erreur de saisie", "Veuillez entrer des valeurs numériques valides.")
            return None

    def _check_reflector_logic(self, p):
        if p["R_reflec"] <= p["R_noyau"]:
            messagebox.showwarning("Incohérence", "Le réflecteur doit être plus grand que le noyau.")
            return False
        return True

    # ---------------------------------------------------------
    # MÉTHODES STANDARD
    # ---------------------------------------------------------
    def action_preview(self):
        p = self.get_current_params()
        if not p or not self._check_reflector_logic(p): return

        geom = ReactorGeometry(R_n=p["R_noyau"], R_hex=p["R_hex"])
        centers, tags, _ = geom.get_tagged_assemblies(n_cr_rings=p["cr_rings"], cr_density=p["cr_density"])

        if len(centers) == 0:
            messagebox.showinfo("Vide", "Aucun assemblage ne rentre dans dimensions.")
            return

        self._plot_static_preview(p, geom, centers, tags)

    def _plot_static_preview(self, p, geom, centers, tags):
        # Sécurité : Ferme la figure précédente si elle existe pour éviter l'empilement
        plt.close("Aperçu Matplotlib - Cœur de Réacteur")
        
        mask_fuel, mask_cr = tags == "FUEL", tags == "CR"
        fuel_centers, cr_centers = centers[mask_fuel], centers[mask_cr]

        angles = np.pi / 6 + np.arange(6) * (np.pi / 3)
        base_hex = np.column_stack((np.cos(angles), np.sin(angles))) * p["R_hex"]
        
        fig, ax = plt.subplots(num="Aperçu Matplotlib - Cœur de Réacteur", figsize=(8, 8))
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

        ax.add_collection(PolyCollection(fuel_centers[:, None, :] + base_hex[None, :, :], facecolors="#264f78", edgecolors=self.BG_COLOR, linewidths=1.5, alpha=0.9, zorder=1))
        if len(cr_centers) > 0:
            ax.add_collection(PolyCollection(cr_centers[:, None, :] + base_hex[None, :, :], facecolors="#842029", edgecolors=self.BG_COLOR, linewidths=1.5, alpha=0.9, zorder=1))

        offsets_fuel, r_fuel = geom.get_local_pin_offsets(p["pins_fuel"])
        offsets_cr, r_cr = geom.get_local_pin_offsets(p["pins_cr"])
        
        if len(fuel_centers) > 0 and len(offsets_fuel) > 0:
            ax.add_collection(PatchCollection([mpatches.Circle(xy, r_fuel) for xy in (fuel_centers[:, None, :] + offsets_fuel[None, :, :]).reshape(-1, 2)], facecolor="#3a82c4", alpha=0.9, zorder=3))
        if len(cr_centers) > 0 and len(offsets_cr) > 0:
            ax.add_collection(PatchCollection([mpatches.Circle(xy, r_cr) for xy in (cr_centers[:, None, :] + offsets_cr[None, :, :]).reshape(-1, 2)], facecolor="#c43a46", alpha=0.9, zorder=3))

        ax.set_aspect("equal")
        limit = p["R_reflec"] * 1.1
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title(f"Aperçu : {len(centers)} assemblages", fontsize=12, fontweight="bold", color=self.FG_COLOR)
        
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def action_gmsh_auto(self):
        p = self.get_current_params()
        if not p or not self._check_reflector_logic(p): return
        geom = ReactorGeometry(R_n=p["R_noyau"], R_hex=p["R_hex"])
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(base_dir, "output", "meshes")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "reactor_core.msh")
        
        self._run_gmsh(geom, p, output_file, show_popup=True)

    # ---------------------------------------------------------
    # SÉLECTION MANUELLE 
    # ---------------------------------------------------------
    def action_manual_selection(self):
        p = self.get_current_params()
        if not p or not self._check_reflector_logic(p): return

        geom = ReactorGeometry(R_n=p["R_noyau"], R_hex=p["R_hex"])
        centers = geom.hex_centers()
        if len(centers) == 0:
            messagebox.showinfo("Vide", "Aucun assemblage à afficher.")
            return

        _, auto_tags, _ = geom.get_tagged_assemblies(n_cr_rings=p["cr_rings"], cr_density=p["cr_density"])
        if len(auto_tags) == len(centers):
            tags = auto_tags
        else:
            tags = np.full(len(centers), "FUEL", dtype=object)

        top = tk.Toplevel(self)
        top.title("Peinture Manuelle du Cœur")
        top.geometry("750x850")
        top.configure(bg=self.BG_COLOR)
        
        lbl_info = tk.Label(top, text="Cliquez et glissez (peinture) sur les hexagones pour basculer :\nBleu (Combustible) ↔ Rouge (Barre de Contrôle)", font=("Segoe UI", 10, "italic"), bg=self.BG_COLOR, fg="#888888", justify="center")
        lbl_info.pack(pady=15)

        # Utilisation de Figure (Orienté Objet) pour que Pyplot ignore ce canvas
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
        collection = PolyCollection(verts, facecolors=colors, edgecolors=self.BG_COLOR, linewidths=1.5)
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
            'is_dragging': False,
            'target_tag': None,
            'last_painted_idx': None
        }

        def update_plot():
            new_colors = np.where(tags == "FUEL", color_fuel, color_cr)
            collection.set_facecolors(new_colors)
            canvas.draw_idle()

        def get_closest_hex(event):
            if event.inaxes != ax: return None
            dist = np.sqrt((centers[:, 0] - event.xdata)**2 + (centers[:, 1] - event.ydata)**2)
            idx = np.argmin(dist)
            if dist[idx] <= apotheme:
                return idx
            return None

        def on_press(event):
            idx = get_closest_hex(event)
            if idx is not None:
                paint_state['is_dragging'] = True
                paint_state['target_tag'] = "CR" if tags[idx] == "FUEL" else "FUEL"
                paint_state['last_painted_idx'] = idx
                tags[idx] = paint_state['target_tag']
                update_plot()

        def on_motion(event):
            if not paint_state['is_dragging']: return
            idx = get_closest_hex(event)
            if idx is not None and idx != paint_state['last_painted_idx']:
                tags[idx] = paint_state['target_tag']
                paint_state['last_painted_idx'] = idx
                update_plot()

        def on_release(event):
            paint_state['is_dragging'] = False
            paint_state['last_painted_idx'] = None

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
            output_file = os.path.join(base_dir, "output", "meshes", "reactor_core_manual.msh")
            self._run_gmsh(geom, p, output_file, show_popup=True)

        ttk.Button(frame_btns, text="👁 Aperçu Détaillé (Crayons)", command=preview_custom).pack(side="left", expand=True, fill="x", padx=5)
        ttk.Button(frame_btns, text="✔ Valider & Générer Maillage", command=generate_custom_mesh, style="Success.TButton").pack(side="right", expand=True, fill="x", padx=5)

    def _run_gmsh(self, geom, p, output_file, show_popup=False):
        try:
            mesh_gen = ReactorMeshGenerator(geometry=geom, params=p)
            mesh_gen.generate(output_filename=output_file)
            if show_popup:
                messagebox.showinfo("Maillage Terminé", f"Sauvegardé dans :\n{output_file}")
        except Exception as e:
            if show_popup:
                messagebox.showerror("Erreur Gmsh", f"Erreur lors du maillage :\n\n{str(e)}")
            else:
                print(f"Erreur silencieuse Gmsh : {e}")

    # ---------------------------------------------------------
    # ÉTUDE PARAMÉTRIQUE
    # ---------------------------------------------------------
    def action_parametric_setup(self):
        p_base = self.get_current_params()
        if not p_base: return

        top = tk.Toplevel(self)
        top.title("Configuration - Étude Paramétrique")
        top.geometry("850x450")
        top.minsize(800, 450)
        top.configure(bg=self.BG_COLOR)
        
        frame_left = ttk.Frame(top, padding=10)
        frame_left.pack(side="left", fill="y")
        
        frame_right = ttk.Frame(top, padding=10)
        frame_right.pack(side="right", fill="both", expand=True)

        var_rmin = tk.DoubleVar(value=10.0)
        var_rmax = tk.DoubleVar(value=30.0)
        var_rstep = tk.DoubleVar(value=2.0)
        var_epaisseur = tk.DoubleVar(value=5.0)
        
        frame_inputs = ttk.LabelFrame(frame_left, text="Paramètres de boucle", padding=15)
        frame_inputs.pack(fill="x", pady=10)

        ttk.Label(frame_inputs, text="Rn Minimum :", background=self.PANEL_BG).grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frame_inputs, textvariable=var_rmin, width=8).grid(row=0, column=1, sticky="e")

        ttk.Label(frame_inputs, text="Rn Maximum :", background=self.PANEL_BG).grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame_inputs, textvariable=var_rmax, width=8).grid(row=1, column=1, sticky="e")

        ttk.Label(frame_inputs, text="Pas (ΔRn) :", background=self.PANEL_BG).grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame_inputs, textvariable=var_rstep, width=8).grid(row=2, column=1, sticky="e")

        ttk.Separator(frame_inputs, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(frame_inputs, text="Épaisseur Réflec. :", background=self.PANEL_BG).grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(frame_inputs, textvariable=var_epaisseur, width=8).grid(row=4, column=1, sticky="e")

        progress_var = tk.DoubleVar(value=0.0)
        pb = ttk.Progressbar(frame_left, variable=progress_var, maximum=100)
        pb.pack(fill="x", pady=(15, 5))

        lbl_status = tk.Label(frame_left, text="Prêt à lancer...", font=("Segoe UI", 9, "italic"), bg=self.BG_COLOR, fg="#888888")
        lbl_status.pack(pady=5)

        # Utilisation de Figure (Orienté Objet) pour le widget paramétrique
        fig = Figure(figsize=(5, 5))
        fig.patch.set_facecolor(self.BG_COLOR)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.BG_COLOR)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Aperçu en direct", fontsize=10, color=self.FG_COLOR)
        
        canvas = FigureCanvasTkAgg(fig, master=frame_right)
        canvas.get_tk_widget().pack(fill="both", expand=True)

        def start_thread():
            r_min, r_max, r_step = var_rmin.get(), var_rmax.get(), var_rstep.get()
            epaisseur = var_epaisseur.get()
            
            if r_min >= r_max or r_step <= 0:
                messagebox.showerror("Erreur", "Paramètres invalides.", parent=top)
                return
                
            btn_start.state(['disabled'])
            lbl_status.config(text="Initialisation...")
            
            max_limit = r_max + epaisseur
            ax.set_xlim(-max_limit, max_limit)
            ax.set_ylim(-max_limit, max_limit)
            
            thread = threading.Thread(
                target=self._parametric_worker,
                args=(r_min, r_max, r_step, epaisseur, p_base, top, progress_var, lbl_status, ax, canvas)
            )
            thread.daemon = True
            thread.start()

        btn_start = ttk.Button(frame_left, text="Démarrer le Batch", command=start_thread, style="Warning.TButton")
        btn_start.pack(pady=(20, 0), fill="x")

    def _parametric_worker(self, r_min, r_max, r_step, epaisseur, p_base, top_window, progress_var, lbl_status, ax, canvas):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        output_dir = os.path.join(base_dir, "output", "meshes", "parametric")
        os.makedirs(output_dir, exist_ok=True)

        r_vals = np.arange(r_min, r_max + 1e-9, r_step)
        total_steps = len(r_vals)
        seen_assemblies = set()
        generated_count = 0

        for i, r_n in enumerate(r_vals):
            geom = ReactorGeometry(R_n=r_n, R_hex=p_base["R_hex"])
            centers, tags, _ = geom.get_tagged_assemblies(n_cr_rings=p_base["cr_rings"], cr_density=p_base["cr_density"])
            n_assemblies = len(centers)

            if n_assemblies in seen_assemblies or n_assemblies == 0:
                self.after(0, self._update_ui_progress, progress_var, lbl_status, i + 1, total_steps, "Saut (topologie identique)")
                continue

            seen_assemblies.add(n_assemblies)

            p_current = p_base.copy()
            p_current["R_noyau"] = r_n
            p_current["R_reflec"] = r_n + epaisseur

            self.after(0, self._update_live_plot, ax, canvas, p_current, geom, centers, tags)

            filename = os.path.join(output_dir, f"mesh_N{n_assemblies}_Rn{r_n:.1f}.msh")
            self._run_gmsh(geom, p_current, filename, show_popup=False)
            generated_count += 1
            
            self.after(0, self._update_ui_progress, progress_var, lbl_status, i + 1, total_steps, f"Généré N={n_assemblies}")

        self.after(0, self._finish_parametric, top_window, generated_count, output_dir)

    def _update_live_plot(self, ax, canvas, p, geom, centers, tags):
        ax.clear()
        ax.axis("off")
        max_limit = p["R_reflec"] * 1.1
        ax.set_xlim(-max_limit, max_limit)
        ax.set_ylim(-max_limit, max_limit)
        ax.set_title(f"Rn = {p['R_noyau']:.1f} cm | {len(centers)} Assemblages", fontsize=10, color=self.FG_COLOR)

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
                ax.add_collection(PolyCollection(fuel_verts, facecolors="#264f78", edgecolors=self.BG_COLOR, linewidths=1.5, alpha=0.8))
            
            if np.any(mask_cr):
                cr_verts = centers[mask_cr][:, None, :] + base_hex[None, :, :]
                ax.add_collection(PolyCollection(cr_verts, facecolors="#842029", edgecolors=self.BG_COLOR, linewidths=1.5, alpha=0.8))

        canvas.draw_idle()

    def _update_ui_progress(self, progress_var, lbl_status, current, total, msg):
        percent = (current / total) * 100
        progress_var.set(percent)
        lbl_status.config(text=f"Étape {current}/{total} : {msg}")

    def _finish_parametric(self, top_window, generated_count, output_dir):
        messagebox.showinfo(
            "Batch Terminé",
            f"Génération terminée !\n\n{generated_count} maillages uniques générés.\nDossier : {output_dir}",
            parent=top_window
        )
        top_window.destroy()


if __name__ == "__main__":
    app = ReactorGUI()
    app.mainloop()
