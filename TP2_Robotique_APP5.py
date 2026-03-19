"""
TP2 — Robotique : Résoudre un labyrinthe avec A*
Implémentation de Dijkstra bidirectionnel et A* bidirectionnel

Auteurs : [Binôme]
Date    : 13 mars 2026
"""

import heapq
import math
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# Question 1 — Dijkstra bidirectionnel
# ============================================================

def dijkstra_bidirectionnel(G, start, goal):
    """
    Algorithme de Dijkstra bidirectionnel sur un graphe NetworkX pondéré.

    Critère d'arrêt :
        Soit µ la longueur du meilleur chemin s–t vu jusqu'ici.
        Soit topf (resp. topr) le plus petit f-value dans la file avant
        (resp. arrière). On s'arrête dès que topf + topr >= µ
        (critère de Goldberg & al.).

    Retourne :
        (distance, chemin, noeuds_explores_fwd, noeuds_explores_bwd)
    """
    if start == goal:
        return 0, [start], set(), set()

    # --- Distances et prédécesseurs ---
    dist_f = {start: 0}   # dist depuis start
    dist_b = {goal:  0}   # dist depuis goal (graphe inversé)
    prev_f = {start: None}
    prev_b = {goal:  None}

    # --- Files de priorité (distance, sommet) ---
    open_f = [(0, start)]
    open_b = [(0, goal)]

    # --- Sommets fermés ---
    closed_f = set()
    closed_b = set()

    mu = math.inf          # longueur du meilleur chemin connu
    best_node = None       # sommet de jonction

    explored_f = set()
    explored_b = set()

    def relax_forward(u):
        nonlocal mu, best_node
        for v, data in G[u].items():
            w = data.get('weight', 1)
            new_d = dist_f[u] + w
            if new_d < dist_f.get(v, math.inf):
                dist_f[v] = new_d
                prev_f[v] = u
                heapq.heappush(open_f, (new_d, v))
            # mise à jour µ si v est déjà fermé côté arrière
            if v in closed_b:
                path_len = dist_f.get(v, math.inf) + dist_b.get(v, math.inf)
                if path_len < mu:
                    mu = path_len
                    best_node = v

    def relax_backward(u):
        nonlocal mu, best_node
        # graphe inversé : on parcourt les prédécesseurs
        for v, data in G.pred[u].items() if G.is_directed() else G[u].items():
            w = data.get('weight', 1)
            new_d = dist_b[u] + w
            if new_d < dist_b.get(v, math.inf):
                dist_b[v] = new_d
                prev_b[v] = u
                heapq.heappush(open_b, (new_d, v))
            if v in closed_f:
                path_len = dist_f.get(v, math.inf) + dist_b.get(v, math.inf)
                if path_len < mu:
                    mu = path_len
                    best_node = v

    while open_f and open_b:
        # Critère d'arrêt : topf + topr >= µ
        top_f = open_f[0][0]
        top_b = open_b[0][0]
        if top_f + top_b >= mu:
            break

        # --- Expansion avant ---
        while open_f and open_f[0][0] == top_f:
            d, u = heapq.heappop(open_f)
            if u in closed_f:
                continue
            closed_f.add(u)
            explored_f.add(u)
            relax_forward(u)
            break   # on alterne après chaque extraction

        # Vérification arrêt après expansion avant
        if open_f and open_b:
            top_f = open_f[0][0]
            top_b = open_b[0][0]
            if top_f + top_b >= mu:
                break

        # --- Expansion arrière ---
        while open_b and open_b[0][0] == top_b:
            d, u = heapq.heappop(open_b)
            if u in closed_b:
                continue
            closed_b.add(u)
            explored_b.add(u)
            relax_backward(u)
            break

    # Reconstruction du chemin
    if best_node is None or mu == math.inf:
        return math.inf, [], explored_f, explored_b

    path = _reconstruct_path(prev_f, prev_b, best_node)
    return mu, path, explored_f, explored_b


def _reconstruct_path(prev_f, prev_b, meeting):
    """Reconstruit le chemin à partir des deux dictionnaires de prédécesseurs."""
    # Partie avant : start → meeting
    path_f = []
    node = meeting
    while node is not None:
        path_f.append(node)
        node = prev_f.get(node)
    path_f.reverse()

    # Partie arrière : meeting → goal
    path_b = []
    node = prev_b.get(meeting)
    while node is not None:
        path_b.append(node)
        node = prev_b.get(node)

    return path_f + path_b


# ============================================================
# Dijkstra classique (référence)
# ============================================================

def dijkstra_classique(G, start, goal):
    """Dijkstra classique — retourne (distance, chemin, noeuds_explores)."""
    dist   = {start: 0}
    prev   = {start: None}
    heap   = [(0, start)]
    closed = set()
    explored = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        explored.add(u)
        if u == goal:
            break
        for v, data in G[u].items():
            w = data.get('weight', 1)
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if goal not in dist:
        return math.inf, [], explored

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev.get(node)
    path.reverse()
    return dist[goal], path, explored


# ============================================================
# Question 2 — Test sur des graphes plus grands
# ============================================================

def test_graphes_grands():
    """Teste le Dijkstra bidirectionnel sur différents types de graphes."""
    print("=" * 60)
    print("Question 2 — Tests sur graphes plus grands")
    print("=" * 60)

    graphes = {
        "Random geometric (2000, 0.05)":
            nx.random_geometric_graph(2000, 0.05),
        "Erdős–Rényi (1000, 0.01)":
            nx.erdos_renyi_graph(1000, 0.01),
        "Barabási–Albert (1000, 5)":
            nx.barabasi_albert_graph(1000, 5),
    }

    for nom, G in graphes.items():
        # Ajout de poids aléatoires
        for u, v in G.edges():
            G[u][v]['weight'] = random.randint(1, 10)

        # Choix d'une paire connectée
        start, goal = _paire_connectee(G)
        if start is None:
            print(f"{nom} : aucune paire connectée trouvée.")
            continue

        t0 = time.perf_counter()
        dist_bi, _, exp_f, exp_b = dijkstra_bidirectionnel(G, start, goal)
        t_bi = time.perf_counter() - t0

        print(f"\n{nom}")
        print(f"  Noeuds : {G.number_of_nodes()}, Arêtes : {G.number_of_edges()}")
        print(f"  Dist({start}→{goal}) = {dist_bi:.2f}")
        print(f"  Temps : {t_bi*1000:.3f} ms")
        print(f"  Sommets explorés (fwd+bwd) : {len(exp_f)+len(exp_b)}")


def _paire_connectee(G, nb_essais=20):
    """Retourne une paire (start, goal) connectée dans G."""
    noeuds = list(G.nodes())
    for _ in range(nb_essais):
        s = random.choice(noeuds)
        t = random.choice(noeuds)
        if s != t and nx.has_path(G, s, t):
            return s, t
    return None, None


# ============================================================
# Question 3 — Comparaison avec NetworkX
# ============================================================

def comparer_avec_networkx(G, start, goal):
    """Compare notre implémentation avec nx.bidirectional_dijkstra."""
    print("\n" + "=" * 60)
    print("Question 3 — Comparaison avec NetworkX bidirectional_dijkstra")
    print("=" * 60)

    # Notre implémentation
    t0 = time.perf_counter()
    notre_dist, notre_path, _, _ = dijkstra_bidirectionnel(G, start, goal)
    t_notre = time.perf_counter() - t0

    # NetworkX
    t0 = time.perf_counter()
    nx_dist, nx_path = nx.bidirectional_dijkstra(G, start, goal, weight='weight')
    t_nx = time.perf_counter() - t0

    print(f"  Notre distance    : {notre_dist:.4f}  (temps : {t_notre*1000:.3f} ms)")
    print(f"  NetworkX distance : {nx_dist:.4f}  (temps : {t_nx*1000:.3f} ms)")
    print(f"  Distances égales  : {math.isclose(notre_dist, nx_dist, rel_tol=1e-9)}")


# ============================================================
# Question 4 — Comparaison Dijkstra classique vs bidirectionnel
# ============================================================

def comparer_dijkstra(G, start, goal):
    """Compare Dijkstra classique et bidirectionnel en termes de sommets explorés."""
    print("\n" + "=" * 60)
    print("Question 4 — Dijkstra classique vs bidirectionnel")
    print("=" * 60)

    t0 = time.perf_counter()
    d_cl, _, exp_cl = dijkstra_classique(G, start, goal)
    t_cl = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_bi, _, exp_f, exp_b = dijkstra_bidirectionnel(G, start, goal)
    t_bi = time.perf_counter() - t0

    exp_bi = len(exp_f) + len(exp_b)

    print(f"  Dijkstra classique   : dist={d_cl:.2f}, exploré={len(exp_cl)}, "
          f"temps={t_cl*1000:.3f} ms")
    print(f"  Dijkstra bidirection : dist={d_bi:.2f}, exploré={exp_bi}, "
          f"temps={t_bi*1000:.3f} ms")
    reduction = (1 - exp_bi / max(len(exp_cl), 1)) * 100
    print(f"  Réduction de sommets explorés : {reduction:.1f}%")
    print("  → Le bidirectionnel explore moins de sommets car les deux recherches")
    print("    se rencontrent au milieu, évitant d'explorer tout le graphe.")


# ============================================================
# Question 5 — A* bidirectionnel
# ============================================================

def astar_bidirectionnel(G, start, goal, heuristique):
    """
    Algorithme A* bidirectionnel sur un graphe NetworkX.

    Paramètres :
        heuristique(u, v) : estimation admissible de dist(u, v).

    Critère d'arrêt (Pohl 1969, repris par Rice & Tsotras 2012) :
        µ : meilleur chemin vu.
        kf = min f_f dans OPEN_f, kb = min f_b dans OPEN_b.
        On s'arrête dès que max(kf, kb) >= µ.

    Retourne :
        (distance, chemin, noeuds_explores_fwd, noeuds_explores_bwd)
    """
    if start == goal:
        return 0, [start], set(), set()

    dist_f = {start: 0}
    dist_b = {goal:  0}
    prev_f = {start: None}
    prev_b = {goal:  None}

    # f(v) = g(v) + h(v)
    hf = lambda v: heuristique(v, goal)   # heuristique forward
    hb = lambda v: heuristique(v, start)  # heuristique backward

    open_f = [(hf(start), 0, start)]   # (f, g, noeud)
    open_b = [(hb(goal),  0, goal)]

    closed_f = set()
    closed_b = set()

    mu = math.inf
    best_node = None
    explored_f = set()
    explored_b = set()

    def expand_f():
        nonlocal mu, best_node
        while open_f:
            f, g, u = heapq.heappop(open_f)
            if u in closed_f:
                continue
            closed_f.add(u)
            explored_f.add(u)
            for v, data in G[u].items():
                w = data.get('weight', 1)
                ng = g + w
                if ng < dist_f.get(v, math.inf):
                    dist_f[v] = ng
                    prev_f[v] = u
                    heapq.heappush(open_f, (ng + hf(v), ng, v))
                # Mise à jour µ via l'arête (u,v) si v fermé côté arrière
                if v in closed_b:
                    candidate = ng + dist_b.get(v, math.inf)
                    if candidate < mu:
                        mu = candidate
                        best_node = v
            return
        return

    def expand_b():
        nonlocal mu, best_node
        while open_b:
            f, g, u = heapq.heappop(open_b)
            if u in closed_b:
                continue
            closed_b.add(u)
            explored_b.add(u)
            for v, data in (G.pred[u].items() if G.is_directed() else G[u].items()):
                w = data.get('weight', 1)
                ng = g + w
                if ng < dist_b.get(v, math.inf):
                    dist_b[v] = ng
                    prev_b[v] = u
                    heapq.heappush(open_b, (ng + hb(v), ng, v))
                if v in closed_f:
                    candidate = dist_f.get(v, math.inf) + ng
                    if candidate < mu:
                        mu = candidate
                        best_node = v
            return
        return

    while open_f and open_b:
        kf = open_f[0][0]   # min f-value côté forward
        kb = open_b[0][0]   # min f-value côté backward
        # Critère d'arrêt de Pohl (1969) : max(kf, kb) >= µ
        if max(kf, kb) >= mu:
            break
        # Critère alternatif : kf + kb >= µ (plus agressif, Rice & Tsotras 2012)
        # Si µ < inf, on peut aussi s'arrêter plus tôt
        if mu < math.inf and kf + kb >= mu:
            break
        # On étend la direction avec le plus petit f-value
        if kf <= kb:
            expand_f()
        else:
            expand_b()

    if best_node is None or mu == math.inf:
        return math.inf, [], explored_f, explored_b

    path = _reconstruct_path(prev_f, prev_b, best_node)
    return mu, path, explored_f, explored_b


# ============================================================
# Question 6 — Comparaison Dijkstra bidirectionnel vs A* bidirectionnel
# ============================================================

def comparer_dijkstra_astar(G, start, goal, heuristique):
    """Compare Dijkstra bidirectionnel et A* bidirectionnel."""
    print("\n" + "=" * 60)
    print("Question 6 — Dijkstra bidirectionnel vs A* bidirectionnel")
    print("=" * 60)

    t0 = time.perf_counter()
    d_dij, _, exp_f_d, exp_b_d = dijkstra_bidirectionnel(G, start, goal)
    t_dij = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_ast, _, exp_f_a, exp_b_a = astar_bidirectionnel(G, start, goal, heuristique)
    t_ast = time.perf_counter() - t0

    exp_dij = len(exp_f_d) + len(exp_b_d)
    exp_ast = len(exp_f_a) + len(exp_b_a)

    print(f"  Dijkstra bi  : dist={d_dij:.2f}, exploré={exp_dij}, "
          f"temps={t_dij*1000:.3f} ms")
    print(f"  A* bi        : dist={d_ast:.2f}, exploré={exp_ast}, "
          f"temps={t_ast*1000:.3f} ms")
    reduction = (1 - exp_ast / max(exp_dij, 1)) * 100
    print(f"  Réduction de sommets explorés : {reduction:.1f}%")
    print("  → A* guide la recherche grâce à l'heuristique, réduisant le nombre")
    print("    de sommets explorés par rapport à Dijkstra bidirectionnel.")


# ============================================================
# Graphe exemple du sujet (petit graphe pour Q1)
# ============================================================

def construire_petit_graphe():
    """
    Petit graphe pondéré pour tester Q1.
    Topologie :
        A --2-- B --3-- C
        |       |       |
        4       1       2
        |       |       |
        D --5-- E --1-- F
    """
    G = nx.Graph()
    aretes = [
        ('A', 'B', 2), ('B', 'C', 3),
        ('A', 'D', 4), ('B', 'E', 1), ('C', 'F', 2),
        ('D', 'E', 5), ('E', 'F', 1),
    ]
    for u, v, w in aretes:
        G.add_edge(u, v, weight=w)
    return G


def test_petit_graphe():
    """Test sur le petit graphe défini ci-dessus."""
    print("=" * 60)
    print("Question 1 — Test sur le petit graphe exemple")
    print("=" * 60)
    G = construire_petit_graphe()
    start, goal = 'A', 'F'

    dist_bi, path_bi, exp_f, exp_b = dijkstra_bidirectionnel(G, start, goal)
    dist_cl, path_cl, exp_cl       = dijkstra_classique(G, start, goal)

    print(f"  Dijkstra classique   : dist={dist_cl}, chemin={path_cl}")
    print(f"  Dijkstra bi          : dist={dist_bi}, chemin={path_bi}")
    assert math.isclose(dist_bi, dist_cl), "Les distances ne correspondent pas !"
    print("  ✓ Les deux distances sont identiques.")
    print(f"\n  Critère d'arrêt : l'algorithme s'arrête dès que")
    print("  topf + topr >= µ, où µ est le meilleur chemin vu.")


# ============================================================
# Graphe grille 2D (labyrinthe) pour A* avec heuristique euclidienne
# ============================================================

def construire_grille(rows=20, cols=20, p_obstacle=0.25):
    """
    Construit un graphe grille 2D (labyrinthe aléatoire).
    Les noeuds sont des tuples (i, j). Les arêtes ont un poids 1.
    p_obstacle : probabilité qu'une case soit un mur (supprimée).
    """
    G = nx.grid_2d_graph(rows, cols)
    murs = [(i, j) for (i, j) in G.nodes()
            if random.random() < p_obstacle and (i, j) not in [(0, 0), (rows-1, cols-1)]]
    G.remove_nodes_from(murs)
    for u, v in G.edges():
        G[u][v]['weight'] = 1
    return G


def heuristique_euclidienne(u, v):
    """Heuristique euclidienne entre deux noeuds (i,j)."""
    return math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)


def heuristique_manhattan(u, v):
    """Heuristique de Manhattan entre deux noeuds (i,j)."""
    return abs(u[0] - v[0]) + abs(u[1] - v[1])


def test_labyrinthe():
    """Test complet sur une grille 2D (labyrinthe)."""
    print("\n" + "=" * 60)
    print("Test labyrinthe 2D — A* bi vs Dijkstra bi")
    print("=" * 60)

    random.seed(42)
    G = construire_grille(30, 30, p_obstacle=0.25)

    start, goal = (0, 0), (29, 29)
    if start not in G or goal not in G or not nx.has_path(G, start, goal):
        print("  Pas de chemin trouvé avec cette graine, essai avec p=0.15")
        G = construire_grille(30, 30, p_obstacle=0.15)

    if not nx.has_path(G, start, goal):
        print("  Aucun chemin disponible.")
        return

    comparer_dijkstra(G, start, goal)
    comparer_dijkstra_astar(G, start, goal, heuristique_manhattan)

    # Visualisation
    _visualiser_labyrinthe(G, start, goal)


def _visualiser_labyrinthe(G, start, goal):
    """Visualise le labyrinthe avec les chemins trouvés."""
    rows = max(n[0] for n in G.nodes()) + 1
    cols = max(n[1] for n in G.nodes()) + 1

    _, path_dij, exp_f_d, exp_b_d = dijkstra_bidirectionnel(G, start, goal)
    _, path_ast, exp_f_a, exp_b_a = astar_bidirectionnel(
        G, start, goal, heuristique_manhattan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, (path, exp_f, exp_b, titre) in zip(axes, [
        (path_dij, exp_f_d, exp_b_d, "Dijkstra bidirectionnel"),
        (path_ast, exp_f_a, exp_b_a, "A* bidirectionnel"),
    ]):
        grid = [[0.0] * cols for _ in range(rows)]
        all_nodes = set(G.nodes())

        for i in range(rows):
            for j in range(cols):
                if (i, j) not in all_nodes:
                    grid[i][j] = -1   # mur

        for n in exp_f:
            if grid[n[0]][n[1]] == 0:
                grid[n[0]][n[1]] = 0.4
        for n in exp_b:
            if grid[n[0]][n[1]] == 0:
                grid[n[0]][n[1]] = 0.6

        for n in path:
            grid[n[0]][n[1]] = 1.0

        import numpy as np
        from matplotlib.colors import ListedColormap
        arr = np.array(grid)
        base = plt.colormaps.get_cmap('RdYlGn')
        colors = ['black'] + [base(i/255) for i in range(256)]
        cmap = ListedColormap(colors)
        # valeurs: -1=mur(idx 0), 0..1 mappé sur idx 1..256
        arr_idx = ((arr + 1) * 128).clip(0, 256).astype(int)
        ax.imshow(arr_idx, cmap=cmap, vmin=0, vmax=256)
        ax.set_title(f"{titre}\n{len(exp_f)+len(exp_b)} sommets explorés")
        ax.axis('off')

        legende = [
            mpatches.Patch(color='black',            label='Mur'),
            mpatches.Patch(color=base(0.4),          label='Exploré (fwd)'),
            mpatches.Patch(color=base(0.6),          label='Exploré (bwd)'),
            mpatches.Patch(color=base(1.0),          label='Chemin'),
        ]
        ax.legend(handles=legende, loc='lower right', fontsize=7)

    plt.suptitle("Comparaison Dijkstra bi vs A* bi sur labyrinthe 30×30", fontsize=13)
    plt.tight_layout()
    import os
    outdir = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(outdir, 'labyrinthe_comparaison.png')
    plt.savefig(outfile, dpi=150)
    plt.show()
    print("  Visualisation sauvegardée dans labyrinthe_comparaison.png")


# ============================================================
# Visualisations pour chaque question
# ============================================================

import os as _os
import numpy as _np

def _savefig(nom):
    outdir = _os.path.dirname(_os.path.abspath(__file__))
    path = _os.path.join(outdir, nom)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  → Graphique sauvegardé : {nom}")


def visu_q1_petit_graphe():
    """Q1 — Visualise le petit graphe avec le chemin bidirectionnel."""
    G = construire_petit_graphe()
    start, goal = 'A', 'F'
    _, path_bi, exp_f, exp_b = dijkstra_bidirectionnel(G, start, goal)
    _, path_cl, exp_cl       = dijkstra_classique(G, start, goal)

    pos = {'A': (0,1), 'B': (1,1), 'C': (2,1),
           'D': (0,0), 'E': (1,0), 'F': (2,0)}
    labels = {u: f"{u}" for u in G.nodes()}
    edge_labels = {(u,v): d['weight'] for u,v,d in G.edges(data=True)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (explored, path, titre) in zip(axes, [
        (exp_cl,        path_cl, "Dijkstra classique"),
        (exp_f | exp_b, path_bi, "Dijkstra bidirectionnel"),
    ]):
        path_edges = list(zip(path, path[1:]))
        node_colors = []
        for n in G.nodes():
            if n == start or n == goal:
                node_colors.append('#e74c3c')
            elif n in path:
                node_colors.append('#2ecc71')
            elif n in explored:
                node_colors.append('#3498db')
            else:
                node_colors.append('#ecf0f1')

        nx.draw_networkx(G, pos=pos, ax=ax, labels=labels,
                         node_color=node_colors, node_size=900,
                         font_weight='bold', edge_color='#bdc3c7', width=2)
        nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=path_edges,
                               edge_color='#e74c3c', width=4)
        nx.draw_networkx_edge_labels(G, pos=pos, ax=ax,
                                     edge_labels=edge_labels, font_size=10)
        ax.set_title(f"{titre}\n{len(explored)} sommets explorés | chemin={path}", fontsize=10)
        ax.axis('off')

    legende = [
        mpatches.Patch(color='#e74c3c', label='Start / Goal'),
        mpatches.Patch(color='#3498db', label='Exploré'),
        mpatches.Patch(color='#2ecc71', label='Chemin optimal'),
        mpatches.Patch(color='#ecf0f1', label='Non exploré'),
    ]
    fig.legend(handles=legende, loc='lower center', ncol=4, fontsize=9)
    plt.suptitle("Q1 — Dijkstra classique vs bidirectionnel (petit graphe)", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _savefig('q1_petit_graphe.png')
    plt.show()


def _couleur_noeuds(nodes, start, goal, explored_f, explored_b, path_set):
    """Retourne la liste de couleurs pour une liste de noeuds."""
    colors = []
    for n in nodes:
        if n == start or n == goal:
            colors.append('#e74c3c')
        elif n in path_set:
            colors.append('#f39c12')
        elif n in explored_f:
            colors.append('#3498db')
        elif n in explored_b:
            colors.append('#9b59b6')
        else:
            colors.append('#d5d8dc')
    return colors


def _draw_graph_visu(G, pos, start, goal, explored_f, explored_b, path,
                     ax, titre, node_size=15, with_labels=False):
    """Dessine un graphe avec les noeuds colorés selon leur état."""
    path_set = set(path)
    path_edges = list(zip(path, path[1:]))
    node_colors = _couleur_noeuds(G.nodes(), start, goal,
                                  explored_f, explored_b, path_set)

    nx.draw_networkx_edges(G, pos=pos, ax=ax,
                           edge_color='#aab7b8', alpha=0.4, width=0.5)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=path_edges,
                           edge_color='#f39c12', width=2.5, alpha=0.9)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax,
                           node_color=node_colors, node_size=node_size)
    if with_labels:
        nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=7)
    ax.set_title(titre, fontsize=9)
    ax.axis('off')


def visu_q2_graphes_grands():
    """Q2 — Pour chaque type de graphe : visualisation du graphe + exploration
    côte à côte (Dijkstra classique vs bidirectionnel), plus barplots synthèse."""

    random.seed(0)
    graphes_def = [
        ("Random geometric (2000, 0.05)",
         nx.random_geometric_graph(2000, 0.05),
         None),           # pos sera extrait des attributs
        ("Erdős–Rényi (1000, 0.01)",
         nx.erdos_renyi_graph(1000, 0.01),
         None),
        ("Barabási–Albert (1000, 5)",
         nx.barabasi_albert_graph(1000, 5),
         None),
    ]

    noms_bar, explores_cl, explores_bi, temps_cl_list, temps_bi_list = [], [], [], [], []

    for nom, G, _ in graphes_def:
        for u, v in G.edges():
            G[u][v]['weight'] = random.randint(1, 10)
        s, t = _paire_connectee(G)
        if s is None:
            continue

        t0 = time.perf_counter()
        d_cl, path_cl, exp_cl = dijkstra_classique(G, s, t)
        t_cl = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        d_bi, path_bi, ef, eb = dijkstra_bidirectionnel(G, s, t)
        t_bi = (time.perf_counter() - t0) * 1000

        noms_bar.append(nom.split(" (")[0])  # version courte pour barplot
        explores_cl.append(len(exp_cl))
        explores_bi.append(len(ef) + len(eb))
        temps_cl_list.append(t_cl)
        temps_bi_list.append(t_bi)

        # --- Positions ---
        if "geometric" in nom.lower():
            pos = {n: G.nodes[n]['pos'] for n in G.nodes()}
        else:
            pos = nx.spring_layout(G, seed=1, k=0.3)

        # --- Figure de visualisation du graphe ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        _draw_graph_visu(G, pos, s, t, exp_cl, set(), path_cl,
                         axes[0],
                         f"Dijkstra classique\n{len(exp_cl)} sommets explorés",
                         node_size=12)
        _draw_graph_visu(G, pos, s, t, ef, eb, path_bi,
                         axes[1],
                         f"Dijkstra bidirectionnel\n{len(ef)+len(eb)} sommets explorés",
                         node_size=12)

        legende = [
            mpatches.Patch(color='#e74c3c', label='Start / Goal'),
            mpatches.Patch(color='#3498db', label='Exploré fwd'),
            mpatches.Patch(color='#9b59b6', label='Exploré bwd'),
            mpatches.Patch(color='#f39c12', label='Chemin'),
            mpatches.Patch(color='#d5d8dc', label='Non exploré'),
        ]
        fig.legend(handles=legende, loc='lower center', ncol=5, fontsize=9)
        plt.suptitle(f"Q2 — {nom}", fontsize=12)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        safe_nom = nom.split(" (")[0].replace("–", "").replace(" ", "_").lower()
        _savefig(f'q2_{safe_nom}.png')
        plt.show()

    # --- Barplots synthèse ---
    x = _np.arange(len(noms_bar))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    b1 = ax1.bar(x - w/2, explores_cl, w, label='Classique',      color='#e74c3c')
    b2 = ax1.bar(x + w/2, explores_bi, w, label='Bidirectionnel', color='#3498db')
    ax1.set_xticks(x); ax1.set_xticklabels(noms_bar, fontsize=9)
    ax1.set_ylabel("Sommets explorés"); ax1.legend()
    ax1.set_title("Sommets explorés — classique vs bidirectionnel")
    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(bar.get_height()), ha='center', va='bottom', fontsize=8)

    b3 = ax2.bar(x - w/2, temps_cl_list, w, label='Classique',      color='#e74c3c')
    b4 = ax2.bar(x + w/2, temps_bi_list, w, label='Bidirectionnel', color='#3498db')
    ax2.set_xticks(x); ax2.set_xticklabels(noms_bar, fontsize=9)
    ax2.set_ylabel("Temps (ms)"); ax2.legend()
    ax2.set_title("Temps de requête — classique vs bidirectionnel")
    for bar in list(b3) + list(b4):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=8)

    plt.suptitle("Q2 — Synthèse : Dijkstra classique vs bidirectionnel", fontsize=12)
    plt.tight_layout()
    _savefig('q2_synthese.png')
    plt.show()

def visu_q3_q4_comparaison(G, start, goal):
    """Q3 & Q4 — Barplot comparatif Dijkstra classique / bi / NetworkX."""
    # Classique
    t0 = time.perf_counter()
    d_cl, _, exp_cl = dijkstra_classique(G, start, goal)
    t_cl = (time.perf_counter() - t0) * 1000

    # Notre bidirectionnel
    t0 = time.perf_counter()
    d_bi, _, ef, eb = dijkstra_bidirectionnel(G, start, goal)
    t_bi = (time.perf_counter() - t0) * 1000

    # NetworkX
    t0 = time.perf_counter()
    d_nx, _ = nx.bidirectional_dijkstra(G, start, goal, weight='weight')
    t_nx = (time.perf_counter() - t0) * 1000

    algos    = ["Dijkstra\nclassique", "Dijkstra\nbi (notre)", "NetworkX\nbidirectional"]
    explores = [len(exp_cl), len(ef) + len(eb), None]  # NetworkX ne retourne pas ce chiffre
    temps    = [t_cl, t_bi, t_nx]
    colors   = ['#e74c3c', '#3498db', '#2ecc71']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sommets explorés (sans NetworkX)
    ax1.bar(algos[:2], explores[:2], color=colors[:2], width=0.4)
    ax1.set_ylabel("Sommets explorés")
    ax1.set_title("Q4 — Sommets explorés")
    for i, val in enumerate(explores[:2]):
        ax1.text(i, val + 1, str(val), ha='center', va='bottom',
                 fontweight='bold', fontsize=11)
    reduc = (1 - explores[1] / max(explores[0], 1)) * 100
    ax1.set_xlabel(f"Réduction bidirectionnel : {reduc:.1f}%", fontsize=10)

    # Temps
    bars = ax2.bar(algos, temps, color=colors, width=0.4)
    ax2.set_ylabel("Temps (ms)")
    ax2.set_title("Q3 & Q4 — Temps de requête")
    for bar, val in zip(bars, temps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f} ms", ha='center', va='bottom', fontsize=9)
    ax2.set_xlabel(f"Distances : classique={d_cl:.1f} | bi={d_bi:.1f} | nx={d_nx:.1f}",
                   fontsize=9)

    plt.suptitle("Q3 & Q4 — Comparaison Dijkstra classique / bidirectionnel / NetworkX",
                 fontsize=11)
    plt.tight_layout()
    _savefig('q3_q4_comparaison.png')
    plt.show()


def visu_q6_astar(G_grid, start, goal):
    """Q6 — Barplot + visualisation grille Dijkstra bi vs A* bi."""
    # Mesures
    t0 = time.perf_counter()
    d_dij, _, ef_d, eb_d = dijkstra_bidirectionnel(G_grid, start, goal)
    t_dij = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    d_ast, _, ef_a, eb_a = astar_bidirectionnel(G_grid, start, goal, heuristique_manhattan)
    t_ast = (time.perf_counter() - t0) * 1000

    exp_dij = len(ef_d) + len(eb_d)
    exp_ast = len(ef_a) + len(eb_a)

    fig = plt.figure(figsize=(15, 6))
    gs  = fig.add_gridspec(1, 3, width_ratios=[1, 1.5, 1.5])

    # --- Barplots ---
    ax0 = fig.add_subplot(gs[0])
    algos  = ["Dijkstra bi", "A* bi"]
    colors = ['#3498db', '#e74c3c']
    ax0.bar(algos, [exp_dij, exp_ast], color=colors, width=0.4)
    ax0.set_ylabel("Sommets explorés")
    ax0.set_title("Sommets explorés")
    for i, v in enumerate([exp_dij, exp_ast]):
        ax0.text(i, v + 2, str(v), ha='center', va='bottom', fontweight='bold')
    reduc = (1 - exp_ast / max(exp_dij, 1)) * 100
    ax0.set_xlabel(f"A* réduit de {reduc:.1f}%\nt_dij={t_dij:.1f}ms  t_ast={t_ast:.1f}ms",
                   fontsize=8)

    # --- Grilles ---
    rows = max(n[0] for n in G_grid.nodes()) + 1
    cols = max(n[1] for n in G_grid.nodes()) + 1
    all_nodes = set(G_grid.nodes())

    from matplotlib.colors import ListedColormap
    base = plt.colormaps.get_cmap('RdYlGn')
    cmap_colors = ['black'] + [base(i/255) for i in range(256)]
    cmap = ListedColormap(cmap_colors)

    _, path_dij, _, _ = dijkstra_bidirectionnel(G_grid, start, goal)
    _, path_ast, _, _ = astar_bidirectionnel(G_grid, start, goal, heuristique_manhattan)

    for ax_idx, (exp_f, exp_b, path, titre) in enumerate([
        (ef_d, eb_d, path_dij, f"Dijkstra bi\n{exp_dij} explorés"),
        (ef_a, eb_a, path_ast, f"A* bi (Manhattan)\n{exp_ast} explorés"),
    ]):
        ax = fig.add_subplot(gs[ax_idx + 1])
        grid = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                if (i, j) not in all_nodes:
                    grid[i][j] = -1
        for n in exp_f:
            if grid[n[0]][n[1]] == 0: grid[n[0]][n[1]] = 0.35
        for n in exp_b:
            if grid[n[0]][n[1]] == 0: grid[n[0]][n[1]] = 0.6
        for n in path:
            grid[n[0]][n[1]] = 1.0

        arr = _np.array(grid)
        arr_idx = ((arr + 1) * 128).clip(0, 256).astype(int)
        ax.imshow(arr_idx, cmap=cmap, vmin=0, vmax=256)
        ax.set_title(titre, fontsize=10)
        ax.axis('off')

    legende = [
        mpatches.Patch(color='black',   label='Mur'),
        mpatches.Patch(color=base(0.35), label='Exploré fwd'),
        mpatches.Patch(color=base(0.6),  label='Exploré bwd'),
        mpatches.Patch(color=base(1.0),  label='Chemin'),
    ]
    fig.legend(handles=legende, loc='lower center', ncol=4, fontsize=9)
    plt.suptitle("Q6 — Dijkstra bi vs A* bi sur labyrinthe 30×30", fontsize=12)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _savefig('q6_astar_vs_dijkstra.png')
    plt.show()


# ============================================================
# Programme principal
# ============================================================

if __name__ == '__main__':
    random.seed(0)

    # --- Q1 : petit graphe (texte + graphique) ---
    test_petit_graphe()
    visu_q1_petit_graphe()

    # --- Q2 : graphes plus grands (texte + graphique) ---
    test_graphes_grands()
    visu_q2_graphes_grands()

    # --- Q3 & Q4 : comparaison NetworkX + classique vs bi (texte + graphique) ---
    print("\n" + "=" * 60)
    print("Graphe pour Q3–Q4 : Erdős–Rényi (500 noeuds)")
    print("=" * 60)
    G_er = nx.erdos_renyi_graph(500, 0.05, seed=1)
    for u, v in G_er.edges():
        G_er[u][v]['weight'] = random.randint(1, 20)
    start_er, goal_er = _paire_connectee(G_er)

    if start_er is not None:
        comparer_avec_networkx(G_er, start_er, goal_er)
        comparer_dijkstra(G_er, start_er, goal_er)
        visu_q3_q4_comparaison(G_er, start_er, goal_er)

    # --- Q5 & Q6 : A* bidirectionnel (texte + graphique) ---
    test_labyrinthe()

    random.seed(42)
    G_grid = construire_grille(30, 30, p_obstacle=0.25)
    grid_start, grid_goal = (0, 0), (29, 29)
    if grid_start in G_grid and grid_goal in G_grid and nx.has_path(G_grid, grid_start, grid_goal):
        visu_q6_astar(G_grid, grid_start, grid_goal)

    print("\n✓ Toutes les questions ont été traitées.")