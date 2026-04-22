#!/usr/bin/env python3
"""
Subgradient Algorithm (PDF'deki Algorithm 1) + Instance Generation + DSatur Graph Coloring

Bu script:
1. n=10 için bir APC instance üretir
2. Subgradient algoritmasını çalıştırır
3. Conflict graph'ı oluşturur:
   - Explicit conflicts (instance'dan gelen)
   - Natural AP conflicts (aynı satır/sütundaki edge'ler birbirleriyle çakışır)
4. DSatur algoritmasıyla graf renklendirme yapar
5. Kaç renkle gruplandığını raporlar
"""

import numpy as np
import random
import time
import sys
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════
#  INSTANCE GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_instance(n=10, num_conflicts=20, seed=None, conflict_density=None):
    """
    n x n bipartite assignment problem with conflicts üretir.
    Her çalıştırmada farklı instance üretir (seed=None ise).

    Args:
        n: problem boyutu
        num_conflicts: explicit conflict sayısı (conflict_density verilmezse)
        seed: None ise her seferinde farklı, int verilirse tekrarlanabilir
        conflict_density: 0.0-1.0 arası, verilirse num_conflicts'i override eder
                          max possible = n^2*(n-1)^2/2 (farklı satır+sütun çiftleri)
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2**31)
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed % (2**31))

    # Conflict density verilmişse num_conflicts hesapla
    if conflict_density is not None:
        # Farklı satır VE farklı sütundan olabilecek edge çifti sayısı
        max_possible = 0
        for i1 in range(n):
            for j1 in range(n):
                for i2 in range(i1 + 1, n):
                    for j2 in range(n):
                        if j1 != j2:
                            max_possible += 1
        num_conflicts = max(1, int(conflict_density * max_possible))

    # Cost matrix: rastgele ağırlıklar [1, 100]
    cost_matrix = np_rng.randint(1, 101, size=(n, n)).tolist()

    # Tüm edge'ler
    all_edges = [(i, j) for i in range(n) for j in range(n)]

    # Conflict üretimi: rastgele edge çiftleri (farklı satır VE farklı sütun)
    conflicts = []
    conflict_set = set()
    attempts = 0
    max_attempts = num_conflicts * 200
    while len(conflicts) < num_conflicts and attempts < max_attempts:
        attempts += 1
        e1 = rng.choice(all_edges)
        e2 = rng.choice(all_edges)
        if e1 == e2:
            continue
        if e1[0] == e2[0] or e1[1] == e2[1]:
            continue
        key = (min(e1, e2), max(e1, e2))
        if key in conflict_set:
            continue
        conflict_set.add(key)
        conflicts.append([e1[0], e1[1], e2[0], e2[1]])

    instance = {
        "n": n,
        "cost_matrix": cost_matrix,
        "conflicts": conflicts,
        "seed": seed,
    }
    return instance


# ═══════════════════════════════════════════════════════════════
#  HUNGARIAN (MAX WEIGHT ASSIGNMENT)
# ═══════════════════════════════════════════════════════════════

def hungarian_max(profit_matrix):
    """
    Scipy'nin linear_sum_assignment'ı minimization yapar.
    Maximization için: -profit_matrix kullan.
    Returns: assignment list [(i, j), ...], total profit
    """
    n = profit_matrix.shape[0]
    row_ind, col_ind = linear_sum_assignment(-profit_matrix)
    assignment = list(zip(row_ind.tolist(), col_ind.tolist()))
    total = float(profit_matrix[row_ind, col_ind].sum())
    return assignment, total


# ═══════════════════════════════════════════════════════════════
#  REPAIR HEURISTIC
# ═══════════════════════════════════════════════════════════════

def find_violations(assignment, conflicts, n):
    """Hangi conflict'ler ihlal ediliyor?"""
    asgn_matrix = np.zeros(n * n, dtype=bool)
    for i, j in assignment:
        asgn_matrix[i * n + j] = True

    if len(conflicts) == 0:
        return []

    c = np.array(conflicts, dtype=int)
    flat_e1 = c[:, 0] * n + c[:, 1]
    flat_e2 = c[:, 2] * n + c[:, 3]
    violated = asgn_matrix[flat_e1] & asgn_matrix[flat_e2]
    return list(np.where(violated)[0])


def repair_heuristic(assignment, cost_matrix, conflicts, n, max_rounds=500):
    """
    Conflict ihlallerini greedy swap ile onarır.
    """
    cost = np.array(cost_matrix, dtype=float) if not isinstance(cost_matrix, np.ndarray) else cost_matrix
    agent_to_task = {}
    task_to_agent = {}
    for i, j in assignment:
        agent_to_task[i] = j
        task_to_agent[j] = i

    for rd in range(max_rounds):
        asgn = [(i, agent_to_task[i]) for i in range(n)]
        viols = find_violations(asgn, conflicts, n)
        if len(viols) == 0:
            break

        vidx = viols[0]
        c = conflicts[vidx]
        i1, j1, i2, j2 = int(c[0]), int(c[1]), int(c[2]), int(c[3])

        best_task = -1
        best_delta = -float("inf")
        best_swap_agent = -1

        for swap_agent in [i1, i2]:
            cur_task = agent_to_task[swap_agent]
            for new_task in range(n):
                if new_task == cur_task:
                    continue
                other_agent = task_to_agent[new_task]
                delta = (
                    cost[swap_agent, new_task]
                    + cost[other_agent, cur_task]
                    - cost[swap_agent, cur_task]
                    - cost[other_agent, new_task]
                )
                if delta > best_delta:
                    best_delta = delta
                    best_task = new_task
                    best_swap_agent = swap_agent

        if best_swap_agent >= 0 and best_task >= 0:
            swap_agent = best_swap_agent
            cur_task = agent_to_task[swap_agent]
            other_agent = task_to_agent[best_task]
            agent_to_task[swap_agent] = best_task
            agent_to_task[other_agent] = cur_task
            task_to_agent[best_task] = swap_agent
            task_to_agent[cur_task] = other_agent

    final = [(i, agent_to_task[i]) for i in range(n)]
    obj = float(sum(cost[i, j] for i, j in final))
    viols = find_violations(final, conflicts, n)
    feasible = len(viols) == 0
    return final, obj, feasible


# ═══════════════════════════════════════════════════════════════
#  SUBGRADIENT ALGORITHM (PDF Algorithm 1)
# ═══════════════════════════════════════════════════════════════

def subgradient_solve(instance, K_max=500, epsilon=1e-6):
    """
    PDF'deki Algorithm 1'in birebir implementasyonu.

    Input: instance dict {n, cost_matrix, conflicts}
    Output: LB, UB, x_LB (best feasible solution)
    """
    n = instance["n"]
    cost = np.array(instance["cost_matrix"], dtype=float)
    conflicts = instance["conflicts"]
    num_conflicts = len(conflicts)

    print(f"\n{'='*60}")
    print(f"  SUBGRADIENT ALGORITHM — n={n}, |C|={num_conflicts}")
    print(f"{'='*60}")

    # ── Initialization (Lines 1-6) ──

    # Line 1: E0 = diagonal assignment (conflict-free)
    E0 = [(i, i) for i in range(n)]

    # Line 2: z(0) = sum of profits on E0
    z0 = sum(cost[i, j] for i, j in E0)

    # Line 3: LB = z(0), x_LB = indicator of E0
    LB = z0
    x_LB = E0[:]

    # Line 4: Solve relaxation over all n^2 edges (no conflict constraints) via Hungarian
    assignment_relax, z0_relax = hungarian_max(cost)

    # Line 5: UB = z(0)_relax
    UB = z0_relax

    # Line 6: k=0, t(0)=0, pi(0)=2, lambda=0
    k = 0
    t_no_improve = 0
    pi_k = 2.0

    # Lambda multipliers: one per conflict
    lambdas = np.zeros(num_conflicts, dtype=float)

    # Conflict'leri numpy array'e çevir
    if num_conflicts > 0:
        c_arr = np.array(conflicts, dtype=int)
    else:
        c_arr = np.empty((0, 4), dtype=int)

    # Her edge'in hangi conflict'lerde yer aldığını bul (edge → conflict index listesi)
    edge_to_conflicts = defaultdict(list)
    for idx in range(num_conflicts):
        i1, j1, i2, j2 = conflicts[idx]
        edge_to_conflicts[(i1, j1)].append(idx)
        edge_to_conflicts[(i2, j2)].append(idx)

    print(f"  Initial LB (diagonal) = {LB:.1f}")
    print(f"  Initial UB (Hungarian) = {UB:.1f}")
    print(f"  Gap = {((UB - LB) / max(abs(LB), 1e-10)) * 100:.2f}%")
    print()

    # Conflict edge index'lerini numpy array olarak hazırla (vektörizasyon)
    if num_conflicts > 0:
        c_e1_flat = c_arr[:, 0] * n + c_arr[:, 1]  # edge 1 flat index
        c_e2_flat = c_arr[:, 2] * n + c_arr[:, 3]  # edge 2 flat index
    else:
        c_e1_flat = np.array([], dtype=int)
        c_e2_flat = np.array([], dtype=int)

    # ── Main Loop (Lines 7-37) ──
    for iteration in range(1, K_max + 1):
        k = iteration

        # Step 1 – Penalize conflicting edges (Line 9) — vectorized
        p_tilde = cost.copy()
        if num_conflicts > 0:
            np.add.at(p_tilde.ravel(), c_e1_flat, -lambdas)
            np.add.at(p_tilde.ravel(), c_e2_flat, -lambdas)

        # Line 10: Solve Max Weight Assignment with p_tilde via Hungarian
        x_star, z_star = hungarian_max(p_tilde)

        # Step 2 – Lagrangian upper bound (Line 11)
        Z_Lag = z_star + float(np.sum(lambdas))
        UB = min(UB, Z_Lag)

        # Assignment matrix (vectorized feasibility check)
        asgn_mat = np.zeros(n * n, dtype=bool)
        for i, j in x_star:
            asgn_mat[i * n + j] = True

        # Step 3 – Feasibility check (Lines 12-28)
        if num_conflicts > 0:
            both_selected = asgn_mat[c_e1_flat] & asgn_mat[c_e2_flat]
            has_violations = np.any(both_selected)
        else:
            has_violations = False

        if not has_violations:
            # Conflict-free — LB update
            obj = float(sum(cost[i, j] for i, j in x_star))
            if obj > LB:
                LB = obj
                x_LB = x_star
                t_no_improve = 0
            else:
                t_no_improve += 1
        else:
            # Has conflicts — repair
            x_hat, z_hat, feasible = repair_heuristic(x_star, cost, conflicts, n)
            if feasible and z_hat > LB:
                LB = z_hat
                x_LB = x_hat
                t_no_improve = 0
            else:
                t_no_improve += 1

        # Step 4 – Step-size halving (Lines 29-33)
        if t_no_improve >= 20:
            t_no_improve = 0
            pi_k = pi_k / 2.0

        # Step 5 – Subgradient update (vectorized)
        if num_conflicts > 0:
            xe = asgn_mat[c_e1_flat].astype(float)
            xf = asgn_mat[c_e2_flat].astype(float)
            s = 1.0 - xe - xf
            s_norm_sq = float(np.dot(s, s))
        else:
            s_norm_sq = 0.0

        if s_norm_sq < 1e-12:
            print(f"  Iteration {k}: Subgradient norm ~0, stopping.")
            break

        alpha = pi_k * (Z_Lag - LB) / s_norm_sq
        lambdas = np.maximum(0.0, lambdas + alpha * s)

        if pi_k < epsilon:
            print(f"  Iteration {k}: pi < epsilon, stopping.")
            break

        # Progress reporting
        if k % 50 == 0 or k <= 5:
            gap = ((UB - LB) / max(abs(LB), 1e-10)) * 100
            print(f"  Iter {k:4d}: LB={LB:.1f}  UB={UB:.1f}  Gap={gap:.2f}%  pi={pi_k:.6f}")

    gap = ((UB - LB) / max(abs(LB), 1e-10)) * 100
    print(f"\n  FINAL: LB={LB:.1f}  UB={UB:.1f}  Gap={gap:.2f}%")
    print(f"  Best feasible solution: {x_LB}")
    print(f"  Objective = {LB:.1f}")

    return LB, UB, x_LB


# ═══════════════════════════════════════════════════════════════
#  CONFLICT GRAPH BUILDER (Explicit + Natural AP Conflicts)
# ═══════════════════════════════════════════════════════════════

def build_full_conflict_graph(n, explicit_conflicts):
    """
    Tam conflict graph'ı oluşturur.

    Her edge = node: v_ij (i, j) ∈ {0..n-1} x {0..n-1}  → toplam n^2 node

    İki tür çakışma:
    1. Natural AP conflicts:
       - Aynı satır: (i,j1) ile (i,j2) çakışır ∀ j1 ≠ j2
       - Aynı sütun: (i1,j) ile (i2,j) çakışır ∀ i1 ≠ i2
       Bu, assignment probleminin doğasından kaynaklanır (her agent tam bir task'a atanmalı).

    2. Explicit conflicts:
       - Instance'dan gelen ek çakışmalar: [i1,j1,i2,j2]

    Returns: adjacency dict {node_id: set(neighbor_ids)}
    """
    # Node index: (i, j) → i * n + j
    num_nodes = n * n
    adj = defaultdict(set)

    # 1. Natural AP conflicts
    natural_count = 0
    for i in range(n):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                # Aynı satır: (i, j1) ve (i, j2) çakışır
                u = i * n + j1
                v = i * n + j2
                adj[u].add(v)
                adj[v].add(u)
                natural_count += 1

    for j in range(n):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                # Aynı sütun: (i1, j) ve (i2, j) çakışır
                u = i1 * n + j
                v = i2 * n + j
                adj[u].add(v)
                adj[v].add(u)
                natural_count += 1

    # 2. Explicit conflicts
    explicit_count = 0
    for c in explicit_conflicts:
        i1, j1, i2, j2 = c[0], c[1], c[2], c[3]
        u = i1 * n + j1
        v = i2 * n + j2
        if v not in adj[u]:  # Zaten yoksa ekle
            adj[u].add(v)
            adj[v].add(u)
            explicit_count += 1

    # Tüm node'ları dahil et (izole olanlar da)
    for node_id in range(num_nodes):
        if node_id not in adj:
            adj[node_id] = set()

    print(f"\n{'='*60}")
    print(f"  CONFLICT GRAPH")
    print(f"{'='*60}")
    print(f"  Nodes (edges in AP): {num_nodes} (n^2 = {n}x{n})")
    print(f"  Natural AP conflicts (row+col): {natural_count}")
    print(f"  Explicit conflicts (instance): {explicit_count}")
    total_edges = sum(len(v) for v in adj.values()) // 2
    print(f"  Total graph edges: {total_edges}")
    density = (2 * total_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    print(f"  Graph density: {density:.4f}")

    return adj, num_nodes


# ═══════════════════════════════════════════════════════════════
#  DSATUR GRAPH COLORING
# ═══════════════════════════════════════════════════════════════

def dsatur_coloring(adj, num_nodes, n):
    """
    DSatur (Degree of Saturation) Algoritması — heap-based O(V log V + E).

    Brezaz (1979). Saturation degree: komşulara atanmış farklı renk sayısı.
    Her adımda en yüksek saturation (eşitlikte en yüksek degree) node seçilir.
    """
    import heapq

    color_map = {}
    degree = [len(adj[v]) for v in range(num_nodes)]
    neighbor_colors = [set() for _ in range(num_nodes)]
    colored = [False] * num_nodes

    # Max-heap: (-saturation, -degree, node_id)
    # Başlangıçta sat=0, en yüksek degree olan önce
    heap = [(-0, -degree[v], v) for v in range(num_nodes)]
    heapq.heapify(heap)

    # Her node'un heap'teki güncel saturation değeri
    current_sat = [0] * num_nodes
    colored_count = 0

    while colored_count < num_nodes:
        # Lazy deletion: zaten boyanmış veya eski entry'leri atla
        while heap:
            neg_sat, neg_deg, v = heapq.heappop(heap)
            if colored[v]:
                continue
            if -neg_sat != current_sat[v]:
                continue  # Eski entry, atla
            break
        else:
            break

        # En küçük kullanılabilir rengi bul
        used = neighbor_colors[v]
        color = 0
        while color in used:
            color += 1

        color_map[v] = color
        colored[v] = True
        colored_count += 1

        # Komşuları güncelle
        for u in adj[v]:
            if not colored[u]:
                if color not in neighbor_colors[u]:
                    neighbor_colors[u].add(color)
                    current_sat[u] += 1
                    heapq.heappush(heap, (-current_sat[u], -degree[u], u))

    num_colors = max(color_map.values()) + 1 if color_map else 0
    return color_map, num_colors


# ═══════════════════════════════════════════════════════════════
#  WELSH-POWELL GRAPH COLORING (karşılaştırma için)
# ═══════════════════════════════════════════════════════════════

def welsh_powell_coloring(adj, num_nodes, n):
    """
    Welsh-Powell Algoritması.

    Node'ları degree'ye göre azalan sırada sırala.
    Sırayla her node'a mümkün olan en küçük rengi ata.

    Returns: color_map {node_id: color}, num_colors
    """
    degree = [(v, len(adj[v])) for v in range(num_nodes)]
    degree.sort(key=lambda x: -x[1])

    color_map = {}
    for v, _ in degree:
        used = set()
        for u in adj[v]:
            if u in color_map:
                used.add(color_map[u])
        color = 0
        while color in used:
            color += 1
        color_map[v] = color

    num_colors = max(color_map.values()) + 1 if color_map else 0
    return color_map, num_colors


# ═══════════════════════════════════════════════════════════════
#  VISUALIZATION & REPORTING
# ═══════════════════════════════════════════════════════════════

def print_coloring_results(color_map, num_colors, n, method_name):
    """Renklendirme sonuçlarını raporla."""
    print(f"\n{'='*60}")
    print(f"  {method_name} — Renk sayısı: {num_colors}")
    print(f"{'='*60}")

    # Her renkteki node sayısını hesapla
    groups = defaultdict(int)
    for node_id, color in color_map.items():
        groups[color] += 1

    # Renk dağılımı özeti
    sizes = sorted(groups.values(), reverse=True)
    print(f"  Grup boyutları (büyükten küçüğe):")
    if len(sizes) <= 20:
        print(f"    {sizes}")
    else:
        print(f"    İlk 10: {sizes[:10]}")
        print(f"    Son  5: {sizes[-5:]}")
        print(f"    Min={min(sizes)}, Max={max(sizes)}, Avg={sum(sizes)/len(sizes):.1f}")
    print()


def demonstrate_conflict_example(n, explicit_conflicts):
    """
    Kullanıcının istediği örneği göster:
    v11 ile v22 conflict'te ise, v11 ve v22'nin tüm çakışmalarını listele.
    """
    print(f"\n{'='*60}")
    print(f"  CONFLICT DETAY ÖRNEĞİ")
    print(f"{'='*60}")

    # İlk explicit conflict'i örnek al
    if len(explicit_conflicts) > 0:
        i1, j1, i2, j2 = explicit_conflicts[0]
        print(f"\n  Explicit conflict: v{i1}{j1} ↔ v{i2}{j2}")

        # v_i1_j1'in doğal çakışmaları
        print(f"\n  v{i1}{j1}'in doğal AP çakışmaları:")
        row_conflicts = [f"v{i1}{j}" for j in range(n) if j != j1]
        col_conflicts = [f"v{i}{j1}" for i in range(n) if i != i1]
        print(f"    Aynı satır (row {i1}): {', '.join(row_conflicts)}")
        print(f"    Aynı sütun (col {j1}): {', '.join(col_conflicts)}")

        # v_i2_j2'nin doğal çakışmaları
        print(f"\n  v{i2}{j2}'nin doğal AP çakışmaları:")
        row_conflicts = [f"v{i2}{j}" for j in range(n) if j != j2]
        col_conflicts = [f"v{i}{j2}" for i in range(n) if i != i2]
        print(f"    Aynı satır (row {i2}): {', '.join(row_conflicts)}")
        print(f"    Aynı sütun (col {j2}): {', '.join(col_conflicts)}")

        print(f"\n  → Bu iki edge hem explicit conflict'le hem de (varsa)")
        print(f"    aynı satır/sütun çakışmasıyla birbirine bağlı olabilir.")
        print(f"    Conflict graph'ta tüm bu ilişkiler tek bir graf'ta birleştirilir.")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    # n ve seed komut satırından, conflict sayısı density'den otomatik hesaplanır
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    density = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # density * n^2 = explicit conflict sayısı
    # n^2 edge var, density bunların ne kadarının conflict'e karıştığını belirler
    num_conflicts = max(1, int(density * n * n))

    print(f"{'#'*60}")
    print(f"#  APC — Subgradient + Conflict Graph Coloring")
    print(f"#  n = {n}, density = {density}, |C| = {num_conflicts}")
    seed_str = "random" if seed is None else str(seed)
    print(f"#  Seed = {seed_str}")
    print(f"{'#'*60}")

    # 1. Instance üret
    print(f"\n[1] Instance üretiliyor (n={n})...")
    instance = generate_instance(n=n, num_conflicts=num_conflicts, seed=seed)
    print(f"    Seed: {instance['seed']}")
    print(f"    Cost matrix: {n}x{n}")
    print(f"    Explicit conflicts: {len(instance['conflicts'])}")
    # Büyük instance'larda hepsini yazdırma, ilk 10 + son 5
    conflicts = instance['conflicts']
    if len(conflicts) <= 20:
        for idx, c in enumerate(conflicts):
            print(f"      C{idx}: v({c[0]},{c[1]}) ↔ v({c[2]},{c[3]})")
    else:
        for idx in range(10):
            c = conflicts[idx]
            print(f"      C{idx}: v({c[0]},{c[1]}) ↔ v({c[2]},{c[3]})")
        print(f"      ... ({len(conflicts) - 15} more) ...")
        for idx in range(len(conflicts) - 5, len(conflicts)):
            c = conflicts[idx]
            print(f"      C{idx}: v({c[0]},{c[1]}) ↔ v({c[2]},{c[3]})")

    # 2. Subgradient çöz
    k_max = min(500, max(100, 1000 // n))  # Büyük n için daha az iterasyon
    print(f"\n[2] Subgradient algoritması çalıştırılıyor (K_max={k_max})...")
    LB, UB, x_LB = subgradient_solve(instance, K_max=k_max, epsilon=1e-6)

    # 3. Conflict örneğini göster (ilk explicit conflict)
    if len(conflicts) > 0:
        demonstrate_conflict_example(n, conflicts)

    # 4. Full conflict graph oluştur
    print(f"\n[3] Conflict graph oluşturuluyor...")
    adj, num_nodes = build_full_conflict_graph(n, conflicts)

    # 5. DSatur ile renklendir
    print(f"\n[4] DSatur algoritması çalıştırılıyor...")
    t0 = time.time()
    dsatur_colors, dsatur_num = dsatur_coloring(adj, num_nodes, n)
    dsatur_time = time.time() - t0
    print_coloring_results(dsatur_colors, dsatur_num, n, "DSatur")
    print(f"  DSatur süresi: {dsatur_time:.2f}s")

    # 6. Welsh-Powell ile renklendir
    print(f"\n[5] Welsh-Powell algoritması çalıştırılıyor...")
    t0 = time.time()
    wp_colors, wp_num = welsh_powell_coloring(adj, num_nodes, n)
    wp_time = time.time() - t0
    print_coloring_results(wp_colors, wp_num, n, "Welsh-Powell")
    print(f"  Welsh-Powell süresi: {wp_time:.2f}s")

    # 7. Özet
    print(f"\n{'='*60}")
    print(f"  ÖZET")
    print(f"{'='*60}")
    print(f"  n = {n}")
    print(f"  Conflict graph: {num_nodes} node, ", end="")
    total_edges = sum(len(v) for v in adj.values()) // 2
    print(f"{total_edges} edge")
    print(f"  Explicit conflicts: {len(conflicts)}")
    natural = n * n * (n - 1)  # her node: (n-1) aynı satır + (n-1) aynı sütun, /2 çift sayma
    print(f"  Natural AP conflicts: {natural}")
    print()
    print(f"  Subgradient: LB={LB:.1f}  UB={UB:.1f}  Gap={((UB-LB)/max(abs(LB),1e-10))*100:.2f}%")
    print()
    print(f"  ┌─────────────────┬───────────┬──────────┐")
    print(f"  │ Algoritma       │ Renk Sayı │ Süre     │")
    print(f"  ├─────────────────┼───────────┼──────────┤")
    print(f"  │ DSatur          │ {dsatur_num:>9} │ {dsatur_time:>6.2f}s  │")
    print(f"  │ Welsh-Powell    │ {wp_num:>9} │ {wp_time:>6.2f}s  │")
    print(f"  └─────────────────┴───────────┴──────────┘")
    print()
    if dsatur_num > n:
        print(f"  → Explicit conflicts {dsatur_num - n} ek renk gerektirdi (DSatur: {dsatur_num}, teorik min: {n})")
    else:
        print(f"  → Explicit conflicts ek renk gerektirmedi (DSatur: {dsatur_num} = n = {n})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
