from scipy.optimize import linear_sum_assignment
import numpy as np
import random

# Parametreler
N = 3
MIN_PROFIT = 1
MAX_PROFIT = 100
MAX_ITER = 100  # <-- İterasyon limiti buraya eklendi

# 1. Rastgele Kâr Matrisi Oluşturma
P = np.random.randint(MIN_PROFIT, MAX_PROFIT + 1, size=(N, N))

# 2. Dinamik ve Anlamlı Çatışma (Conflict) Üretimi
max_conflicts = ( (N**2) * ((N-1)**2) ) // 2
num_conflicts = random.randint(1, max_conflicts)
conflicts_set = set()

while len(conflicts_set) < num_conflicts:
    r1, r2 = random.sample(range(N), 2)
    c1, c2 = random.sample(range(N), 2)
    conflict_pair = tuple(sorted(((r1, c1), (r2, c2))))
    conflicts_set.add(conflict_pair)

conflicts = list(conflicts_set)

print(f"--- RASTGELE ÜRETİLEN VERİLER (N={N}) ---")
print("Kâr Matrisi (P):")
# Matris çok büyük olduğu için numpy otomatik özetleyerek yazdırır
print(P)
print(f"\nÜretilen Çatışma Sayısı: {num_conflicts} (Maksimum sınır: {max_conflicts})")

# Konsolu kilitlememek için sadece ilk 10 çatışmayı yazdırıyoruz
for idx, (cell1, cell2) in enumerate(conflicts[:10]):
    print(f"Çatışma {idx + 1}: Hücre 1: {cell1} ve Hücre 2: {cell2} aynı anda atanamaz.")
if num_conflicts > 10:
    print(f"... ve {num_conflicts - 10} adet daha çatışma gizlendi.")

def solve_assignment(profit_matrix):
    row_ind, col_ind = linear_sum_assignment(-profit_matrix)
    x = np.zeros_like(profit_matrix)
    x[row_ind, col_ind] = 1
    return x

def solve_subproblem(P_matrix, keep_cell, forbid_cell):
    P_sub = P_matrix.copy().astype(float)
    r_keep, c_keep = keep_cell
    for j in range(P_sub.shape[1]):
        if j != c_keep: P_sub[r_keep, j] = -1e9
    for i in range(P_sub.shape[0]):
        if i != r_keep: P_sub[i, c_keep] = -1e9

    r_forbid, c_forbid = forbid_cell
    P_sub[r_forbid, c_forbid] = -1e9

    x_sub = solve_assignment(P_sub)
    profit_sub = np.sum(P_matrix * x_sub)
    return x_sub, profit_sub

# max_iterations parametresi fonksiyona eklendi
def solve_apc_lagrangean(P_matrix, conflict_list, max_iterations):
    # Şimdilik sadece ilk çatışmayı (conflict_list[0]) dikkate alıyoruz
    c1, c2 = conflict_list[0]

    lambda_val = 0.0
    pi = 2.0

    best_lb = -np.inf
    best_ub = np.inf
    best_feasible_x = None

    # ==========================================
    # İLK AŞAMA (WARM START): Başlangıç Çözümü
    # ==========================================
    print("\n[Aşama 1] Başlangıç Çözümü (Hungarian) Aranıyor...")
    initial_x = solve_assignment(P_matrix)

    if initial_x[c1] + initial_x[c2] <= 1:
        best_lb = np.sum(P_matrix * initial_x)
        best_feasible_x = initial_x.copy()
        print(f"  -> Çatışma yok. Başlangıç Alt Sınırı (LB): {best_lb}")
    else:
        print("  -> Çatışma tespit edildi. Onarım (Repair) uygulanıyor...")
        x_rep_A, profit_rep_A = solve_subproblem(P_matrix, keep_cell=c1, forbid_cell=c2)
        x_rep_B, profit_rep_B = solve_subproblem(P_matrix, keep_cell=c2, forbid_cell=c1)

        if profit_rep_A > profit_rep_B:
            best_lb = profit_rep_A
            best_feasible_x = x_rep_A.copy()
        else:
            best_lb = profit_rep_B
            best_feasible_x = x_rep_B.copy()
        print(f"  -> Onarım başarılı. Başlangıç Alt Sınırı (LB): {best_lb}")
    # ==========================================

    print("\n[Aşama 2] Lagrangean Relaxation & Subgradient Başlıyor...")
    # Döngü artık dışarıdan gelen max_iterations parametresine bağlı
    for iteration in range(max_iterations):
        P_prime = P_matrix.copy().astype(float)
        P_prime[c1] -= lambda_val
        P_prime[c2] -= lambda_val

        x_relaxed = solve_assignment(P_prime)
        relaxed_profit = np.sum(P_prime * x_relaxed)
        Z_lambda = relaxed_profit + lambda_val
        best_ub = min(best_ub, Z_lambda)

        if x_relaxed[c1] + x_relaxed[c2] <= 1:
            true_profit = np.sum(P_matrix * x_relaxed)
            if true_profit > best_lb:
                best_lb = true_profit
                best_feasible_x = x_relaxed.copy()
        else:
            x_rep_A, profit_rep_A = solve_subproblem(P_matrix, keep_cell=c1, forbid_cell=c2)
            x_rep_B, profit_rep_B = solve_subproblem(P_matrix, keep_cell=c2, forbid_cell=c1)

            if profit_rep_A > best_lb:
                best_lb = profit_rep_A
                best_feasible_x = x_rep_A.copy()

            if profit_rep_B > best_lb:
                best_lb = profit_rep_B
                best_feasible_x = x_rep_B.copy()

        g = 1 - (x_relaxed[c1] + x_relaxed[c2])
        if g == 0 and x_relaxed[c1] + x_relaxed[c2] <= 1:
            break

        if best_lb != -np.inf and g != 0:
            step_size = pi * (Z_lambda - best_lb) / (g ** 2)
            lambda_val = max(0.0, lambda_val - step_size * g)

        pi *= 0.95

    return best_lb, best_ub, best_feasible_x

# Kodu Çalıştırma (MAX_ITER argüman olarak gönderiliyor)
lb, ub, optimal_x = solve_apc_lagrangean(P, conflicts, MAX_ITER)

print("\n--- NİHAİ SONUÇLAR ---")
print(f"Hafızada Tutulan En İyi Geçerli Kâr (Best LB): {lb}")
print(f"En İyi Gevşetilmiş Üst Sınır (Best UB): {ub:.2f}")
# Matris çok büyük olduğu için x matrisini tamamen yazdırmak yerine sadece özet veriyoruz
print("Optimal Atama Matrisi (Çatışmasız En İyi Alternatif) Bulundu.")
