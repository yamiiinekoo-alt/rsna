import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.gridspec as gridspec

# =============================================================================
# 1. THE MATHEMATICAL FORMULA (PENJELASAN RUMUS)
# =============================================================================
"""
Formula:
    Final Score = (1 / 12) * sum_{i=0}^{11} AUC_i

Penjelasan Detail Rumus Matematika:
1. AUC_i (Area Under the Receiver Operating Characteristic Curve untuk target i):
   AUC mengukur probabilitas bahwa model memberikan ranking skor probabilitas
   yang lebih tinggi pada pasien dengan kelainan (positif) dibandingkan pasien
   tanpa kelainan (negatif):
       AUC_i = P(f(x_pos) > f(x_neg))
   Secara grafis:
       AUC_i = \int_{0}^{1} TPR_i(FPR_i^{-1}(t)) dt
   Di mana:
     - TPR (True Positive Rate / Sensitivity) = TP / (TP + FN)  [Sumbu Y]
     - FPR (False Positive Rate / 1 - Specificity) = FP / (FP + TN) [Sumbu X]

2. Macro-Averaged Mean:
   Nilai akhir dihitung dengan merata-ratakan nilai AUC secara adil untuk
   ke-12 jenis patologi lutut:
       Final Score = (1 / K) * sum_{i=0}^{K-1} AUC_i,  dengan K = 12.
   Setiap patologi memiliki bobot yang sama (1/12 ≈ 8.33%), sehingga model
   tidak boleh hanya pandai menebak penyakit yang umum saja, tetapi juga harus
   pandai mendeteksi penyakit langka (seperti fraktur atau baker's cyst).
"""

TARGET_COLS = [
    'ACL Injury',
    'MCL Injury',
    'Medial Meniscus Tear',
    'Lateral Meniscus Tear',
    'Medial OA (Osteoarthritis)',
    'Lateral OA',
    'Patellofemoral OA',
    'Joint Effusion',
    'Synovitis',
    'Baker\'s Cyst',
    'Bone Contusion',
    'Fracture'
]

# =============================================================================
# 2. SIMULASI DATA PREDIKSI UNTUK 12 TARGET LUTUT
# =============================================================================
np.random.seed(42)
n_samples = 600

# Prevalensi klinis realistis
prevalences = [0.25, 0.18, 0.35, 0.28, 0.40, 0.22, 0.30, 0.45, 0.20, 0.15, 0.22, 0.10]
y_true_all = []
y_score_all = []

base_discrimination_powers = [2.2, 1.9, 2.0, 1.85, 2.3, 2.1, 2.0, 2.4, 1.75, 1.8, 2.1, 2.5]

for prev, power in zip(prevalences, base_discrimination_powers):
    y_true = np.random.binomial(1, prev, size=n_samples)
    scores = np.random.normal(loc=0.3, scale=0.2, size=n_samples)
    scores[y_true == 1] += np.random.normal(loc=0.35 * power, scale=0.15, size=np.sum(y_true == 1))
    probs = 1.0 / (1.0 + np.exp(-3.5 * (scores - np.mean(scores))))
    y_true_all.append(y_true)
    y_score_all.append(probs)

# =============================================================================
# 3. PERHITUNGAN RUMUS (SOLVING THE MATH FORMULA)
# =============================================================================
auc_scores = []
roc_data = []

print("=" * 75)
print(" RSNA KNEE ABNORMALITY DETECTION: SOLUSI MATEMATIKA EVALUASI ")
print("=" * 75)
print(f"{'Target ID (i)':<15} | {'Patologi / Abnormality':<28} | {'AUC_i':<10}")
print("-" * 75)

for i, col_name in enumerate(TARGET_COLS):
    fpr, tpr, _ = roc_curve(y_true_all[i], y_score_all[i])
    val_auc = auc(fpr, tpr)
    auc_scores.append(val_auc)
    roc_data.append((fpr, tpr, val_auc))
    print(f"Target {i:<8} | {col_name:<28} | {val_auc:.4f}")

print("-" * 75)
final_score = np.mean(auc_scores)
print(f"Rumus: Final Score = (1 / 12) * sum(AUC_0 .. AUC_11)")
print(f"                    = (1 / 12) * ({sum(auc_scores):.4f})")
print(f"HASIL AKHIR (MACRO ROC AUC): {final_score:.4f}")
print("=" * 75)

# =============================================================================
# 4. VISUALISASI GRAFIK MULTI-PLOT ROC CURVE (PUBLICATION QUALITY)
# =============================================================================
fig = plt.figure(figsize=(18, 12), facecolor='#0f172a')
gs = gridspec.GridSpec(3, 4, wspace=0.3, hspace=0.45)

colors = [
    '#38bdf8', '#818cf8', '#c084fc', '#f472b6',
    '#fb7185', '#fb923c', '#facc15', '#4ade80',
    '#2dd4bf', '#22d3ee', '#a78bfa', '#f87171'
]

fig.suptitle(
    f"RSNA Knee Abnormality Detection: 12 Pathologies Multi-ROC Curves\n"
    f"Evaluated Metric: Final Score = (1/12) ∑ AUC_i = {final_score:.4f} (Macro-Averaged ROC AUC)",
    fontsize=18, fontweight='bold', color='#f8fafc', y=0.98
)

for idx, (col_name, (fpr, tpr, score), color) in enumerate(zip(TARGET_COLS, roc_data, colors)):
    ax = fig.add_subplot(gs[idx])
    ax.set_facecolor('#1e293b')

    ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {score:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.18, color=color)
    ax.plot([0, 1], [0, 1], color='#64748b', linestyle='--', lw=1.2, label="Random (0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title(f"{idx}: {col_name}", fontsize=12, fontweight='bold', color='#f1f5f9', pad=8)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=9, color='#94a3b8')
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=9, color='#94a3b8')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.grid(color='#334155', linestyle=':', linewidth=0.8, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color('#334155')

    ax.legend(loc="lower right", facecolor='#0f172a', edgecolor='#334155', fontsize=8, labelcolor='#e2e8f0')

output_img = "rsna_knee_roc_curves.png"
plt.savefig(output_img, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"\n[INFO] Gambar grafik multi-ROC berhasil disimpan sebagai: {output_img}")

# =============================================================================
# 5. GRAFIK REKAPITULASI KONTRIBUSI BOBOT PER TARGET
# =============================================================================
fig_bar, ax_bar = plt.subplots(figsize=(14, 6), facecolor='#0f172a')
ax_bar.set_facecolor('#1e293b')

y_pos = np.arange(len(TARGET_COLS))
bars = ax_bar.barh(y_pos, auc_scores, color=colors, edgecolor='#0f172a', height=0.65)

ax_bar.axvline(final_score, color='#facc15', linestyle='--', lw=2, label=f'Macro Average (Final Score) = {final_score:.4f}')

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(TARGET_COLS, color='#f8fafc', fontsize=10, fontweight='bold')
ax_bar.set_xlabel('Area Under ROC Curve (AUC)', color='#94a3b8', fontsize=11)
ax_bar.set_title('AUC Comparison Across All 12 Knee Abnormalities', color='#f8fafc', fontsize=15, fontweight='bold', pad=15)
ax_bar.set_xlim([0.0, 1.05])
ax_bar.tick_params(colors='#94a3b8')
ax_bar.grid(color='#334155', linestyle=':', linewidth=0.8, axis='x')

for spine in ax_bar.spines.values():
    spine.set_color('#334155')

for bar in bars:
    w = bar.get_width()
    ax_bar.text(w + 0.01, bar.get_y() + bar.get_height()/2, f'{w:.4f}',
                va='center', ha='left', color='#e2e8f0', fontsize=9, fontweight='bold')

ax_bar.legend(loc='lower left', facecolor='#0f172a', edgecolor='#334155', labelcolor='#facc15', fontsize=10)
plt.tight_layout()

bar_img = "rsna_knee_auc_barchart.png"
plt.savefig(bar_img, dpi=200, bbox_inches='tight', facecolor=fig_bar.get_facecolor())
print(f"[INFO] Gambar barchart berhasil disimpan sebagai: {bar_img}")
