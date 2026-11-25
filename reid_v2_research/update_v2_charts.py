import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
OUT_ROOT = os.path.join(BUNDLE_ROOT, "runs_workstation", "reid_v2_research")
SUM_PATH = os.path.join(OUT_ROOT, "v2_reid_summary.json")


def main():
    with open(SUM_PATH, "r", encoding="utf-8") as f:
        sdata = json.load(f)

    r = sdata["results"]
    v0 = r["v0_baseline"]
    v1 = r["v1_p0_full"]
    zsm = r["zs_megadescriptor_t"]
    zsd = r["zs_dinov2_s"]
    exa = r["exp_a_mega_t"]
    exb = r["exp_b_dinov2_s"]
    exc = r["exp_c_frozen_dino_adapter"]

    fp_b = sdata["fp16_export"]["dinov2_s"]
    fp_a = sdata["fp16_export"]["mega_t"]
    fp_c = sdata["fp16_export"]["frozen_adapter"]

    # Update report with all 7 variants
    lines = [
        "=" * 116,
        "Track 1 (v2) Foundation + GeM + SubCenter-AdaArcFace + Anchor-KD Re-ID Full Report (incl. Exp C)",
        "=" * 116,
        f"Primary v2 Winner (Occlusion & Robustness): v2 DINOv2-S* (Exp B) | FP16={fp_b['fp16_mb']:.2f}MB ({fp_b['latency_fp16_ms']:.2f}ms)",
        f"Clean Cross-Domain Runner-Up: v2 Frozen-DINOv2+Adapter (Exp C) | FP16={fp_c['fp16_mb']:.2f}MB ({fp_c['latency_fp16_ms']:.2f}ms)",
        f"MegaDescriptor-T* (Exp A): FP16={fp_a['fp16_mb']:.2f}MB ({fp_a['latency_fp16_ms']:.2f}ms)",
        "",
        f"{'Metric':<34} | {'v0 Base':>8} | {'v1 P0':>8} | {'ZS Mega':>8} | {'ZS DINO':>8} | {'ExpA Mg':>8} | {'ExpB DN':>8} | {'ExpC Fr':>8}",
        "-" * 114,
        f"{'Closed Clean Acc':<34} | {v0['closed_clean']['acc']*100:>7.2f}% | {v1['closed_clean']['acc']*100:>7.2f}% | {zsm['closed_clean']['acc']*100:>7.2f}% | {zsd['closed_clean']['acc']*100:>7.2f}% | {exa['closed_clean']['acc']*100:>7.2f}% | {exb['closed_clean']['acc']*100:>7.2f}% | {exc['closed_clean']['acc']*100:>7.2f}%",
        f"{'Closed Unseen-Fence Acc':<34} | {v0['closed_occluded']['acc']*100:>7.2f}% | {v1['closed_occluded']['acc']*100:>7.2f}% | {zsm['closed_occluded']['acc']*100:>7.2f}% | {zsd['closed_occluded']['acc']*100:>7.2f}% | {exa['closed_occluded']['acc']*100:>7.2f}% | {exb['closed_occluded']['acc']*100:>7.2f}% | {exc['closed_occluded']['acc']*100:>7.2f}%",
        f"{'SJ Clean Top-5 Rank-1':<34} | {v0['shaojiang']['clean']['rank1']*100:>7.2f}% | {v1['shaojiang']['clean']['rank1']*100:>7.2f}% | {zsm['shaojiang']['clean']['rank1']*100:>7.2f}% | {zsd['shaojiang']['clean']['rank1']*100:>7.2f}% | {exa['shaojiang']['clean']['rank1']*100:>7.2f}% | {exb['shaojiang']['clean']['rank1']*100:>7.2f}% | {exc['shaojiang']['clean']['rank1']*100:>7.2f}%",
        f"{'SJ Clean ROC-AUC':<34} | {v0['shaojiang']['clean']['auc']:>8.4f} | {v1['shaojiang']['clean']['auc']:>8.4f} | {zsm['shaojiang']['clean']['auc']:>8.4f} | {zsd['shaojiang']['clean']['auc']:>8.4f} | {exa['shaojiang']['clean']['auc']:>8.4f} | {exb['shaojiang']['clean']['auc']:>8.4f} | {exc['shaojiang']['clean']['auc']:>8.4f}",
        f"{'SJ Clean EER (lower=better)':<34} | {v0['shaojiang']['clean']['eer']*100:>7.2f}% | {v1['shaojiang']['clean']['eer']*100:>7.2f}% | {zsm['shaojiang']['clean']['eer']*100:>7.2f}% | {zsd['shaojiang']['clean']['eer']*100:>7.2f}% | {exa['shaojiang']['clean']['eer']*100:>7.2f}% | {exb['shaojiang']['clean']['eer']*100:>7.2f}% | {exc['shaojiang']['clean']['eer']*100:>7.2f}%",
        f"{'SJ Clean Cos Gap Delta':<34} | {v0['shaojiang']['clean']['gap']:>+8.4f} | {v1['shaojiang']['clean']['gap']:>+8.4f} | {zsm['shaojiang']['clean']['gap']:>+8.4f} | {zsd['shaojiang']['clean']['gap']:>+8.4f} | {exa['shaojiang']['clean']['gap']:>+8.4f} | {exb['shaojiang']['clean']['gap']:>+8.4f} | {exc['shaojiang']['clean']['gap']:>+8.4f}",
        f"{'SJ Occluded Top-5 Rank-1':<34} | {v0['shaojiang']['occluded']['rank1']*100:>7.2f}% | {v1['shaojiang']['occluded']['rank1']*100:>7.2f}% | {zsm['shaojiang']['occluded']['rank1']*100:>7.2f}% | {zsd['shaojiang']['occluded']['rank1']*100:>7.2f}% | {exa['shaojiang']['occluded']['rank1']*100:>7.2f}% | {exb['shaojiang']['occluded']['rank1']*100:>7.2f}% | {exc['shaojiang']['occluded']['rank1']*100:>7.2f}%",
        f"{'SJ Occluded ROC-AUC':<34} | {v0['shaojiang']['occluded']['auc']:>8.4f} | {v1['shaojiang']['occluded']['auc']:>8.4f} | {zsm['shaojiang']['occluded']['auc']:>8.4f} | {zsd['shaojiang']['occluded']['auc']:>8.4f} | {exa['shaojiang']['occluded']['auc']:>8.4f} | {exb['shaojiang']['occluded']['auc']:>8.4f} | {exc['shaojiang']['occluded']['auc']:>8.4f}",
        f"{'SJ Occluded EER (lower=better)':<34} | {v0['shaojiang']['occluded']['eer']*100:>7.2f}% | {v1['shaojiang']['occluded']['eer']*100:>7.2f}% | {zsm['shaojiang']['occluded']['eer']*100:>7.2f}% | {zsd['shaojiang']['occluded']['eer']*100:>7.2f}% | {exa['shaojiang']['occluded']['eer']*100:>7.2f}% | {exb['shaojiang']['occluded']['eer']*100:>7.2f}% | {exc['shaojiang']['occluded']['eer']*100:>7.2f}%",
        f"{'Hard Probe 295-6 Fence Prob':<34} | {v0['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>7.2f}% | {v1['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>7.2f}% |      n/a |      n/a | {exa['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>7.2f}% | {exb['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>7.2f}% | {exc['hard_probe']['295_6_vertical_fence']['prob_295_6']*100:>7.2f}%",
        f"{'Hard Probe pig_16 Rails Margin':<34} | {v0['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f} | {v1['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f} | {zsm['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f} | {zsd['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f} | {exa['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f} | {exb['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f} | {exc['hard_probe']['pig16_angled_rails']['margin_over_next']:>+8.4f}",
        "",
        "Max-Cosine Unknown Rejection Sweep (Exp B Winner: v2 DINOv2-S*):",
    ]
    for sw in exb["local_unknown"]["cos_sweep"]:
        lines.append(
            f"  cos >= {sw['cos_thresh']:.2f} -> Known Accept={sw['known_accept']*100:5.2f}% | "
            f"LocalUnk TRR={sw['local_unk_trr']*100:5.2f}% | SJ Unk TRR={sw['sj_unk_trr']*100:5.2f}%"
        )
    lines.append("Max-Cosine Unknown Rejection Sweep (Exp C Frozen-DINOv2+Adapter):")
    for sw in exc["local_unknown"]["cos_sweep"]:
        lines.append(
            f"  cos >= {sw['cos_thresh']:.2f} -> Known Accept={sw['known_accept']*100:5.2f}% | "
            f"LocalUnk TRR={sw['local_unk_trr']*100:5.2f}% | SJ Unk TRR={sw['sj_unk_trr']*100:5.2f}%"
        )
    lines.append("=" * 116)
    rep_txt = "\n".join(lines)
    with open(os.path.join(OUT_ROOT, "v2_reid_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#444", "axes.linewidth": 0.9})

    # Fig 1: All 7 variants
    methods = [
        ("v0 ResNet18", v0),
        ("v1 P0-Full", v1),
        ("ZS Mega-T", zsm),
        ("ZS DINOv2-S", zsd),
        ("ExpA Mega-T*", exa),
        ("ExpB DINOv2*", exb),
        ("ExpC Frz+Adp", exc),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8), dpi=180)
    x = np.arange(len(methods))
    w = 0.36

    r1_c = [m[1]["shaojiang"]["clean"]["rank1"] * 100 for m in methods]
    r1_o = [m[1]["shaojiang"]["occluded"]["rank1"] * 100 for m in methods]
    b1 = axes[0].bar(x - w / 2, r1_c, width=w, label="SJ Clean Rank-1", color="#2b6cb0")
    b2 = axes[0].bar(x + w / 2, r1_o, width=w, label="SJ Occ Rank-1", color="#38a169")
    for b in list(b1) + list(b2):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.0)
    axes[0].set_ylim(0, 108)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m[0] for m in methods], rotation=22, ha="right", fontsize=8.2)
    axes[0].set_ylabel("Rank-1 Accuracy (%)")
    axes[0].set_title("(a) External Unseen Pigs Rank-1 (Clean & Occ)", fontweight="bold")
    axes[0].legend(fontsize=8.2)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    auc_c = [m[1]["shaojiang"]["clean"]["auc"] * 100 for m in methods]
    auc_o = [m[1]["shaojiang"]["occluded"]["auc"] * 100 for m in methods]
    b1 = axes[1].bar(x - w / 2, auc_c, width=w, label="SJ Clean AUC", color="#4c51bf")
    b2 = axes[1].bar(x + w / 2, auc_o, width=w, label="SJ Occ AUC", color="#dd6b20")
    for b in list(b1) + list(b2):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.0)
    axes[1].set_ylim(45, 108)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m[0] for m in methods], rotation=22, ha="right", fontsize=8.2)
    axes[1].set_ylabel("ROC-AUC (%)")
    axes[1].set_title("(b) External Open-Set Verification ROC-AUC", fontweight="bold")
    axes[1].legend(fontsize=8.2)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    trained = [
        ("v0 ResNet18", v0),
        ("v1 P0-Full", v1),
        ("ExpA Mega-T*", exa),
        ("ExpB DINOv2*", exb),
        ("ExpC Frz+Adp", exc),
    ]
    xt = np.arange(len(trained))
    c_occ = [m[1]["closed_occluded"]["acc"] * 100 for m in trained]
    eer_c = [m[1]["shaojiang"]["clean"]["eer"] * 100 for m in trained]
    eer_o = [m[1]["shaojiang"]["occluded"]["eer"] * 100 for m in trained]
    w3 = 0.25
    b1 = axes[2].bar(xt - w3, c_occ, width=w3, label="Closed Occ Acc (High=Best)", color="#16a34a")
    b2 = axes[2].bar(xt, eer_c, width=w3, label="SJ Clean EER (Low=Best)", color="#e53e3e")
    b3 = axes[2].bar(xt + w3, eer_o, width=w3, label="SJ Occ EER (Low=Best)", color="#ed8936")
    for b in list(b1) + list(b2) + list(b3):
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7.0)
    axes[2].set_ylim(0, 112)
    axes[2].set_xticks(xt)
    axes[2].set_xticklabels([m[0] for m in trained], rotation=18, ha="right", fontsize=8.2)
    axes[2].set_ylabel("Percentage (%)")
    axes[2].set_title("(c) Closed Occluded Acc vs Open-Set EER", fontweight="bold")
    axes[2].legend(fontsize=7.8)
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("Track 1 (v2 Re-ID): Foundation Backbones + GeM + SubCenter AdaArcFace + Anchor-KD (7 Variants)", fontsize=13.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v2_reid_fig1_method_comparison.png"), bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Cosine Histograms & ROC Curves including Exp C
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.2), dpi=180)
    bins = np.linspace(-0.1, 1.0, 34)
    for ax, title, d_sj in [
        (axes[0, 0], f"v1 P0-Full Clean Cosine (Gap={v1['shaojiang']['clean']['gap']:+.3f})", v1["shaojiang"]["clean"]),
        (axes[0, 1], f"Exp B v2 DINOv2-S* Occ Cosine (Gap={exb['shaojiang']['occluded']['gap']:+.3f})", exb["shaojiang"]["occluded"]),
    ]:
        ax.hist(d_sj["inter_vals"], bins=bins, density=True, alpha=0.55, color="#e53e3e", label=f"Diff-Pig ({d_sj['inter_mean']:.3f})")
        ax.hist(d_sj["intra_vals"], bins=bins, density=True, alpha=0.60, color="#16a34a", label=f"Same-Pig ({d_sj['intra_mean']:.3f})")
        ax.axvline(d_sj["youden_thresh"], color="black", linestyle="--", lw=1.3, label=f"Youden {d_sj['youden_thresh']:.2f}")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.3, linestyle="--")

    for ax, split_k, stitle in [
        (axes[1, 0], "clean", "External Shaojiang Clean ROC Curves"),
        (axes[1, 1], "occluded", "External Shaojiang Unseen-Fence Occluded ROC Curves"),
    ]:
        for lbl, res_d, col, ls in [
            ("v0 Baseline", v0, "#718096", ":"),
            ("v1 P0-Full", v1, "#3182ce", "--"),
            ("ZS DINOv2-S", zsd, "#805ad5", "-."),
            ("Exp A Mega-T*", exa, "#dd6b20", "-."),
            ("Exp B DINOv2-S*", exb, "#16a34a", "-"),
            ("Exp C Frozen+Adp", exc, "#0d9488", "-"),
        ]:
            d = res_d["shaojiang"][split_k]
            ax.plot(d["fpr_curve"], d["tpr_curve"], color=col, linestyle=ls, lw=2.0, label=f"{lbl} (AUC={d['auc']:.3f}, EER={d['eer']*100:.1f}%)")
        ax.plot([0, 1], [0, 1], "k:", alpha=0.4)
        ax.set_title(stitle, fontweight="bold")
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.legend(fontsize=7.8, loc="lower right")
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Cosine Separation & Open-Set ROC Diagnostics (Exp B DINOv2-S* & Exp C Frozen+Adapter)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_ROOT, "v2_reid_fig2_cosine_and_roc.png"), bbox_inches="tight")
    plt.close(fig)
    print(rep_txt)


if __name__ == "__main__":
    main()
