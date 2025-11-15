#!/usr/bin/env python3
# compare_runs.py
# Creates beautiful comparison table of all landmark model versions

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("LANDMARK MODEL COMPARISON")
print("="*70)

# -------------------
# LOAD RESULTS
# -------------------
results = []
versions = []

# Try to load each version
files = [
    ('landmark_v1_results/v1_metrics.csv', 'LM-V1'),
    ('landmark_v2_results/v2_metrics.csv', 'LM-V2'),
    ('landmark_v3_results/v3_metrics.csv', 'LM-V3')
]

for filepath, version_name in files:
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        results.append(df)
        versions.append(version_name)
        print(f"✅ Loaded {version_name} from {filepath}")
    else:
        print(f"⚠️  {version_name} not found at {filepath}")

if not results:
    print("\n❌ No results found! Run the models first:")
    print("  python3 landmark_model_v1.py")
    print("  python3 landmark_model_v2.py")
    print("  python3 landmark_model_v3.py")
    exit(1)

# Combine all results
comparison = pd.concat(results, ignore_index=True)

# -------------------
# CREATE FORMATTED TABLE
# -------------------
print("\n" + "="*70)
print("PERFORMANCE COMPARISON TABLE")
print("="*70)

# Select and order columns
display_cols = ['model', 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity']
display_df = comparison[display_cols].copy()

# Format as percentages for display
display_df['accuracy'] = (display_df['accuracy'] * 100).round(2)
display_df['precision'] = (display_df['precision'] * 100).round(2)
display_df['recall'] = (display_df['recall'] * 100).round(2)
display_df['f1_score'] = (display_df['f1_score'] * 100).round(2)
display_df['auc'] = display_df['auc'].round(4)
display_df['specificity'] = (display_df['specificity'] * 100).round(2)

# Rename columns for display
display_df.columns = ['Model', 'Accuracy (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'AUC', 'Specificity (%)']

# Print beautiful table
print(display_df.to_string(index=False))
print("="*70)

# -------------------
# SAVE RESULTS
# -------------------
# Save raw comparison
comparison.to_csv('landmark_models_comparison.csv', index=False)
print(f"\n✅ Raw metrics saved to: landmark_models_comparison.csv")

# Save formatted comparison
display_df.to_csv('landmark_models_comparison_formatted.csv', index=False)
print(f"✅ Formatted table saved to: landmark_models_comparison_formatted.csv")

# -------------------
# STATISTICAL SUMMARY
# -------------------
print("\n" + "="*70)
print("STATISTICAL SUMMARY")
print("="*70)

# Best model for each metric
metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']

print("\n🏆 Best Performance by Metric:")
for metric, name in zip(metrics, metric_names):
    best_idx = comparison[metric].idxmax()
    best_model = comparison.loc[best_idx, 'model']
    best_value = comparison.loc[best_idx, metric]
    print(f"  {name:15s}: {best_model} ({best_value:.4f})")

# Performance improvements
if len(results) > 1:
    print("\n📈 Improvements from V1 to V3:")
    v1_metrics = comparison.iloc[0]
    v3_metrics = comparison.iloc[-1]
    
    for metric, name in zip(metrics, metric_names):
        v1_val = v1_metrics[metric]
        v3_val = v3_metrics[metric]
        diff = v3_val - v1_val
        pct_change = (diff / v1_val * 100) if v1_val != 0 else 0
        arrow = "↗️" if diff > 0 else ("↘️" if diff < 0 else "→")
        print(f"  {name:15s}: {v1_val:.4f} → {v3_val:.4f} ({arrow} {pct_change:+.2f}%)")

# -------------------
# CONFUSION MATRICES
# -------------------
print("\n" + "="*70)
print("CONFUSION MATRICES")
print("="*70)

for i, row in comparison.iterrows():
    model_name = row['model']
    tn, fp, fn, tp = int(row['tn']), int(row['fp']), int(row['fn']), int(row['tp'])
    
    print(f"\n{model_name}:")
    print(f"                 Predicted")
    print(f"             Not Dep  Depressed")
    print(f"Actual Not   {tn:4d}     {fp:4d}")
    print(f"Actual Dep   {fn:4d}     {tp:4d}")
    print(f"  Accuracy: {row['accuracy']:.4f}, Recall: {row['recall']:.4f}, Precision: {row['precision']:.4f}")

# -------------------
# VISUALIZATIONS
# -------------------
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

plot_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'specificity']
plot_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for idx, (metric, name, color) in enumerate(zip(plot_metrics, plot_names, colors)):
    ax = axes[idx]
    
    models = comparison['model'].values
    values = comparison[metric].values
    
    bars = ax.bar(models, values, color=color, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(name, fontsize=11, fontweight='bold')
    ax.set_title(f'{name} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0, max(1.0, max(values) * 1.1)])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Random')
    
    # Rotate x labels if needed
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

plt.suptitle('Landmark Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('landmark_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Visualization saved to: landmark_models_comparison.png")

# -------------------
# CONFUSION MATRIX HEATMAPS
# -------------------
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))

if len(results) == 1:
    axes = [axes]

for idx, (ax, row) in enumerate(zip(axes, comparison.iterrows())):
    row = row[1]
    model_name = row['model']
    
    cm = np.array([
        [row['tn'], row['fp']],
        [row['fn'], row['tp']]
    ])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Not Dep', 'Depressed'],
               yticklabels=['Not Dep', 'Depressed'],
               ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'{model_name}\nAcc: {row["accuracy"]:.3f}, AUC: {row["auc"]:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)

plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Confusion matrices saved to: confusion_matrices_comparison.png")

# -------------------
# LATEX TABLE
# -------------------
print("\n" + "="*70)
print("LATEX TABLE (Copy-paste ready)")
print("="*70)

print("\n% Copy this into your LaTeX document:\n")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Landmark Model Performance Comparison}")
print("\\begin{tabular}{|l|c|c|c|c|c|c|}")
print("\\hline")
print("\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{AUC} & \\textbf{Specificity} \\\\")
print("\\hline")

for _, row in comparison.iterrows():
    print(f"{row['model']} & {row['accuracy']:.4f} & {row['precision']:.4f} & {row['recall']:.4f} & {row['f1_score']:.4f} & {row['auc']:.4f} & {row['specificity']:.4f} \\\\")

print("\\hline")
print("\\end{tabular}")
print("\\label{tab:landmark_comparison}")
print("\\end{table}")

# -------------------
# MARKDOWN TABLE
# -------------------
print("\n" + "="*70)
print("MARKDOWN TABLE (Copy-paste ready)")
print("="*70)

print("\n| Model | Accuracy | Precision | Recall | F1-Score | AUC | Specificity |")
print("|-------|----------|-----------|--------|----------|-----|-------------|")
for _, row in comparison.iterrows():
    print(f"| {row['model']} | {row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['auc']:.4f} | {row['specificity']:.4f} |")

print("\n" + "="*70)
print("COMPARISON COMPLETE! ✅")
print("="*70)
print("\nGenerated files:")
print("  📄 landmark_models_comparison.csv (raw data)")
print("  📄 landmark_models_comparison_formatted.csv (formatted table)")
print("  📊 landmark_models_comparison.png (metrics visualization)")
print("  📊 confusion_matrices_comparison.png (confusion matrices)")
print("="*70)