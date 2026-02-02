import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import numpy as np

def plot_cd_heatmap_plt(results_list):
    if not results_list:
        print("no data to plot")
        return

    df = pd.DataFrame(results_list)
    
    pivot_df = df.pivot(index="Initial_Cd", columns="Duration", values="Estimated_Cd")
    
    durations = pivot_df.columns.values
    guesses = pivot_df.index.values
    data = pivot_df.values

    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(data, aspect='auto', origin='lower', cmap='viridis',
                   extent=[durations.min()-0.5, durations.max()+0.5, 
                           guesses.min()-0.15, guesses.max()+0.15])

    for i in range(len(guesses)):
        for j in range(len(durations)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(durations[j], guesses[i], f'{val:.2f}', 
                        ha="center", va="center", color="w", fontsize=8)

    plt.colorbar(im, label='Estimated $C_d$')
    ax.set_xticks(durations)
    ax.set_yticks(guesses)
    
    ax.set_title('Estimated $C_d$ vs Arc Duration & Initial Guess')
    ax.set_xlabel('Arc Duration (Hours)')
    ax.set_ylabel('Initial $C_d$ Guess')

    plt.tight_layout()
    plt.savefig('cd_heatmap_matplotlib.png', dpi=300)
    plt.show()


def plot_combined_heatmaps(results_list):
    if not results_list:
        print("no data to plot_")
        return

    df = pd.DataFrame(results_list)
    
    pivot_cd = df.pivot(index="Initial_Cd", columns="Duration", values="Estimated_Cd")
    pivot_res = df.pivot(index="Initial_Cd", columns="Duration", values="Residual")
    
    durations = pivot_cd.columns.values
    guesses = pivot_cd.index.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    im1 = ax1.imshow(pivot_cd.values, aspect='auto', origin='lower', cmap='magma',
                    extent=[durations.min()-1.5, durations.max()+1.5, guesses.min()-0.15, guesses.max()+0.15])
    ax1.set_title('Estimated $C_d$')
    ax1.set_ylabel('Initial $C_d$ Guess')
    ax1.set_xlabel('Arc Duration (Hours)')
    fig.colorbar(im1, ax=ax1, label='Cd Value')

    im2 = ax2.imshow(pivot_res.values, aspect='auto', origin='lower', cmap='viridis',
                    norm=colors.LogNorm(vmin=pivot_res.min().min(), vmax=pivot_res.max().max()),
                    extent=[durations.min()-1.5, durations.max()+1.5, guesses.min()-0.15, guesses.max()+0.15])
    ax2.set_title('Final RMS Residual (Log Scale)')
    ax2.set_xlabel('Arc Duration (Hours)')
    fig.colorbar(im2, ax=ax2, label='RMS Residual (m)')

    for i in range(len(guesses)):
        for j in range(len(durations)):
            ax1.text(durations[j], guesses[i], f'{pivot_cd.values[i,j]:.2f}', ha="center", va="center", color="w", fontsize=8)
            res_val = pivot_res.values[i,j]
            if not np.isnan(res_val):
                ax2.text(durations[j], guesses[i], f'{res_val:.1e}', ha="center", va="center", color="w", fontsize=7)

    plt.tight_layout()
    plt.savefig('cd_and_residual_study.png', dpi=300)
    plt.show()