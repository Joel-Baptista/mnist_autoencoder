import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Set a Seaborn theme for aesthetics
sns.set_theme(style="whitegrid")

# Folder containing your CSV files
folder_path = 'results'

# Iterate through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Read the CSV file
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Automatically use the first column as the x-axis
        x_column = df.columns[0]

        
        # Melt the DataFrame to long-form for seaborn
        df_melted = pd.melt(df, id_vars=[x_column], var_name='Variable', value_name='Value')
        
        # Plotting using seaborn with enhanced styling
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=x_column, y='Value', hue='Variable', data=df_melted, marker='', palette='tab10')

        # Customize the title and labels
        plt.title(f'{filename}', fontsize=16, weight='bold', color='darkblue', pad=20)
        plt.xlabel(x_column, fontsize=14, weight='bold', color='darkblue')
        plt.ylabel('Values', fontsize=14, weight='bold', color='darkblue')

        plt.ylim(0, 0.8)  # Set the y-axis from 0 to 0.8
        plt.yticks(np.arange(0, 0.81, 0.025))  # Set y-ticks from 0 to 0.8, with step 0.02

        # plt.xticks(np.arange(df[x_column].min(), df[x_column].max() + 0.05, 1))  # Set x-ticks with step 0.02
        
        
        # Improve the legend
        plt.legend(title='Variables', title_fontsize='13', fontsize='11', loc='upper right', fancybox=True, framealpha=0.7)

        # Adjust the grid and spines for better aesthetics
        sns.despine(trim=True)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Save the plot as a PNG file (optional)
        output_filename = os.path.join(folder_path, f'{filename[:-4]}_plot.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')

        # Show the plot
        # plt.show()
