import os
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd
import sys

# Use a fallback mechanism with multiple fonts
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Japanese']
# Make sure Unicode minus signs render correctly
plt.rcParams['axes.unicode_minus'] = False

# Directory containing the CSV files
log_directory = 'log'

# Get all files in the directory
all_files = os.listdir(log_directory)

# Filter files that start with "test_" and end with ".csv"
csv_files = [f for f in all_files if f.startswith('test_') and f.endswith('.csv')]

# Check if there are any matching files
if not csv_files:
    print("No CSV files found that match the pattern 'test_*.csv'")
    sys.exit()
    
# Analyze each CSV file
for file_name in csv_files:
    file_path = os.path.join(log_directory, file_name)
    print(f"Processing file: {file_path}") # Add print

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        continue  # Skip to the next file

    # Combine all tile data into a single list
    all_tiles = []
    if 'HandComplete' in df.columns: # Check if the column exists
        for hand in df['HandComplete'].dropna():
            tiles = hand.replace('|', '').split()
            all_tiles.extend(tiles)
    else:
        print(f"Column 'HandComplete' not found in file {file_path}. Skipping.")
        continue # Skip to the next file

    # Count frequencies of each tile
    tile_counts = Counter(all_tiles)

    # Convert to a sorted list of tuples for consistent visualization
    sorted_tile_counts = sorted(tile_counts.items(), key=lambda x: x[0])

    # Separate tile names and their counts
    tiles, counts = zip(*sorted_tile_counts)

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.bar(tiles, counts)
    plt.xlabel('Mahjong Tiles')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Each Mahjong Tile in HandComplete - {file_name}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'./log/{file_name.split(".")[0]}.png', dpi=300, bbox_inches='tight')
    plt.close() # Close the figure to prevent overlap

# Debugging - print available fonts (only once)
print("\nAvailable fonts with Japanese support:")
japanese_fonts = [f.name for f in fm.fontManager.ttflist
                  if any(keyword in f.name.lower() for keyword in ['cjk', 'jp', 'japanese', 'gothic', 'mincho'])]
for font in sorted(japanese_fonts):
    print(f" - {font}")
