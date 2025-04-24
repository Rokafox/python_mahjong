from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd

# Set font to Noto Sans CJK
rcParams['font.family'] = 'Noto Sans CJK JP'

file_path = 'log/testing_Kurtv2.csv'
df = pd.read_csv(file_path)

# Combine all tile data into a single list
all_tiles = []
for hand in df['HandComplete'].dropna():
    tiles = hand.replace('|', '').split()
    all_tiles.extend(tiles)

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
plt.title('Frequency of Each Mahjong Tile in HandComplete')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('tile_frequency.png', dpi=300, bbox_inches='tight')