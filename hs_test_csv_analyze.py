import os
import csv
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd
import sys


def tile_to_index(self, tile):
    """Convert tile to unique index"""
    if tile.何者 == "萬子":
        return tile.その上の数字 - 1
    elif tile.何者 == "筒子":
        return 9 + tile.その上の数字 - 1
    elif tile.何者 == "索子":
        return 18 + tile.その上の数字 - 1
    elif tile.何者 == "東風":
        return 27
    elif tile.何者 == "南風":
        return 28
    elif tile.何者 == "西風":
        return 29
    elif tile.何者 == "北風":
        return 30
    elif tile.何者 == "白ちゃん":
        return 31
    elif tile.何者 == "發ちゃん":
        return 32
    elif tile.何者 == "中ちゃん":
        return 33

# --- Configuration ---
# Directory containing the CSV files
log_directory = 'log'
# Threshold frequency for a tile to be considered "preferred"
preferred_tile_threshold = 2222
# Output summary filename
summary_filename = 'agent_summary.csv'
# --- End Configuration ---

# Use a fallback mechanism with multiple fonts
# This is crucial for potential non-ASCII characters in future updates,
# though standard tile names like m1, p5, s9, ji1 are ASCII.
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Japanese']
# Make sure Unicode minus signs render correctly
plt.rcParams['axes.unicode_minus'] = False

# --- Optional: Print available fonts for debugging ---
# print("\nAvailable fonts with Japanese support:")
# japanese_fonts = [f.name for f in fm.fontManager.ttflist
#                   if any(keyword in f.name.lower() for keyword in ['cjk', 'jp', 'japanese', 'gothic', 'mincho', 'unicode'])]
# for font in sorted(japanese_fonts):
#     print(f" - {font}")
# print("-" * 20)
# --- End Optional ---

# Get all files in the directory
try:
    all_files = os.listdir(log_directory)
except FileNotFoundError:
    print(f"Error: Directory '{log_directory}' not found.")
    sys.exit()

# Filter files that start with "test_" and end with ".csv"
csv_files = [f for f in all_files if f.startswith('test_') and f.endswith('.csv')]

# Check if there are any matching files
if not csv_files:
    print(f"No CSV files found in '{log_directory}' that match the pattern 'test_*.csv'.")
    sys.exit()

# List to store summary data for each agent
summary_data = []

# Analyze each CSV file
print(f"Found {len(csv_files)} files matching 'test_*.csv'. Starting analysis...")
for file_name in csv_files:
    file_path = os.path.join(log_directory, file_name)
    print(f"\nProcessing file: {file_path}")

    avg_turn = 'N/A' # Default value
    tile_counts = Counter() # Initialize empty counter

    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path)

        # --- Calculate Average Turn ---
        if 'Turn' in df.columns:
            # Convert 'Turn' column to numeric, coercing errors to NaN
            numeric_turns = pd.to_numeric(df['Turn'], errors='coerce')
            # Drop NaN values before calculating mean
            valid_turns = numeric_turns.dropna()
            if not valid_turns.empty:
                avg_turn = valid_turns.mean()
                print(f"  Calculated Average Turn: {avg_turn:.2f}")
            else:
                 print("  'Turn' column found, but contains no valid numeric data.")
                 avg_turn = 'No valid turn data'
        else:
            print("  Column 'Turn' not found in file. Cannot calculate average turn.")
            avg_turn = 'Turn column missing'

        # --- Analyze Tile Frequencies ---
        if 'HandComplete' in df.columns:
            # Combine all tile data into a single list
            all_tiles = []
            # Iterate through non-null entries in 'HandComplete'
            for hand in df['HandComplete'].dropna():
                # Remove '|' and split by space
                tiles = hand.replace('|', '').split()
                all_tiles.extend(tiles)

            if all_tiles:
                # Count frequencies of each tile
                tile_counts = Counter(all_tiles)

                # Convert to a sorted list of tuples for consistent visualization
                sorted_tile_counts = sorted(tile_counts.items(), key=lambda x: x[0])

                if sorted_tile_counts:
                    # Separate tile names and their counts for plotting
                    tiles, counts = zip(*sorted_tile_counts)

                    # Plotting
                    plt.figure(figsize=(15, 6))
                    plt.bar(tiles, counts)
                    plt.xlabel('Mahjong Tiles')
                    plt.ylabel('Frequency')
                    plt.title(f'Frequency of Each Mahjong Tile in HandComplete - {file_name}')
                    plt.xticks(rotation=90)
                    plt.tight_layout()
                    # Save the plot in the log directory
                    plot_output_path = os.path.join(log_directory, f'{file_name.split(".")[0]}.png')
                    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
                    plt.close() # Close the figure to free memory
                    print(f"  Generated tile frequency plot: {plot_output_path}")
                else:
                    print("  No tile data found after processing 'HandComplete'. Skipping tile plot.")

            else:
                 print("  'HandComplete' column found, but contains no valid tile data after processing.")

        else:
            print("  Column 'HandComplete' not found in file. Skipping tile analysis and plot.")


    except Exception as e:
        print(f"  Error processing file {file_path}: {e}")
        # Assign error indicators if processing failed
        if avg_turn == 'N/A': avg_turn = f'Processing error: {e}'
        # tile_counts remains empty Counter()
        # Plotting would have been skipped

    # --- Determine Preferred Tiles ---
    preferred_tiles = [tile for tile, count in tile_counts.items() if count > preferred_tile_threshold]
    preferred_tiles_str = ", ".join(preferred_tiles) if preferred_tiles else 'None'
    print(f"  Preferred Tiles (> {preferred_tile_threshold} frequency): {preferred_tiles_str}")


    # --- Store Summary Data ---
    summary_data.append({
        'Filename': file_name,
        'Average Turn': f'{avg_turn:.2f}' if isinstance(avg_turn, (int, float)) else avg_turn, # Format if numeric
        'Preferred Tiles': preferred_tiles_str
    })

# --- Create Summary CSV ---
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(log_directory, summary_filename)

    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nAnalysis complete. Summary written to: {summary_csv_path}")
    except Exception as e:
        print(f"\nError writing summary CSV file {summary_csv_path}: {e}")
else:
    print("\nNo data was successfully processed to create a summary.")
