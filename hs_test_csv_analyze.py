import os
import csv
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
import pandas as pd
import sys


# --- Configuration ---
# Directory containing the CSV files
log_directory = 'log'
# Number of top frequent tiles to report
top_tiles_count = 20
# Output summary filename
summary_filename = 'agent_summary.csv'
# --- End Configuration ---

# plt.rcParams['font.family'] = ['Droid Sans Japanese', 'Droid Sans', 'Noto Sans', 'Noto Sans CJK JP']
plt.rcParams['font.family'] = ['Noto Sans', 'Noto Sans CJK JP']

try:
    all_files = os.listdir(log_directory)
except FileNotFoundError:
    print(f"Error: Directory '{log_directory}' not found.")
    sys.exit()

# Filter files that start with "test_" and end with ".csv"
csv_files = [f for f in all_files if f.startswith('test_') and f.endswith('.csv')]

# The csv file looks like this:
# Episode,Seat,Score,Turn,Yaku,MZ_Score,TEZ_Score,TAZ_Score,Tenpai,HandComplete,AgariRate
# 1,North,5000,74,中 赤ドラ1 混一色,0,0,0,0,萬子3 萬子4 萬子5赤ドラ 萬子5 萬子6 萬子7 萬子7 萬子7 | 萬子8 萬子8 萬子8 中ちゃん 中ちゃん 中ちゃん,100.000
# 2,West,3000,68,混一色,0,0,0,0,索子4 索子5 索子6 北風 北風 | 索子1 索子2 索子3 索子3 索子4 索子5 索子9 索子9 索子9,100.000
# 3,East,0,110,,0,0,0,0,萬子3 萬子9 萬子9 萬子9 | 萬子8 萬子8 萬子8 筒子2 筒子2 筒子2 筒子6 筒子7 筒子8,66.667


# Check if there are any matching files
if not csv_files:
    print(f"No CSV files found in '{log_directory}' that match the pattern 'test_*.csv'.")
    sys.exit()

summary_data = []

print(f"Found {len(csv_files)} files matching 'test_*.csv'. Starting analysis...")
for file_name in csv_files:
    file_path = os.path.join(log_directory, file_name)
    print(f"\nProcessing file: {file_path}")

    avg_turn = 'N/A'
    avg_score = 'N/A'
    tile_counts = Counter()
    yaku_counts = Counter()  # New counter for tracking yaku

    try:
        df = pd.read_csv(file_path)
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

        if 'Score' in df.columns:
            # Convert 'Score' column to numeric, coercing errors to NaN
            numeric_score = pd.to_numeric(df['Score'], errors='coerce')
            # Drop NaN values before calculating mean
            valid_scores = numeric_score.dropna()
            if not valid_scores.empty:
                avg_score = valid_scores.mean()
                print(f"  Calculated Average Score: {avg_score:.2f}")
            else:
                 print("  'Score' column found, but contains no valid numeric data.")
                 avg_score = 'No valid score data'
        else:
            print("  Column 'Score' not found in file. Cannot calculate average score.")
            avg_score = 'Score column missing'

        # --- Count Yaku Frequencies ---
        if 'Yaku' in df.columns:
            # Process each non-null Yaku entry
            for yaku_str in df['Yaku'].dropna():
                # Split the yaku string by spaces to get individual yaku
                yakus = yaku_str.split()
                # Count each individual yaku
                for yaku in yakus:
                    yaku_counts[yaku] += 1
            
            # Get the top yaku occurrences
            most_common_yakus = yaku_counts.most_common()
            top_yakus_str = ", ".join([f"{yaku} ({count})" for yaku, count in most_common_yakus]) if most_common_yakus else 'None'
            print(f"  Yaku Frequencies: {top_yakus_str}")
        else:
            print("  Column 'Yaku' not found in file. Cannot analyze yaku frequencies.")
            top_yakus_str = 'Yaku column missing'

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
        if avg_score == 'N/A': avg_score = f'Processing error: {e}'
        top_yakus_str = f'Processing error: {e}'
        # tile_counts remains empty Counter()
        # Plotting would have been skipped

    # --- Determine Most Frequent Tiles ---
    most_common_tiles = tile_counts.most_common(top_tiles_count)
    top_tiles_str = ", ".join([f"{tile} ({count})" for tile, count in most_common_tiles]) if most_common_tiles else 'None'
    print(f"  Top {top_tiles_count} Most Frequent Tiles: {top_tiles_str}")

    # --- Store Summary Data ---
    summary_data.append({
        'Filename': file_name,
        'Average Turn': f'{avg_turn:.2f}' if isinstance(avg_turn, (int, float)) else avg_turn, # Format if numeric
        'Average Score': f'{avg_score:.2f}' if isinstance(avg_score, (int, float)) else avg_score, # Format if numeric
        'Top Tiles': top_tiles_str,
        'Yaku Frequencies': top_yakus_str  # New column for yaku frequencies
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