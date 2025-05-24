# Stores unused code

# def 聴牌まで最小交換回数(tiles: list[麻雀牌], seat: int, max_swaps: int = 13) -> tuple[int, list[tuple[list[麻雀牌], list[麻雀牌]]]]:
#     """
#     Calculate minimum number of tile swaps needed to achieve tenpai.
#     Args:
#         tiles: Current 13-tile hand
#         seat: Player seat number
#         max_swaps: Maximum number of swaps to try (default 13)
#     Returns:
#         tuple of (minimum_swaps_needed, list_of_optimal_swap_solutions)
#         If no solution found within max_swaps, returns (max_swaps + 1, [])
#     """
#     if len([t for t in tiles if not t.副露]) != 13:
#         raise ValueError(f"手牌は13枚でなければならない")
    
#     # Generate all possible tiles
#     全候補: list[tuple[str, int]] = []
#     for suit in ("萬子", "筒子", "索子"):
#         for num in range(1, 10):
#             全候補.append((suit, num))
#     全候補 += [(honor, 0) for honor in 
#                 ("東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん")]
    
#     # Check if already tenpai
#     is_tenpai, _ = 聴牌ですか(tiles, seat)
#     if is_tenpai:
#         return 0, [([], [])]  # No swaps needed
    
#     # Try increasing numbers of swaps
#     for num_swaps in range(1, max_swaps + 1):
#         solutions = []
        
#         # Try all combinations of tiles to remove
#         non_melded_tiles = [t for t in tiles if not t.副露]
#         for tiles_to_remove in combinations(non_melded_tiles, num_swaps):
#             # Create remaining hand after removal
#             remaining_tiles = [t for t in tiles if t not in tiles_to_remove]
            
#             # Try all combinations of replacement tiles
#             for replacement_combo in product(全候補, repeat=num_swaps):
#                 # Create new hand with replacements
#                 new_tiles = remaining_tiles.copy()
#                 replacement_tiles = []
                
#                 for 何者, 数字 in replacement_combo:
#                     new_tile = 麻雀牌(何者, 数字, False)
#                     new_tiles.append(new_tile)
#                     replacement_tiles.append(new_tile)
                
#                 # Check if this hand is tenpai
#                 try:
#                     is_tenpai, _ = 聴牌ですか(new_tiles, seat)
#                     if is_tenpai:
#                         solutions.append((list(tiles_to_remove), replacement_tiles))
#                 except:
#                     # Skip invalid hands
#                     continue
#         # If we found solutions with this number of swaps, return them
#         if solutions:
#             return num_swaps, solutions
#     # No solution found within max_swaps
#     return max_swaps + 1, []


# # Helper function to display results nicely
# def 結果表示(tiles: list[麻雀牌], result: tuple[int, list[tuple[list[麻雀牌], list[麻雀牌]]]]):
#     """
#     Display the results in a readable format.
#     """
#     min_swaps, solutions = result
    
#     if min_swaps > 13:  # No solution found
#         print("聴牌に到達できません（指定された最大交換回数内で）")
#         return
    
#     if min_swaps == 0:
#         print("既に聴牌です！")
#         return
    
#     print(f"最小交換回数: {min_swaps}枚")
#     print(f"解の数: {len(solutions)}個")
#     for i, (remove, add) in enumerate(solutions[:5]):  # Show max 5 solutions
#         print(f"\n解 {i+1}:")
#         print(f"  除去: {[f'{t.何者}{t.その上の数字 if t.その上の数字 > 0 else ""}' for t in remove]}")
#         print(f"  追加: {[f'{t.何者}{t.その上の数字 if t.その上の数字 > 0 else ""}' for t in add]}")
    
#     if len(solutions) > 5:
#         print(f"\n... 他 {len(solutions) - 5} 個の解があります")



# def generate_random_meld():
#     is_triplet = random.choice([True, False])
#     is_exposed = random.choice([True, False])
#     if is_triplet:
#         # Generate a triplet (three identical tiles)
#         suits = ["萬子", "筒子", "索子", "東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん"]
#         weights = [9, 9, 9, 1, 1, 1, 1, 1, 1, 1]
#         suit = random.choices(suits, weights=weights, k=1)[0]
#         if suit in ["萬子", "筒子", "索子"]:
#             num = random.randint(1, 9)
#         else:
#             num = 0
#         return [麻雀牌(suit, num, False, is_exposed), 麻雀牌(suit, num, False, is_exposed), 麻雀牌(suit, num, False, is_exposed)]
#     else:
#         # Generate a sequence (three consecutive numbers in the same suit)
#         suit = random.choice(["萬子", "筒子", "索子"])  # Only numbered suits can form sequences
#         # Can only start a sequence with 1-7
#         start_num = random.randint(1, 7)
#         return [麻雀牌(suit, start_num, False), 麻雀牌(suit, start_num+1, False), 麻雀牌(suit, start_num+2, False)]

# def generate_random_tile():
#     suits = ["萬子", "筒子", "索子", "東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん"]
#     weights = [9, 9, 9, 1, 1, 1, 1, 1, 1, 1]
#     suit = random.choices(suits, weights=weights, k=1)[0]
#     if suit in ["萬子", "筒子", "索子"]:
#         num = random.randint(1, 9)
#     else:
#         num = 0
#     return 麻雀牌(suit, num, False)


# def generate_random_41_13_hand():
#     hand = []
#     for _ in range(4):
#         meld = generate_random_meld()
#         hand.extend(meld)
#     hand.append(generate_random_tile())
#     return hand


# def generate_tenpai(max_attempts):
#     # Generate hands until we find one in tenpai
#     total_attempts = 0

#     while total_attempts < max_attempts:
#         total_attempts += 1
#         hand = generate_random_41_13_hand()
#         counter = Counter((t.何者, t.その上の数字) for t in hand)
#         # check valid hand
#         hand_is_valid = True
#         for key, cnt in counter.items():
#             if cnt > 4:
#                 hand_is_valid = False
#                 break
#         if not hand_is_valid:
#             continue
#         is_tenpai, waiting_tiles = 聴牌ですか(hand.copy(), 0)  # Passing seat as 0
#         if is_tenpai:
#             with open("tenpai_hands.txt", "a", encoding="utf-8") as f:
#                 f.write(f"{nicely_print_tiles(hand)}\n")



# def create_mahjong_tiles_from_line(line: str) -> list[麻雀牌]:
#     if line.endswith(" |"):
#         line = line[:-2].strip()
#     tiles = []
#     tile_specs = line.split()
#     for tile_spec in tile_specs:
#         if " " in tile_spec:
#             # Handle multi-tile input, which is not implemented.
#             raise ValueError("Multi-tile input is not supported.")
#         if tile_spec in {"東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん"}:
#             tiles.append(麻雀牌(tile_spec, 0))
#         else:
#             牌名 = tile_spec[:-1]
#             数字 = int(tile_spec[-1])
#             tiles.append(麻雀牌(牌名, 数字))
#     return tiles
