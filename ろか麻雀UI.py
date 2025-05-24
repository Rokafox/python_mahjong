from collections import Counter
import collections
from copy import copy
import csv
import random
import numpy as np
import pygame, pygame_gui
import pygame.freetype
import sys
import os
from typing import List, Dict, Tuple, Optional

import torch
from ろかAI_v6 import DQNAgent
from ろか麻雀 import calculate_weighted_preference_score, nicely_print_tiles, 山を作成する, 点数計算, 聴牌ですか, 麻雀牌



class Env:
    """
    Mahjong game env
    """

    SUIT_MAPPING = {"萬子": 0, "筒子": 1, "索子": 2}
    HONOR_ID_MAPPING = {
        "東風": 1, "南風": 2, "西風": 3, "北風": 4,
        "白ちゃん": 5, "發ちゃん": 6, "中ちゃん": 7
    }
    _MAX_DISCARD_GROUPS = 14
    ACTION_NOPON = _MAX_DISCARD_GROUPS + 0
    ACTION_PON = _MAX_DISCARD_GROUPS + 1
    ACTION_NOCHI = _MAX_DISCARD_GROUPS + 2
    ACTION_CHI = _MAX_DISCARD_GROUPS + 3
    AGENT_HAND_SIZE = 13

    def __init__(self):
        self.山: list[麻雀牌] = []
        self.player_hand: list[麻雀牌] = []
        self.opponent_hand: list[麻雀牌] = []
        self.turn: int = 1
        self.current_actor: int = 0 # 0: player, 1: opponent
        self.discard_pile_player = []
        self.discard_pile_opponent = []
        self.player_seat: int = 0  # 0=東, 1=南, 2=西, 3=北
        self.opponent_seat: int = 2  # 0=東, 1=南, 2=西, 3=北
        self.game_complete = False
        self.player_name = "令狐蘆花"
        self.opponent_name = "ヒルチャール"
        self.player_points = 25000
        self.opponent_points = 25000
        self.starting_points = 25000 
        self.action_lgid_to_canonical_map: dict[int, int] = {} # For dqn_agent

    def start_new_game(self, opponent_name: str):
        self.opponent_name = opponent_name
        self.generate_pile()
        # give 13 tiles for each player
        self.player_hand = self.山[:13]
        self.opponent_hand = self.山[13:26]
        self.山 = self.山[26:]
        self.player_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        self.opponent_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        assert len(self.player_hand) == len(self.opponent_hand) == 13
        self.turn = 1
        self.current_actor = 0
        self.discard_pile_player = []
        self.discard_pile_opponent = []
        self.player_seat = random.randint(0, 3)
        match self.player_seat:
            case 0:
                self.opponent_seat = 2
            case 1:
                self.opponent_seat = 3
            case 2:
                self.opponent_seat = 0
            case 3:
                self.opponent_seat = 1
        self.game_complete = False
        assert len(self.山) == 136 - 26
        self.action_lgid_to_canonical_map.clear()

    def generate_pile(self):
        self.山 = 山を作成する()
        # self.山 = 基礎訓練山を作成する()

    def player_draw_tile(self) -> Optional[麻雀牌]:
        if len(self.山) == 0:
            return None
        tile = self.山.pop(0)
        self.player_hand.append(tile)
        self.player_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        return tile

    def opponent_draw_tile(self) -> Optional[麻雀牌]:
        if len(self.山) == 0:
            return None
        tile = self.山.pop(0)
        self.opponent_hand.append(tile)
        self.opponent_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        return tile
    

    # ==========================
    # Methods for RL agent
    # ==========================

    def _get_canonical_id(self, tile: 麻雀牌) -> int:
        """Still useful for comparing tiles or other internal logic."""
        if tile is None: return -1
        category, num = tile.何者, tile.その上の数字
        if category == "萬子": return num - 1
        if category == "筒子": return 9 + num - 1
        if category == "索子": return 18 + num - 1
        if category == "東風": return 27
        if category == "南風": return 28
        if category == "西風": return 29
        if category == "北風": return 30
        if category == "白ちゃん": return 31
        if category == "發ちゃん": return 32
        if category == "中ちゃん": return 33
        return -2 # Should not happen

    def _are_tiles_identical_type(self, tile1: 麻雀牌, tile2: 麻雀牌) -> bool:
        """Checks if two tiles are of the exact same type (e.g., 1 Man == 1 Man)."""
        if tile1 is None or tile2 is None:
            return tile1 is tile2 # Both None is True, one None is False
        # Using __eq__ defined in 麻雀牌 if available and appropriate,
        # or comparing canonical IDs.
        # return tile1 == tile2 # If __eq__ is defined for type matching
        return self._get_canonical_id(tile1) == self._get_canonical_id(tile2)


    def _get_tile_base_properties(self, tile: 麻雀牌 | None) -> tuple:
        """
        Extracts base properties (excluding local group ID).
        Returns: (suit_cat, number, is_exposed, is_honor, honor_id_val)
        """
        if tile is None:
            return (-1, 0, 0, 0, 0) # suit_cat, number, exposed, is_honor, honor_id

        suit_val, num_val = tile.何者, tile.その上の数字
        is_honor_val, honor_id_val, suit_cat = 0, 0, -1

        if suit_val in self.SUIT_MAPPING:
            suit_cat = self.SUIT_MAPPING[suit_val]
        else: # Honor tiles
            suit_cat = 3 # Dedicated category for honors
            is_honor_val = 1
            honor_id_val = self.HONOR_ID_MAPPING.get(suit_val, 0)
            num_val = 0

        is_exposed_val = tile.get_exposed_status()
        return (suit_cat, num_val, is_exposed_val, is_honor_val, honor_id_val)


    def _get_state_unified(self, last_tile: 麻雀牌 | None = None) -> np.ndarray:
        actual_hand_tiles = sorted(
            [tile for tile in self.opponent_hand if tile is not None],
            key=lambda t: self._get_canonical_id(t)
        )

        # Calculate tile relationships for 順子 detection
        tile_relationships = [0]  # First tile has no previous tile to compare
        for i in range(1, len(actual_hand_tiles)):
            current_tile = actual_hand_tiles[i]
            prev_tile = actual_hand_tiles[i-1]
            
            # Get properties for comparison
            current_suit, current_num = current_tile.何者, current_tile.その上の数字
            prev_suit, prev_num = prev_tile.何者, prev_tile.その上の数字
            
            # Default relationship: no relation
            relationship = 0
            
            # Check if same suit/honor category
            if current_suit == prev_suit:
                if current_num == prev_num:
                    # Same tile type (helps with 刻子 detection)
                    relationship = 1
                elif abs(current_num - prev_num) == 1:
                    # Sequential tiles (helps with 順子 detection)
                    relationship = 2
            
            tile_relationships.append(relationship)
        
        # Pad relationships to match hand size
        while len(tile_relationships) < self.AGENT_HAND_SIZE:
            tile_relationships.append(0)

        hand_local_group_ids = []
        canonical_to_group_map = {}
        next_available_group_id = 0
        
        for tile in actual_hand_tiles:
            tile_cano_id = self._get_canonical_id(tile)
            if tile_cano_id not in canonical_to_group_map:
                canonical_to_group_map[tile_cano_id] = next_available_group_id
                hand_local_group_ids.append(next_available_group_id)
                next_available_group_id += 1
            else:
                hand_local_group_ids.append(canonical_to_group_map[tile_cano_id])
        
        max_local_id_in_hand = next_available_group_id -1 if next_available_group_id > 0 else -1

        hand_features_list = []
        for i in range(self.AGENT_HAND_SIZE):
            if i < len(actual_hand_tiles): # Use actual_hand_tiles here
                tile = actual_hand_tiles[i]
                local_id = hand_local_group_ids[i]
                relationship_code = tile_relationships[i]  # Add the relationship code
                base_props = self._get_tile_base_properties(tile)
                hand_features_list.extend([local_id, relationship_code, *base_props])
            else: 
                padding_local_id = -1
                relationship_code = 0 
                base_props_padding = self._get_tile_base_properties(None)
                hand_features_list.extend([padding_local_id, relationship_code, *base_props_padding])
        
        hand_tiles_state = np.array(hand_features_list, dtype=np.float32)

        # --- Last Tile Encoding ---
        last_tile_local_id = -1 # Default for padding 
        last_tile_relationship = 0  # Default relationship for last tile
        base_props_last_tile = self._get_tile_base_properties(last_tile)

        if last_tile is not None:
            last_tile_cano_id = self._get_canonical_id(last_tile)
            if last_tile_cano_id in canonical_to_group_map:
                last_tile_local_id = canonical_to_group_map[last_tile_cano_id]
            else:
                last_tile_local_id = max_local_id_in_hand + 1
            
            # Check relationship with ANY tile in the hand
            has_same_tile = False
            has_sequential_tile = False
            
            if actual_hand_tiles:
                last_suit, last_num = last_tile.何者, last_tile.その上の数字
                
                for hand_tile in actual_hand_tiles:
                    hand_suit, hand_num = hand_tile.何者, hand_tile.その上の数字
                    
                    if last_suit == hand_suit:
                        if last_num == hand_num:
                            has_same_tile = True
                        elif abs(last_num - hand_num) == 1:
                            has_sequential_tile = True
                    
                    # If we already found both relationships, we can stop checking
                    if has_same_tile and has_sequential_tile:
                        break
            
            # Encode relationship as:
            # 0: No relationship
            # 1: Same tile exists
            # 2: Sequential tile exists
            # 3: Both relationships exist
            if has_same_tile and has_sequential_tile:
                last_tile_relationship = 3
            elif has_same_tile:
                last_tile_relationship = 1
            elif has_sequential_tile:
                last_tile_relationship = 2
        
        last_tile_state_list = [last_tile_local_id, last_tile_relationship, *base_props_last_tile]
        last_tile_state = np.array(last_tile_state_list, dtype=np.float32)
        
        # --- Seat Encoding ---
        seat_state = np.zeros(4, dtype=np.float32)
        seat_state[self.opponent_seat] = 1.0
        
        actor_state = np.zeros(4, dtype=np.float32)
        if self.current_actor == 1:
            actor_state[0] = 1.0
        else:
            actor_state[1] = 1.0

        # --- Turn Encoding ---
        # Cap turn at 57
        capped_turn = min(self.turn, 57)
        # Normalize to [0,1] range
        normalized_turn = capped_turn / 57.0
        turn_state = np.array([normalized_turn], dtype=np.float32)
        
        # --- Tile Availability Encoding ---
        tile_availability = []
        
        # Create a counter for tiles in discard pile
        discard_counter = {}
        for tile in self.discard_pile_opponent + self.discard_pile_player:
            cano_id = self._get_canonical_id(tile)
            discard_counter[cano_id] = discard_counter.get(cano_id, 0) + 1
        
        # Create a counter for tiles in hand
        hand_counter = {}
        for tile in self.opponent_hand:
            if tile is not None:
                cano_id = self._get_canonical_id(tile)
                hand_counter[cano_id] = hand_counter.get(cano_id, 0) + 1

        # Create a counter for exposed tiles of player
        roka_tiles_exposed_counter = {}
        for tile in self.player_hand:
            if tile is not None and tile.副露:
                cano_id = self._get_canonical_id(tile)
                hand_counter[cano_id] = hand_counter.get(cano_id, 0) + 1
        
        # Calculate availability for each hand tile
        for i in range(self.AGENT_HAND_SIZE):
            if i < len(actual_hand_tiles):
                tile = actual_hand_tiles[i]
                cano_id = self._get_canonical_id(tile)
                total_count = hand_counter.get(cano_id, 0) + discard_counter.get(cano_id, 0)
                available = max(0, 4 - total_count)  # Ensure non-negative
                tile_availability.append(available)
            else:
                # Padding for empty slots
                tile_availability.append(0)
        
        tile_availability_state = np.array(tile_availability, dtype=np.float32)

        # state size calculation:
        # (AGENT_HAND_SIZE) * (1 local ID + 1 relationship + 5 base props) -> 13 * 7 = 91 (for hand slots)
        # + 1 * (1 local ID + 1 relationship + 5 base props)                -> 1 * 7 = 7  (for last tile slot)
        # + 4                                                               -> 4          (for seat)
        # + 4                                                               -> 4          (for actor)
        # Total: 91 + 7 + 4 + 4 = 106

        # state size calculation update:
        # Previous total: 106
        # + 1 (turn)
        # + 13 (tile availability)
        # New total: 120
        
        return np.concatenate([
            hand_tiles_state,          # (AGENT_HAND_SIZE) * 7 features
            last_tile_state,           # 1 * 7 features
            seat_state,                # 4 features
            actor_state,               # 4 features
            turn_state,                # 1 feature
            tile_availability_state    # AGENT_HAND_SIZE features
        ])




    def get_valid_actions(self, last_tile: 麻雀牌 | None = None, tenpai: bool = False) -> list[int]:
        """
        Generates a list of valid action IDs.
        For agent's turn (discard): action IDs are local group IDs of discardable tiles.
        For opponent's turn (call/pass): action IDs are special constants (14, 15, 16, 17): (nopon, pon, nochii, chii)
        """
        self.action_lgid_to_canonical_map.clear() # Clear map from previous turn

        if self.current_actor == 1: # Agent's turn: Discard a tile
            discardable_tiles_in_hand = [t for t in self.opponent_hand if not t.副露]
            effective_hand_for_action = [t for t in self.opponent_hand] # instead of discardable_tiles_in_hand
            if last_tile:
                effective_hand_for_action.append(last_tile)
            if not effective_hand_for_action: # Should not happen in a normal game state
                raise Exception
            effective_hand_for_action.sort(key=lambda x: (x.sort_order, x.その上の数字)) # Relies on 麻雀牌.__lt__
            # Generate local group IDs specifically for this action context
            action_specific_lgids = []
            # Temp map for THIS call, to build action_lgid_to_canonical_map and get unique lgids
            temp_cano_to_action_lgid = {} 
            next_action_lgid = 0

            for tile in effective_hand_for_action:
                cano_id = self._get_canonical_id(tile)
                if cano_id not in temp_cano_to_action_lgid:
                    assigned_lgid = next_action_lgid
                    temp_cano_to_action_lgid[cano_id] = assigned_lgid
                    # Populate the instance map for use in step()
                    self.action_lgid_to_canonical_map[assigned_lgid] = cano_id 
                    next_action_lgid += 1
                else:
                    assigned_lgid = temp_cano_to_action_lgid[cano_id]
                if not tile.副露:
                    action_specific_lgids.append(assigned_lgid)

            if tenpai and last_tile:
                last_tile_cano_id = self._get_canonical_id(last_tile)
                # Find the action LGID assigned to this last_tile's type
                lgid_for_last_tile_action = temp_cano_to_action_lgid.get(last_tile_cano_id)
                
                if lgid_for_last_tile_action is not None:
                    # Ensure the map for step() is correctly populated for this single action
                    # (already done above if last_tile was processed, but good to be explicit if we were to optimize)
                    # self.action_lgid_to_canonical_map = {lgid_for_last_tile_action: last_tile_cano_id}
                    return [lgid_for_last_tile_action]
                else:
                    raise Exception
            # Return unique, sorted local group IDs that can be discarded
            unique_valid_action_lgids = sorted(list(set(action_specific_lgids)))
            return unique_valid_action_lgids
        
        else:
            return [self.ACTION_NOPON, self.ACTION_PON, self.ACTION_NOCHI, self.ACTION_CHI] # 14, 15, 16, 17


    def map_action_lgid_to_canonical_id(self, action_lgid: int) -> int | None:
        """
        Maps a local group ID (used as an action) to the canonical tile ID it represents.
        This map is populated by get_valid_actions.
        """
        return self.action_lgid_to_canonical_map.get(action_lgid)



    # ==========================
    # End of Methods for RL agent
    # ==========================



def add_outline_to_image(surface, outline_color, outline_thickness):
    """
    Adds an outline to the image at the specified index in the image_slots list.

    Parameters:
    image (pygame.Surface): Image to add the outline to.
    outline_color (tuple): Color of the outline in RGB format.
    outline_thickness (int): Thickness of the outline.
    """
    new_image = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    new_image.fill((255, 255, 255, 0)) 
    new_image.blit(surface, (0, 0))

    rect = pygame.Rect((0, 0), surface.get_size())
    pygame.draw.rect(new_image, outline_color, rect, outline_thickness)

    return new_image



if __name__ == "__main__":
    pygame.init()
    clock = pygame.time.Clock()

    antique_white = pygame.Color("#FAEBD7")
    deep_dark_blue = pygame.Color("#000022")
    light_yellow = pygame.Color("#FFFFE0")
    light_purple = pygame.Color("#f0eaf5")
    light_red = pygame.Color("#fbe4e4")
    light_green = pygame.Color("#e5fae5")
    light_blue = pygame.Color("#e6f3ff")
    light_pink = pygame.Color("#fae5eb")

    display_surface = pygame.display.set_mode((1600, 900), flags=pygame.SCALED | pygame.RESIZABLE)
    ui_manager_lower = pygame_gui.UIManager((1600, 900), "theme_light_purple.json", starting_language='ja')
    ui_manager = pygame_gui.UIManager((1600, 900), "theme_light_purple.json", starting_language='ja')
    ui_manager_overlay = pygame_gui.UIManager((1600, 900), "theme_light_purple.json", starting_language='ja')
    # debug_ui_manager = pygame_gui.UIManager((1600, 900), "./asset/ui_theme/theme_light_yellow.json", starting_language='ja')
    # ui_manager.get_theme().load_theme("./asset/ui_theme/theme_light_purple.json")
    # ui_manager.rebuild_all_from_changed_theme_data()

    pygame.display.set_caption("Roka Mahjong")
    # if there is icon, use it
    try:
        icon = pygame.image.load("icon.png")
        pygame.display.set_icon(icon)
    except Exception as e:
        print(f"Error loading icon: {e}")

    try:
        dqn_agent = DQNAgent(120, 18, device="cuda")
    except Exception:
        print("Cuda Device not found, using cpu version.")
        dqn_agent = DQNAgent(120, 18, device="cpu")

    dqn_agent.epsilon = 0

    print("Starting!")
    running = True 

    def create_tile_slots(ui_manager: pygame_gui.UIManager,
                                start_pos: tuple[int, int],
                                tile_size: tuple[int, int] = (66, 90),
                                count: int = 14,
                                spacing: int = 4
                                ) -> list[pygame_gui.elements.UIImage]:
        """
        手牌スロットを一気に作成する。
        ui_manager: pygame_gui.UIManager のインスタンス
        start_pos: (x, y) のタプル。最初の牌スロットの左上座標
        tile_size: 各スロットのサイズ (幅, 高さ)
        count: 作成するスロット数（デフォルト 14）
        spacing: スロット間のピクセル間隔（デフォルト 4）
        ->
        UIImage オブジェクトのリスト
        """
        slots = []
        x0, y0 = start_pos
        w, h = tile_size

        for i in range(count):
            rect = pygame.Rect((x0 + i * (w + spacing), y0), (w, h))
            surface = pygame.Surface((w, h))
            # 必要に応じて surface.fill() や画像貼り付けを行っておく
            slot = pygame_gui.elements.UIImage(
                relative_rect=rect,
                image_surface=surface,
                manager=ui_manager
            )
            slots.append(slot)

        return slots

    # Player hands have at most 14 tiles, exposed is not considered
    # player_tile_slot_1 = pygame_gui.elements.UIImage(pygame.Rect((100, 700), (66, 90)),
    #                                 pygame.Surface((66, 90)),
    #                                 ui_manager)
    player_tile_slots = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 700),
        tile_size=(66, 90),
        count=14,
        spacing=4
    )

    opponent_tile_slots = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 200 - 90),
        tile_size=(66, 90),
        count=14,
        spacing=4
    )

    player_discarded_tiles_group_a = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 480),
        tile_size=(50, 68),
        count=27,
        spacing=3
    )

    opponent_discarded_tiles_group_a = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 420 - 66),
        tile_size=(50, 68),
        count=27,
        spacing=3
    )

    player_discarded_tiles_group_b = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 480 + 71),
        tile_size=(50, 68),
        count=27,
        spacing=3
    )

    opponent_discarded_tiles_group_b = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 420 - 66 - 71),
        tile_size=(50, 68),
        count=27,
        spacing=3
    )

    player_name_label = pygame_gui.elements.UILabel(pygame.Rect((100, 800), (200, 50)),
                                                    text="Player",
                                                    manager=ui_manager)
    opponent_name_label = pygame_gui.elements.UILabel(pygame.Rect((100, 50), (350, 50)),
                                                    text="Opponent",
                                                    manager=ui_manager)

    game_state_text_box = pygame_gui.elements.UITextEntryBox(pygame.Rect((1100, 700), (480, 180)),"", ui_manager)


    env = Env()
    image_405: pygame.Surface = pygame.image.load("asset/405.png")

    def draw_ui_player_hand():
        assert len(env.player_hand) <= 14
        env.player_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        for i, t in enumerate(env.player_hand):
            image_path = t.get_asset()
            try:
                image_surface = pygame.image.load(image_path)
                player_tile_slots[i].set_temp_marked = False
                # We need to consider whether the tile is exposed, if so, add sliver outline
                if not t.副露:
                    player_tile_slots[i].set_image(image_surface)
                else:
                    new_image = add_outline_to_image(image_surface, (186, 85, 211), 2)
                    player_tile_slots[i].set_image(new_image)
            except pygame.error as e:
                print(f"Error loading image {image_path}: {e}")
        if len(env.player_hand) < 14:
            for i in range(len(env.player_hand), 14):
                player_tile_slots[i].set_image(image_405)
                player_tile_slots[i].set_temp_marked = True


    def draw_ui_player_discarded_tiles():
        if not env.discard_pile_player:
            for uiimage in player_discarded_tiles_group_a + player_discarded_tiles_group_b:
                uiimage.set_image(image_405)
        pile_length = len(env.discard_pile_player)
        for i, t in enumerate(env.discard_pile_player):
            if i < pile_length - 3:
                # optimize: no point redrawing previous tiles.
                continue
            image_path = t.get_asset()
            image_surface = pygame.image.load(image_path)
            if i < 27:
                player_discarded_tiles_group_a[i].set_image(image_surface)
            elif 27 <= i < 54:
                player_discarded_tiles_group_b[i - 27].set_image(image_surface)
            else:
                raise Exception("Too many discarded tiles")
        # the next tile in pile is removed to handle pon or chii.
        if pile_length < 27:
            player_discarded_tiles_group_a[pile_length].set_image(image_405)
        elif 27 <= pile_length < 53:
            player_discarded_tiles_group_b[pile_length - 27].set_image(image_405)


    def draw_ui_opponent_hand():
        image_tile_hidden = pygame.image.load("asset/tile_hidden_questionmark.png")
        env.opponent_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        for i, t in enumerate(env.opponent_hand):
            image_path = t.get_asset()
            try:
                image_surface = pygame.image.load(image_path)
                opponent_tile_slots[i].set_temp_marked = False
                if env.game_complete:
                    if not t.副露:
                        opponent_tile_slots[i].set_image(image_surface)
                    else:
                        new_image = add_outline_to_image(image_surface, (186, 85, 211), 2)
                        opponent_tile_slots[i].set_image(new_image)
                else:
                    if not t.副露:
                        opponent_tile_slots[i].set_image(image_tile_hidden)
                    else:
                        new_image = add_outline_to_image(image_surface, (186, 85, 211), 2)
                        opponent_tile_slots[i].set_image(new_image)
            except pygame.error as e:
                print(f"Error loading image {image_path}: {e}")
        if len(env.opponent_hand) < 14:
            for i in range(len(env.opponent_hand), 14):
                opponent_tile_slots[i].set_image(image_405)
                opponent_tile_slots[i].set_temp_marked = True



    def draw_ui_opponent_discarded_tiles():
        if not env.discard_pile_opponent:
            for uiimage in opponent_discarded_tiles_group_a + opponent_discarded_tiles_group_b:
                uiimage.set_image(image_405)
        pile_length = len(env.discard_pile_opponent)
        for i, t in enumerate(env.discard_pile_opponent):
            if i < pile_length - 3:
                # optimize: no point redrawing previous tiles.
                continue
            image_path = t.get_asset()
            image_surface = pygame.image.load(image_path)
            if i < 27:
                opponent_discarded_tiles_group_a[i].set_image(image_surface)
            elif 27 <= i < 54:
                opponent_discarded_tiles_group_b[i - 27].set_image(image_surface)
            else:
                raise Exception("Too many discarded tiles")
        # the next tile in pile is removed to handle pon or chii.
        if pile_length < 27:
            opponent_discarded_tiles_group_a[pile_length].set_image(image_405)
        elif 27 <= pile_length < 53:
            opponent_discarded_tiles_group_b[pile_length - 27].set_image(image_405)



    def player_check_discard_what_to_tenpai():
        # remove all prev tooltips
        for s in player_tile_slots:
            s.set_tooltip("", delay=0.1)
        assert len(env.player_hand) == 14
        for i, t in enumerate(env.player_hand):
            if t.副露:
                continue
            pirate_hand = env.player_hand.copy()
            pirate_hand.pop(i)
            is_tenpai, list_of_tiles_to_tenpai = 聴牌ですか(pirate_hand, env.player_seat)
            if is_tenpai:
                s = str(env.player_hand[i])
                tp = f"\n聴牌:\n{nicely_print_tiles(list_of_tiles_to_tenpai)[:-2].strip()}"
                player_tile_slots[i].set_tooltip(s + tp, delay=0.1)
                image = player_tile_slots[i].image
                new_image = add_outline_to_image(image, (255, 215, 0), 2)
                player_tile_slots[i].set_image(new_image)


    last_tile_discarded_by_opponent: 麻雀牌 | None = None
    last_tile_discarded_by_player: 麻雀牌 | None = None
    player_chii_tiles: List[麻雀牌] = []
    player_win_points: int = 0
    player_win_yaku: list[str] = []


    def progress_opponent_turn():
        global last_tile_discarded_by_opponent, last_tile_discarded_by_player, player_chii_tiles
        global player_win_points, player_win_yaku
        for s in player_tile_slots:
            s.set_tooltip("", delay=0.1)
        player_can_call = False

        if last_tile_discarded_by_player:
            assert last_tile_discarded_by_player.副露 == False

        # Opponent Ron
        temp_opponent_hand = env.opponent_hand.copy()
        if last_tile_discarded_by_player:
            temp_opponent_hand.append(last_tile_discarded_by_player)
        a0, b0, c0 = 点数計算(temp_opponent_hand, env.opponent_seat)
        if c0:
            # opponent win by ron
            env.game_complete = True
            game_state_text_box.append_html_text(f"{env.opponent_name}: ロン！\n")
            game_state_text_box.append_html_text(f"{env.opponent_name}: {' '.join(b0)}\n")
            game_state_text_box.append_html_text(f"{env.opponent_name}: {a0}点\n")
            button_tsumo.hide()
            button_chii.hide()
            button_pon.hide()
            button_ron.hide()
            button_pass.hide()
            player_win_points = 0
            player_win_yaku = []
            for s in player_tile_slots:
                s.set_tooltip("", delay=0.1)
            draw_ui_opponent_hand()
            env.opponent_points += a0
            env.player_points -= a0
            draw_ui_player_labels()
            return None

        hiruchaaru_is_tenpai, list_of_hr_tenpai = 聴牌ですか(env.opponent_hand, env.opponent_seat)
        # Check if opponent can pon or chii
        hiruchaaru_to_pon: bool = False
        hiruchaaru_to_chii: bool = False
        # pon
        if len([t for t in env.opponent_hand if not t.副露]) > 1 and len(env.discard_pile_opponent) < 54 and not hiruchaaru_is_tenpai:
            same_tile_count = 0
            same_tiles = []
            for i, tile in enumerate(env.opponent_hand):
                if last_tile_discarded_by_player:
                    if (tile.何者, tile.その上の数字) == (last_tile_discarded_by_player.何者, last_tile_discarded_by_player.その上の数字) and not tile.副露:
                        same_tile_count += 1
                        same_tiles.append(i)
            if same_tile_count >= 2:
                # can pon
                valid_actions: list = env.get_valid_actions(last_tile_discarded_by_player)
                valid_actions.remove(16)
                valid_actions.remove(17)
                assert valid_actions == [14, 15]
                action, full_dict = dqn_agent.act(env._get_state_unified(last_tile_discarded_by_player), valid_actions)
                if action == 15:
                    game_state_text_box.append_html_text(f"{env.opponent_name}: ポン！\n")
                    # Mark the 2 tiles in agent's hand as 副露
                    for i in sorted(same_tiles[:2], reverse=True):
                        env.opponent_hand[i].mark_as_exposed("pon")
                    
                    # Add the discarded tile to agent's hand and mark it as 副露
                    if last_tile_discarded_by_player:
                        last_tile_discarded_by_player.mark_as_exposed("pon")
                        env.opponent_hand.append(last_tile_discarded_by_player)
                    last_tile_discarded_by_player = None
                    
                    # Remove the discarded tile from the discard pile
                    env.discard_pile_player.pop()
                    hiruchaaru_to_pon = True


                else: # action 14
                    pass
        # chii
        if len([t for t in env.opponent_hand if not t.副露]) > 2 and len(env.discard_pile_opponent) < 54 and last_tile_discarded_by_player and not hiruchaaru_is_tenpai:
            can_chii = False
            chii_tiles = []
            
            if "中張牌" in last_tile_discarded_by_player.固有状態:
                dt_num = last_tile_discarded_by_player.その上の数字
                possible_sequence = [(last_tile_discarded_by_player.何者, dt_num-1), (last_tile_discarded_by_player.何者, dt_num+1)] # [n-1,n,n+1]
                
                # print(f"Checking sequence: {possible_sequence}")
                # Checking sequence: [('筒子', 6), ('筒子', 7)]
                check = 0
                # check if both these tiles are in the hand
                for combo in possible_sequence:
                    # combo: ('筒子', 6) then ('筒子', 7)
                    for p in env.opponent_hand:
                        if p.何者 == combo[0] and p.その上の数字 == combo[1] and not p.副露:
                            check += 1
                            chii_tiles.append(p)
                            break
                assert check <= 2, "No, not happening"
                if check == 2:
                    can_chii = True
                else:
                    chii_tiles.clear()
                            

                if can_chii:
                    valid_actions: list = env.get_valid_actions(last_tile_discarded_by_player)
                    valid_actions.remove(14)
                    valid_actions.remove(15)
                    assert valid_actions == [16, 17]
                    action, full_dict = dqn_agent.act(env._get_state_unified(last_tile_discarded_by_player), valid_actions)
                    if action == 17:
                        game_state_text_box.append_html_text(f"{env.opponent_name}: チー！\n")
                        for t in env.opponent_hand:
                            if (t.何者, t.その上の数字) == (chii_tiles[0].何者, chii_tiles[0].その上の数字) and not t.副露:
                                t.mark_as_exposed("chii")
                                break
                        for t in env.opponent_hand:
                            if (t.何者, t.その上の数字) == (chii_tiles[1].何者, chii_tiles[1].その上の数字) and not t.副露:
                                t.mark_as_exposed("chii")
                                break

                        last_tile_discarded_by_player.mark_as_exposed("chii")
                        env.opponent_hand.append(last_tile_discarded_by_player)
                        last_tile_discarded_by_player = None
                        env.discard_pile_player.pop()
                        hiruchaaru_to_chii = True
                    else: # action 16
                        pass


        env.current_actor = 1
        env.turn += 1
        newly_drawn_tile_index: int = -1
        # tenpai_before_draw: bool = False

        if not hiruchaaru_to_pon and not hiruchaaru_to_chii:
            if len(env.discard_pile_opponent) < 54:
                # Opponent draw a tile
                # tenpai_before_draw, list_of_tanpai = 聴牌ですか(env.opponent_hand, env.opponent_seat)
                new_tile = env.opponent_draw_tile()
                

                # Opponent tsumo
                a, b, c = 点数計算(env.opponent_hand, env.opponent_seat) # This func returns 点数int, 完成した役list, 和了形bool
                if c:
                    # opponent win by tsumo
                    env.game_complete = True
                    game_state_text_box.append_html_text(f"{env.opponent_name}: ツモ！\n")
                    game_state_text_box.append_html_text(f"{env.opponent_name}: {' '.join(b)}\n")
                    game_state_text_box.append_html_text(f"{env.opponent_name}: {a}点\n")
                    button_tsumo.hide()
                    button_chii.hide()
                    button_pon.hide()
                    button_ron.hide()
                    button_pass.hide()
                    player_win_points = 0
                    player_win_yaku = []
                    for s in player_tile_slots:
                        s.set_tooltip("", delay=0.1)
                    draw_ui_opponent_hand()
                    env.opponent_points += a
                    env.player_points -= a
                    draw_ui_player_labels()
                    return None
            else:
                game_state_text_box.append_html_text("無役終局\n")
                env.game_complete = True
                button_tsumo.hide()
                button_chii.hide()
                button_pon.hide()
                button_ron.hide()
                button_pass.hide()
                player_win_points = 0
                player_win_yaku = []
                for s in player_tile_slots:
                    s.set_tooltip("", delay=0.1)
                draw_ui_opponent_hand()
                return None
        else:
            # Hiruchaaru poned or chiied, so they do not draw, but need to discard.
            new_tile = None

        valid_actions: list = env.get_valid_actions(new_tile, tenpai=hiruchaaru_is_tenpai)
        assert len(valid_actions) > 0
        action, full_dict = dqn_agent.act(env._get_state_unified(new_tile), valid_actions)
        # print(f"Opponent action: {action}")
        # print(env._index_to_tile_type(action))discarded_tile
        # print(tile_type) # ('筒子', 5)
        assert 0 <= action <= 13
        tile_type = env.map_action_lgid_to_canonical_id(int(action))
        discarded_tile = None
        for t in [t for t in env.opponent_hand if not t.副露]:
            if env._get_canonical_id(t) == tile_type:
                discarded_tile = t
                break
        if discarded_tile is None:
            print(nicely_print_tiles(env.opponent_hand))
            print(action)
            print(tile_type)
            raise ValueError(f"Invalid action: {tile_type} not in hand.")
        # Remove the tile from the hand and add it to the discard pile
        assert discarded_tile.副露 == False, f"Exposed tile can not be discarded! {discarded_tile.何者}{discarded_tile.その上の数字}"


        last_tile_discarded_by_opponent = discarded_tile
        env.discard_pile_opponent.append(discarded_tile)
        env.opponent_hand.remove(discarded_tile)
        draw_ui_opponent_discarded_tiles()
        if hiruchaaru_to_chii or hiruchaaru_to_pon:
            draw_ui_opponent_hand()
            draw_ui_player_discarded_tiles()

        # Check if player can ron with this tile
        temp_hand = env.player_hand.copy()
        temp_hand.append(discarded_tile)
        点数, 役, 和了形 = 点数計算(temp_hand, env.player_seat)
        
        if 和了形:
            button_pass.show()
            button_ron.show()
            player_win_points = 点数
            player_win_yaku = 役
            player_can_call = True

        if len([t for t in env.player_hand if not t.副露]) > 1 and len(env.discard_pile_player) < 54:
            same_tile_count = 0
            for i, tile in enumerate(env.player_hand):
                if (tile.何者, tile.その上の数字) == (discarded_tile.何者, discarded_tile.その上の数字) and not tile.副露:
                    same_tile_count += 1
            if same_tile_count >= 2:
                button_pon.show()
                button_pass.show()
                player_can_call = True

        if len([t for t in env.player_hand if not t.副露]) > 2 and len(env.discard_pile_player) < 54:
            can_chii = False
            player_chii_tiles = []
            
            if "中張牌" in discarded_tile.固有状態:
                dt_num = discarded_tile.その上の数字
                possible_sequence = [(discarded_tile.何者, dt_num-1), (discarded_tile.何者, dt_num+1)] # [n-1,n,n+1]
                check = 0
                for combo in possible_sequence:
                    for p in env.player_hand:
                        if p.何者 == combo[0] and p.その上の数字 == combo[1] and not p.副露:
                            check += 1
                            player_chii_tiles.append(p)
                            break
                assert check <= 2, "No, not happening"
                if check == 2:
                    can_chii = True
                else:
                    player_chii_tiles.clear()
                if can_chii:
                    button_pass.show()
                    button_chii.show()
                    player_can_call = True

        if player_can_call:
            return None
        else:
            if len(env.discard_pile_player) < 54:
                new_tile = env.player_draw_tile()
                点数, 役, 和了形 = 点数計算(env.player_hand, env.player_seat)
                if 和了形:
                    button_pass.show()
                    button_tsumo.show()
                    player_win_points = 点数
                    player_win_yaku = 役
                env.current_actor = 0
                draw_ui_player_hand()
                player_check_discard_what_to_tenpai()
                draw_ui_opponent_hand()
                draw_ui_player_discarded_tiles()
                draw_ui_opponent_discarded_tiles()
            else:
                game_state_text_box.append_html_text("無役終局\n")
                env.game_complete = True
                button_tsumo.hide()
                button_chii.hide()
                button_pon.hide()
                button_ron.hide()
                button_pass.hide()
                player_win_points = 0
                player_win_yaku = []
                for s in player_tile_slots:
                    s.set_tooltip("", delay=0.1)
                draw_ui_opponent_hand()

    def draw_ui_player_labels():
        dic = {0: "東", 1: "南", 2: "西", 3: "北"}
        player_name_label.set_text(f"{dic.get(env.player_seat)} : {env.player_name} : {env.player_points}点")
        opponent_name_label.set_text(f"{dic.get(env.opponent_seat)} : {env.opponent_name} : {env.opponent_points}点")


    def player_win(is_tsumo: bool):
        global player_win_points, player_win_yaku
        env.game_complete = True
        win_type = "ツモ" if is_tsumo else "ロン"
        game_state_text_box.append_html_text(f"{env.player_name}: {win_type}！\n")
        game_state_text_box.append_html_text(f"{env.player_name}: {' '.join(player_win_yaku)}\n")
        game_state_text_box.append_html_text(f"{env.player_name}: {player_win_points}点\n")
        button_tsumo.hide()
        button_chii.hide()
        button_pon.hide()
        button_ron.hide()
        button_pass.hide()
        env.player_points += player_win_points
        env.opponent_points -= player_win_points
        draw_ui_player_labels()
        draw_ui_opponent_hand()
        # Reset
        player_win_points = 0
        player_win_yaku = []
        for s in player_tile_slots:
            s.set_tooltip("", delay=0.1)

    def player_ron():
        player_win(is_tsumo=False)

    def player_tsumo():
        player_win(is_tsumo=True)


    def player_pon():
        if last_tile_discarded_by_opponent:
            for t in env.player_hand:
                if (t.何者, t.その上の数字) == (last_tile_discarded_by_opponent.何者, last_tile_discarded_by_opponent.その上の数字) and not t.副露:
                    t.mark_as_exposed("pon")
                    break
            for t in env.player_hand:
                if (t.何者, t.その上の数字) == (last_tile_discarded_by_opponent.何者, last_tile_discarded_by_opponent.その上の数字) and not t.副露:
                    t.mark_as_exposed("pon")
                    break

            game_state_text_box.append_html_text(f"{env.player_name}: ポン！\n")
            last_tile_discarded_by_opponent.mark_as_exposed("pon")
            env.player_hand.append(last_tile_discarded_by_opponent)
            env.discard_pile_opponent.pop()

            env.current_actor = 0
            button_tsumo.hide()
            button_chii.hide()
            button_pon.hide()
            button_ron.hide()
            button_pass.hide()
            draw_ui_player_hand()
            player_check_discard_what_to_tenpai()
            draw_ui_opponent_hand()
            draw_ui_player_discarded_tiles()
            draw_ui_opponent_discarded_tiles()


    def player_chii():
        if last_tile_discarded_by_opponent:
            for t in env.player_hand:
                if (t.何者, t.その上の数字) == (player_chii_tiles[0].何者, player_chii_tiles[0].その上の数字) and not t.副露:
                    t.mark_as_exposed("chii")
                    break
            for t in env.player_hand:
                if (t.何者, t.その上の数字) == (player_chii_tiles[1].何者, player_chii_tiles[1].その上の数字) and not t.副露:
                    t.mark_as_exposed("chii")
                    break

            game_state_text_box.append_html_text(f"{env.player_name}: チー！\n")
            last_tile_discarded_by_opponent.mark_as_exposed("chii")
            env.player_hand.append(last_tile_discarded_by_opponent)
            env.discard_pile_opponent.pop()

            env.current_actor = 0
            button_tsumo.hide()
            button_chii.hide()
            button_pon.hide()
            button_ron.hide()
            button_pass.hide()
            draw_ui_player_hand()
            player_check_discard_what_to_tenpai()
            draw_ui_opponent_hand()
            draw_ui_player_discarded_tiles()
            draw_ui_opponent_discarded_tiles()



    def player_pass():
        global player_win_points, player_win_yaku
        if env.current_actor == 1:
            env.current_actor = 0
            if len(env.discard_pile_player) < 54:
                new_tile = env.player_draw_tile()
                点数, 役, 和了形 = 点数計算(env.player_hand, env.player_seat)
                if 和了形:
                    button_pass.show()
                    button_tsumo.show()
                    player_win_points = 点数
                    player_win_yaku = 役
                draw_ui_player_hand()
                player_check_discard_what_to_tenpai()
                draw_ui_opponent_hand()
                draw_ui_player_discarded_tiles()
                draw_ui_opponent_discarded_tiles()
            else:
                game_state_text_box.append_html_text("無役終局\n")
                env.game_complete = True

                for s in player_tile_slots:
                    s.set_tooltip("", delay=0.1)
                draw_ui_opponent_hand()
            button_ron.hide()
        elif env.current_actor == 0:
            # Cancelled tsumo
            pass
        player_win_points = 0
        player_win_yaku = []
        button_tsumo.hide()
        button_chii.hide()
        button_pon.hide()
        button_pass.hide()


    # draw_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 100), (150, 15)),
    #                                     text='Draw Tile',
    #                                     manager=ui_manager,
    #                                     tool_tip_text = "Some tool tip text.")
    # discard_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 140), (150, 15)),
    #                                     text='Discard Tile',
    #                                     manager=ui_manager,
    #                                     tool_tip_text = "Some tool tip text.")

    new_game_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 85), (150, 50)),
                                                   text='Start New Game',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.")

    exit_game_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1260, 85), (100, 50)),
                                                   text='Exit',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.")

    def print_accepted_yaku():
        global game_state_text_box
        game_state_text_box.set_text("================\n")
        s = ""
        yaku_data_noyaku = {
            "東": 1000, "南": 1000, "西": 1000, "北": 1000,
            "發": 1000, "中": 1000, "白": 1000,
            "赤ドラ": 1000, # This will be adjusted for count
            "断么九": 1000, "平和": 1000,
            "二槓子": 3000, "三槓子": 6000,
        }
        yaku_data_collection = {
            "混全帯么九": 3000, "純全帯么九": 6000,
            "混一色": 3000,"清一色": 6000, "混老頭": 6000, "清老頭": 32000,
            "字一色|大七星": 32000,
            "緑一色": 32000, "黒一色": 32000,
            "五門斉": 5000,
        }
        yaku_data_k = {
            "対々和": 3000, "三色小同刻": 6000, "三色同刻": 32000,
            "三連刻": 3000, "四連刻": 32000, "三色連刻": 32000,
            "小三風": 3000, "三風刻": 6000, "客風三刻": 32000,
            "四喜和": 32000, "小三元": 6000, "大三元": 32000,
        }
        yaku_data_k1 = {
            "三暗刻": 6000, "四暗刻": 32000,
        }
        yaku_data_s = {
            "三色同順": 6000, "三色三步": 6000,
            "一気通貫": 3000, "三色通貫": 6000,
        }
        yaku_data_s1 = {
            "一盃口": 3000, "二盃口": 6000,
        }
        yaku_data_special = {
            "鏡同和": 6000, 
            "七対子": 3000,
            "国士無双": 32000,
        }

        s += "採用役リスト:\n"
        s += "無役:\n"
        for yaku_name, score in yaku_data_noyaku.items():
            if yaku_name == "赤ドラ":
                s += f"- {yaku_name}: {score}点/枚\n"
            else:
                s += f"- {yaku_name}: {score}点\n"
        s += "何かを揃える役:\n"
        for yaku_name, score in yaku_data_collection.items():
            s += f"- {yaku_name}: {score}点\n"
        s += "刻子役:\n"
        for yaku_name, score in yaku_data_k.items():
            s += f"- {yaku_name}: {score}点\n"
        s += "鳴き不可刻子役:\n"
        for yaku_name, score in yaku_data_k1.items():
            s += f"- {yaku_name}: {score}点\n"
        s += "順子役:\n"
        for yaku_name, score in yaku_data_s.items():
            s += f"- {yaku_name}: {score}点\n"
        s += "鳴き不可順子役:\n"
        for yaku_name, score in yaku_data_s1.items():
            s += f"- {yaku_name}: {score}点\n"
        s += "特殊役:\n"
        for yaku_name, score in yaku_data_special.items():
            s += f"- {yaku_name}: {score}点\n"

        game_state_text_box.append_html_text(s)
        game_state_text_box.append_html_text("================\n")



    accepted_yaku_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1370, 85), (100, 50)),
                                                   text='採用役',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.",
                                                   command=print_accepted_yaku)

    button_tsumo = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((470, 820), (120, 60)),
                                                   text='ツモ',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.",
                                                   command=player_tsumo)

    button_tsumo.hide()

    button_ron = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((500, 820), (120, 60)),
                                                   text='ロン',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.",
                                                   command=player_ron)

    button_ron.hide()
    button_pon = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((630, 820), (120, 60)),
                                                   text='ポン',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.",
                                                   command=player_pon)
    button_pon.hide()

    button_chii = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((760, 820), (120, 60)),
                                                    text='チー',
                                                    manager=ui_manager,
                                                    tool_tip_text="Some tool tip text.",
                                                    command=player_chii)
    button_chii.hide()

    button_pass = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((890, 820), (185, 60)),
                                                   text='パス',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.",
                                                   command=player_pass)
    button_pass.hide()

    agent_selection_list = ["Mixed"]
    # 1.
    agent_files = [f for f in os.listdir("./DQN_agents/") if f.endswith(".pth")]
    agent_state_dicts: dict[str, collections.OrderedDict] = {}
    for filename in agent_files:
        try:
            state_dict: collections.OrderedDict = torch.load(os.path.join("./DQN_agents/", filename), map_location="cuda")
        except Exception as e:
            print(e)
            state_dict: collections.OrderedDict = torch.load(os.path.join("./DQN_agents/", filename), map_location="cpu")
        agent_state_dicts[filename] = state_dict
    # 2.
    agent_preferred_tiles = {}
    with open("./log/agent_summary.csv", mode='r', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader) # Skip header
        # Find the column indices for 'Filename' and 'Top Tiles'
        filename_col_idx = header.index("Filename")
        top_tiles_col_idx = header.index("Top Tiles")

        for row in reader:
            # Ensure the row has enough columns
            if len(row) > max(filename_col_idx, top_tiles_col_idx):
                filename_in_csv = row[filename_col_idx].strip()
                # test_5th_hiruchaaru_1200.csv -> 5th_hiruchaaru_1200.pth
                filename_in_csv = filename_in_csv.replace("test_", "").replace(".csv", ".pth")
                preferred_tiles_str = row[top_tiles_col_idx].strip()
                # Only add preference data for agents whose files were successfully loaded
                if filename_in_csv in agent_state_dicts:
                    agent_preferred_tiles[filename_in_csv] = preferred_tiles_str
                    # print(f" - Read preferences for {filename_in_csv}")
                else:
                    print(f"Warning: Summary for agent file '{filename_in_csv}' not found in. Skipping preference data.")
            else:
                print(f"Warning: Skipping malformed row in ./log/agent_summary.csv: {row}")


    agent_selection_list.extend(agent_files)

    dqn_agent_selection_menu = pygame_gui.elements.UIDropDownMenu(agent_selection_list, "Mixed", 
                                                                  relative_rect=pygame.Rect((1100, 150), (300, 50)),
                                                                  manager=ui_manager)

    reload_hiruchaaru_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1410, 150), (150, 50)),
                                                   text='Reload Agent',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.")


    def start_new_game(reset_points: bool = False, opponent_name: str = ""):
        global player_win_points, player_win_yaku
        env.start_new_game(opponent_name)
        if opponent_name == "Mixed":
            # Then decide which agent to assign to this game
            best_score = -1
            selected_agent_filename = None
            # Calculate preference score for each agent's preferred tiles against the current hand
            for filename, preferred_tiles_string in agent_preferred_tiles.items():
                # Check if the agent's state dict was actually loaded
                if filename in agent_state_dicts:
                    score = calculate_weighted_preference_score(env.opponent_hand, preferred_tiles_string, 1)
                    if score > best_score:
                        best_score = score
                        selected_agent_filename = filename

            # Load the state dict of the selected agent
            if selected_agent_filename:
                dqn_agent.model.load_state_dict(agent_state_dicts[selected_agent_filename])
                # print(f"Episode {ep+1}: Selected agent {selected_agent_filename} (Score: {best_score})")
            elif agent_state_dicts:
                print(f"Warning: No agent had a positive preference score for hand. Selecting the first loaded agent.")
                selected_agent_filename = list(agent_state_dicts.keys())[0]
                dqn_agent.model.load_state_dict(agent_state_dicts[selected_agent_filename])
            else:
                raise Exception
        else:
            dqn_agent.model.load_state_dict(agent_state_dicts[f"{opponent_name}.pth"])

        new_tile = env.player_draw_tile()
        draw_ui_player_hand()
        player_check_discard_what_to_tenpai()
        draw_ui_opponent_hand()
        draw_ui_player_discarded_tiles()
        draw_ui_opponent_discarded_tiles()
        game_state_text_box.set_text("")
        if reset_points:
            env.player_points = env.starting_points
            env.opponent_points = env.starting_points
        draw_ui_player_labels()
        button_tsumo.hide()
        button_chii.hide()
        button_pon.hide()
        button_ron.hide()
        button_pass.hide()
        a00, b00, c00 = 点数計算(env.player_hand, env.player_seat)
        if c00:
            button_pass.show()
            button_tsumo.show()
            player_win_points = a00
            player_win_yaku = b00


    start_new_game(reset_points=True, opponent_name=dqn_agent_selection_menu.selected_option[0].split('.')[0])
    draw_ui_player_labels()

    while running:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                pass
                            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = True
                if event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL:
                    ctrl_held = True
                if event.key == pygame.K_s:
                    s_key_held = True
                if event.key == pygame.K_q:
                    q_key_held = True
                if event.key == pygame.K_w:
                    w_key_held = True
                if event.key == pygame.K_e:
                    e_key_held = True
                if event.key == pygame.K_r:
                    r_key_held = True
                if event.key == pygame.K_t:
                    t_key_held = True
                if event.key == pygame.K_1:
                    one_key_held = True
                if event.key == pygame.K_2:
                    two_key_held = True
                if event.key == pygame.K_3:
                    three_key_held = True
                if event.key == pygame.K_4:
                    four_key_held = True
                if event.key == pygame.K_5:
                    five_key_held = True
                if event.key == pygame.K_6:
                    six_key_held = True
                if event.key == pygame.K_7:
                    seven_key_held = True
                if event.key == pygame.K_8:
                    eight_key_held = True
                if event.key == pygame.K_9:
                    nine_key_held = True
                if event.key == pygame.K_0:
                    zero_key_held = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    shift_held = False
                if event.key == pygame.K_LCTRL or event.key == pygame.K_RCTRL:
                    ctrl_held = False
                if event.key == pygame.K_s:
                    s_key_held = False
                if event.key == pygame.K_q:
                    q_key_held = False
                if event.key == pygame.K_w:
                    w_key_held = False
                if event.key == pygame.K_e:
                    e_key_held = False
                if event.key == pygame.K_r:
                    r_key_held = False
                if event.key == pygame.K_t:
                    t_key_held = False
                if event.key == pygame.K_1:
                    one_key_held = False
                if event.key == pygame.K_2:
                    two_key_held = False
                if event.key == pygame.K_3:
                    three_key_held = False
                if event.key == pygame.K_4:
                    four_key_held = False
                if event.key == pygame.K_5:
                    five_key_held = False
                if event.key == pygame.K_6:
                    six_key_held = False
                if event.key == pygame.K_7:
                    seven_key_held = False
                if event.key == pygame.K_8:
                    eight_key_held = False
                if event.key == pygame.K_9:
                    nine_key_held = False
                if event.key == pygame.K_0:
                    zero_key_held = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # character selection and party member swap
                for index, image_slot in enumerate(player_tile_slots):
                    if image_slot.rect.collidepoint(event.pos) and env.current_actor == 0 and not image_slot.set_temp_marked and not env.game_complete:
                        if not env.player_hand[index].副露:
                            # discard the tile
                            tile = env.player_hand.pop(index)
                            env.discard_pile_player.append(tile)
                            last_tile_discarded_by_player = tile
                            draw_ui_player_hand()
                            draw_ui_player_discarded_tiles()
                            button_tsumo.hide()
                            button_pass.hide()
                            progress_opponent_turn()
                        else:
                            pass
                            # print("You can not discard an exposed tile!")


            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                selected_opponent = dqn_agent_selection_menu.selected_option[0].split('.')[0]
                if event.ui_element == new_game_button:
                    start_new_game(reset_points=False, opponent_name=selected_opponent)
                if event.ui_element == reload_hiruchaaru_button:
                    start_new_game(reset_points=True, opponent_name=selected_opponent)
                if event.ui_element == exit_game_button:
                    running = False

            if event.type == pygame_gui.UI_TEXT_BOX_LINK_CLICKED:
                pass

            if event.type == pygame_gui.UI_DROP_DOWN_MENU_CHANGED:
                pass

            ui_manager_lower.process_events(event)
            ui_manager.process_events(event)
            ui_manager_overlay.process_events(event)

        ui_manager_lower.update(time_delta)
        ui_manager.update(time_delta)
        ui_manager_overlay.update(time_delta)
        display_surface.fill(light_purple)
        # if global_vars.theme == "Yellow Theme":
        #     display_surface.fill(light_yellow)
        # elif global_vars.theme == "Purple Theme":
        #     display_surface.fill(light_purple)
        # elif global_vars.theme == "Red Theme":
        #     display_surface.fill(light_red)
        # elif global_vars.theme == "Green Theme":
        #     display_surface.fill(light_green)
        # elif global_vars.theme == "Blue Theme":
        #     display_surface.fill(light_blue)
        # elif global_vars.theme == "Pink Theme":
        #     display_surface.fill(light_pink)
        ui_manager_lower.draw_ui(display_surface)
        ui_manager.draw_ui(display_surface)
        ui_manager_overlay.draw_ui(display_surface)

        pygame.display.update()

    pygame.quit()