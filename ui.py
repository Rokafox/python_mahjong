from collections import Counter
from copy import copy
import random
import numpy as np
import pygame, pygame_gui
import pygame.freetype
import sys
import os
from typing import List, Dict, Tuple, Optional

import torch
from ろかAI import DQNAgent
from ろか麻雀 import nicely_print_tiles, 基礎訓練山を作成する, 山を作成する, 点数計算, 聴牌ですか, 麻雀牌



class Env:
    """
    Mahjong game env
    """
    N_TILE_TYPES = 34               # 萬・筒・索 (9*3) + 字牌 7
    STATE_BITS = 5                 # 0,1,2,3,4 枚の one‑hot

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

    def _get_state(self, ndtidx: int = -1) -> np.ndarray:
        """
        ntidx: 他家がさき捨てた牌を記録したい時用
        """
        state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS, dtype=np.float32)
        counter = Counter(self._tile_to_index(t) for t in self.opponent_hand)
        # Counter({22: 2, 0: 1, 1: 1, 2: 1, 9: 1})
        for idx in range(self.N_TILE_TYPES):
            cnt = counter.get(idx, 0)
            state[idx * self.STATE_BITS + cnt] = 1.0  # 0〜4
        seat = np.zeros(4, dtype=np.float32) # 東南西北
        seat[self.opponent_seat] = 1.0
        discarded_tiles_counts = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        for tile in self.discard_pile_opponent:
            discarded_tiles_counts[self._tile_to_index(tile)] += 1.0
        discarded_tiles_counts /= 4.0  # 正規化（最大4枚）
        # NOTE: 他家の副露も捨て牌として考えればよい
        new_discarded_tile_by_others = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        if ndtidx >= 0:
            new_discarded_tile_by_others[ndtidx] = 1.0
        # 170 + 4 + 34 + 34 = 242次元
        return np.concatenate([state, seat, discarded_tiles_counts, new_discarded_tile_by_others])


    def _tile_to_index(self, tile):
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
    

    def _index_to_tile_type(self, idx):
        """Convert index to tile type (for action selection)"""
        if 0 <= idx < 9:
            return ("萬子", idx + 1)
        elif 9 <= idx < 18:
            return ("筒子", idx - 9 + 1)
        elif 18 <= idx < 27:
            return ("索子", idx - 18 + 1)
        elif idx == 27:
            return ("東風", 0)
        elif idx == 28:
            return ("南風", 0)
        elif idx == 29:
            return ("西風", 0)
        elif idx == 30:
            return ("北風", 0)
        elif idx == 31:
            return ("白ちゃん", 0)
        elif idx == 32:
            return ("發ちゃん", 0)
        elif idx == 33:
            return ("中ちゃん", 0)


    def get_valid_actions(self, extra_tile_idx: int, tenpai: bool=False) -> list[int]:
        if self.current_actor == 1: # Opponent
            # 行動：牌を捨てる
            # 聴牌の時、もらった牌を捨てるだけにしとこう
            if tenpai and extra_tile_idx >= 0:
                return [extra_tile_idx]
            hand_tiles = [self._tile_to_index(t) for t in self.opponent_hand if not t.副露]
            if extra_tile_idx >= 0:
                hand_tiles.append(extra_tile_idx)
            hand_tiles = sorted(set(hand_tiles))
            return hand_tiles
        else:
            # 行動：何もしない・ポン・チー
            # 34, 35, 36
            return [34, 35, 36]

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
    # debug_ui_manager = pygame_gui.UIManager((1600, 900), "theme_light_yellow.json", starting_language='ja')
    # ui_manager.get_theme().load_theme("theme_light_purple.json")
    # ui_manager.rebuild_all_from_changed_theme_data()

    pygame.display.set_caption("Roka Mahjong")
    # if there is icon, use it
    try:
        icon = pygame.image.load("icon.png")
        pygame.display.set_icon(icon)
    except Exception as e:
        print(f"Error loading icon: {e}")

    dqn_agent = DQNAgent(242, 34 + 3, device="cuda")
    dqn_agent.model.load_state_dict(torch.load("./DQN_agents/Agent.pth", map_location="cuda"))
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
        tile_size=(66, 90),
        count=20,
        spacing=4
    )

    opponent_discarded_tiles_group_a = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 420 - 66),
        tile_size=(66, 90),
        count=20,
        spacing=4
    )

    player_discarded_tiles_group_b = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 480 + 94),
        tile_size=(66, 90),
        count=20,
        spacing=4
    )

    opponent_discarded_tiles_group_b = create_tile_slots(
        ui_manager=ui_manager,
        start_pos=(100, 420 - 66 - 94),
        tile_size=(66, 90),
        count=20,
        spacing=4
    )

    player_name_label = pygame_gui.elements.UILabel(pygame.Rect((100, 800), (200, 50)),
                                                    text="Player",
                                                    manager=ui_manager)
    opponent_name_label = pygame_gui.elements.UILabel(pygame.Rect((100, 50), (200, 50)),
                                                    text="Opponent",
                                                    manager=ui_manager)

    game_state_text_box = pygame_gui.elements.UITextEntryBox(pygame.Rect((1100, 700), (480, 180)),"", ui_manager)


    env = Env()

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
                player_tile_slots[i].set_image(pygame.image.load("asset/405.png"))
                player_tile_slots[i].set_temp_marked = True


    def draw_ui_player_discarded_tiles():
        for uiimage in player_discarded_tiles_group_a + player_discarded_tiles_group_b:
            uiimage.set_image(pygame.image.load("asset/405.png"))
        for i, t in enumerate(env.discard_pile_player):
            image_path = t.get_asset()
            try:
                image_surface = pygame.image.load(image_path)
                if i < 20:
                    player_discarded_tiles_group_a[i].set_image(image_surface)
                elif 20 <= i < 40:
                    player_discarded_tiles_group_b[i - 20].set_image(image_surface)
                else:
                    raise Exception("Too many discarded tiles")
            except pygame.error as e:
                print(f"Error loading image {image_path}: {e}")


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
                opponent_tile_slots[i].set_image(pygame.image.load("asset/405.png"))
                opponent_tile_slots[i].set_temp_marked = True



    def draw_ui_opponent_discarded_tiles():
        for uiimage in opponent_discarded_tiles_group_a + opponent_discarded_tiles_group_b:
            uiimage.set_image(pygame.image.load("asset/405.png"))
        for i, t in enumerate(env.discard_pile_opponent):
            image_path = t.get_asset()
            try:
                image_surface = pygame.image.load(image_path)
                if i < 20:
                    opponent_discarded_tiles_group_a[i].set_image(image_surface)
                elif 20 <= i < 40:
                    opponent_discarded_tiles_group_b[i - 20].set_image(image_surface)
                else:
                    raise Exception("Too many discarded tiles")
            except pygame.error as e:
                print(f"Error loading image {image_path}: {e}")


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


    def start_new_game(reset_points: bool = False, opponent_name: str = ""):
        global player_win_points, player_win_yaku
        env.start_new_game(opponent_name)
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

        # Opponent Ron
        temp_opponent_hand = env.opponent_hand.copy()
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


        # Check if opponent can pon or chii
        hiruchaaru_to_pon: bool = False
        hiruchaaru_to_chii: bool = False
        # pon
        if len([t for t in env.opponent_hand if not t.副露]) > 1:
            same_tile_count = 0
            same_tiles = []
            for i, tile in enumerate(env.opponent_hand):
                if (tile.何者, tile.その上の数字) == (last_tile_discarded_by_player.何者, last_tile_discarded_by_player.その上の数字) and not tile.副露:
                    same_tile_count += 1
                    same_tiles.append(i)
            if same_tile_count >= 2:
                # can pon
                valid_actions: list = env.get_valid_actions(-1)
                valid_actions.remove(36)
                assert 34 in valid_actions and 35 in valid_actions, "34 do nothing 35 pon 36 chii"
                action: int
                action, full_dict = dqn_agent.act(env._get_state(ndtidx=env._tile_to_index(last_tile_discarded_by_player)), valid_actions)
                if action == 35:
                    game_state_text_box.append_html_text(f"{env.opponent_name}: ポン！\n")
                    # Mark the 2 tiles in agent's hand as 副露
                    for i in sorted(same_tiles[:2], reverse=True):
                        env.opponent_hand[i].mark_as_exposed()
                    
                    # Add the discarded tile to agent's hand and mark it as 副露
                    last_tile_discarded_by_player.mark_as_exposed()
                    env.opponent_hand.append(last_tile_discarded_by_player)
                    
                    # Remove the discarded tile from the discard pile
                    env.discard_pile_player.pop()
                    hiruchaaru_to_pon = True


                else: # action 34
                    pass
        # chii
        if len([t for t in env.opponent_hand if not t.副露]) > 2:
            can_chii = False
            chii_tiles = []
            
            if "中張牌" in last_tile_discarded_by_player.固有状態:
                dt_num = last_tile_discarded_by_player.その上の数字
                discarded_tile_idx = env._tile_to_index(last_tile_discarded_by_player)
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
                    valid_actions: list = env.get_valid_actions(-1)
                    valid_actions.remove(35)
                    assert 34 in valid_actions and 36 in valid_actions, "34 do nothing 35 pon 36 chii"
                    action: int
                    action, full_dict = dqn_agent.act(env._get_state(ndtidx=discarded_tile_idx), valid_actions)
                    if action == 36:
                        game_state_text_box.append_html_text(f"{env.opponent_name}: チー！\n")
                        for t in env.opponent_hand:
                            if (t.何者, t.その上の数字) == (chii_tiles[0].何者, chii_tiles[0].その上の数字) and not t.副露:
                                t.mark_as_exposed()
                                break
                        for t in env.opponent_hand:
                            if (t.何者, t.その上の数字) == (chii_tiles[1].何者, chii_tiles[1].その上の数字) and not t.副露:
                                t.mark_as_exposed()
                                break

                        last_tile_discarded_by_player.mark_as_exposed()
                        env.opponent_hand.append(last_tile_discarded_by_player)
                        env.discard_pile_player.pop()
                        hiruchaaru_to_chii = True
                    else:
                        pass


        env.current_actor = 1
        env.turn += 1
        newly_drawn_tile_index: int = -1
        tenpai_before_draw: bool = False

        if not hiruchaaru_to_pon and not hiruchaaru_to_chii:
            if len(env.discard_pile_opponent) < 40:
                # Opponent draw a tile
                tenpai_before_draw, list_of_tanpai = 聴牌ですか(env.opponent_hand, env.opponent_seat)
                new_tile = env.opponent_draw_tile()
                newly_drawn_tile_index = env._tile_to_index(new_tile)
                

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
                return None
        else:
            # Hiruchaaru poned or chiied, so they do not draw, but need to discard.
            pass

        valid_actions: list = env.get_valid_actions(extra_tile_idx=newly_drawn_tile_index,
                                                     tenpai=tenpai_before_draw)
        assert len(valid_actions) > 0
        action: int
        action, full_dict = dqn_agent.act(env._get_state(), valid_actions)
        # print(f"Opponent action: {action}")
        # print(env._index_to_tile_type(action))discarded_tile
        # print(tile_type) # ('筒子', 5)
        discarded_tile = None
        for t in [t for t in env.opponent_hand if not t.副露]:
            if (t.何者, t.その上の数字) == env._index_to_tile_type(action):
                discarded_tile = t
                break
        if discarded_tile is None:
            raise ValueError(f"Invalid action: {env._index_to_tile_type(action)} not in hand.")
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

        if len([t for t in env.player_hand if not t.副露]) > 1:
            same_tile_count = 0
            for i, tile in enumerate(env.player_hand):
                if (tile.何者, tile.その上の数字) == (discarded_tile.何者, discarded_tile.その上の数字) and not tile.副露:
                    same_tile_count += 1
            if same_tile_count >= 2:
                button_pon.show()
                button_pass.show()
                player_can_call = True

        if len([t for t in env.player_hand if not t.副露]) > 2:
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
            if len(env.discard_pile_player) < 40:
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

    def draw_ui_player_labels():
        player_name_label.set_text(f"{env.player_name} : {env.player_points}点")
        opponent_name_label.set_text(f"{env.opponent_name} : {env.opponent_points}点")


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
        for t in env.player_hand:
            if (t.何者, t.その上の数字) == (last_tile_discarded_by_opponent.何者, last_tile_discarded_by_opponent.その上の数字) and not t.副露:
                t.mark_as_exposed()
                break
        for t in env.player_hand:
            if (t.何者, t.その上の数字) == (last_tile_discarded_by_opponent.何者, last_tile_discarded_by_opponent.その上の数字) and not t.副露:
                t.mark_as_exposed()
                break

        game_state_text_box.append_html_text(f"{env.player_name}: ポン！\n")
        last_tile_discarded_by_opponent.mark_as_exposed()
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
        for t in env.player_hand:
            if (t.何者, t.その上の数字) == (player_chii_tiles[0].何者, player_chii_tiles[0].その上の数字) and not t.副露:
                t.mark_as_exposed()
                break
        for t in env.player_hand:
            if (t.何者, t.その上の数字) == (player_chii_tiles[1].何者, player_chii_tiles[1].その上の数字) and not t.副露:
                t.mark_as_exposed()
                break

        game_state_text_box.append_html_text(f"{env.player_name}: チー！\n")
        last_tile_discarded_by_opponent.mark_as_exposed()
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
            if len(env.discard_pile_player) < 40:
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


    # draw_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 100), (150, 35)),
    #                                     text='Draw Tile',
    #                                     manager=ui_manager,
    #                                     tool_tip_text = "Some tool tip text.")
    # discard_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 140), (150, 35)),
    #                                     text='Discard Tile',
    #                                     manager=ui_manager,
    #                                     tool_tip_text = "Some tool tip text.")

    new_game_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 100), (150, 35)),
                                                   text='Start New Game',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.")

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


    agent_list = [f for f in os.listdir("./DQN_agents") if os.path.isfile(os.path.join("./DQN_agents", f))]
    dqn_agent_selection_menu = pygame_gui.elements.UIDropDownMenu(agent_list, agent_list[0], 
                                                                  relative_rect=pygame.Rect((1100, 150), (150, 35)),
                                                                  manager=ui_manager)

    reload_hiruchaaru_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1260, 150), (150, 35)),
                                                   text='Reload Agent',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.")



    start_new_game(reset_points=True, opponent_name=dqn_agent_selection_menu.selected_option[0].split('.')[0])
    draw_ui_player_labels()

    while running:
        time_delta = clock.tick(60)/1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # build_quit_game_window()
                running = False
            # right click to deselect from inventory
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
                if event.ui_element == new_game_button:
                    start_new_game(reset_points=False, opponent_name=dqn_agent_selection_menu.selected_option[0].split('.')[0])
                if event.ui_element == reload_hiruchaaru_button:
                    env.opponent_name = dqn_agent_selection_menu.selected_option[0].split('.')[0]
                    agent_path = "./DQN_agents/" + dqn_agent_selection_menu.selected_option[0]
                    dqn_agent.model.load_state_dict(torch.load(agent_path, map_location="cpu"))
                    start_new_game(reset_points=True, opponent_name=dqn_agent_selection_menu.selected_option[0].split('.')[0])

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