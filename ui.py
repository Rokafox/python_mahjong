import pygame, pygame_gui
import pygame.freetype
import sys
import os
from typing import List, Dict, Tuple, Optional
from ろか麻雀 import 山を作成する, 麻雀牌



class Env:
    """
    Mahjong game env
    """
    def __init__(self):
        self.山: list[麻雀牌] = []
        self.player_hand: list[麻雀牌] = []
        self.opponent_hand: list[麻雀牌] = []
        self.turn: int = 1
        self.current_actor: int = 0 # 0: player, 1: opponent
        self.discard_pile_player = []
        self.discard_pile_opponent = []

    def start_new_game(self):
        self.generate_pile()
        # give 13 tiles for each player
        self.player_hand = self.山[:13]
        self.opponent_hand = self.山[13:26]
        self.player_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        self.opponent_hand.sort(key=lambda x: (x.sort_order, x.その上の数字))
        assert len(self.player_hand) == len(self.opponent_hand) == 13
        self.turn = 1
        self.current_actor = 0
        self.discard_pile_player = []
        self.discard_pile_opponent = []

    def generate_pile(self):
        self.山 = 山を作成する()

    def player_draw_tile(self) -> Optional[麻雀牌]:
        if len(self.山) == 0:
            return None
        tile = self.山.pop(0)
        self.player_hand.append(tile)
        return tile

    def player_discard_tile(self, index: int):
        if index < 0 or index >= len(self.player_hand):
            raise ValueError("Invalid index")
        tile = self.player_hand.pop(index)
        self.discard_pile_player.append(tile)
        self.discard_pile_player.sort(key=lambda x: (x.sort_order, x.その上の数字))
        return tile



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
    ui_manager_lower = pygame_gui.UIManager((1600, 900), "theme_light_yellow.json", starting_language='ja')
    ui_manager = pygame_gui.UIManager((1600, 900), "theme_light_yellow.json", starting_language='ja')
    ui_manager_overlay = pygame_gui.UIManager((1600, 900), "theme_light_yellow.json", starting_language='ja')
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

    env = Env()

    def draw_ui_player_hand():
        for i, t in enumerate(env.player_hand):
            image_path = t.get_asset()
            try:
                image_surface = pygame.image.load(image_path)
                player_tile_slots[i].set_temp_marked = False
                player_tile_slots[i].set_image(image_surface)
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
                player_discarded_tiles_group_a[i].set_image(image_surface)
            except pygame.error as e:
                print(f"Error loading image {image_path}: {e}")


    def draw_ui_opponent_hand():
        for i, t in enumerate(env.opponent_hand):
            image_path = t.get_asset()
            try:
                image_surface = pygame.image.load(image_path)
                opponent_tile_slots[i].set_temp_marked = False
                opponent_tile_slots[i].set_image(image_surface)
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
                opponent_discarded_tiles_group_a[i].set_image(image_surface)
            except pygame.error as e:
                print(f"Error loading image {image_path}: {e}")


    def start_new_game():
        env.start_new_game()
        new_tile = env.player_draw_tile()
        draw_ui_player_hand()
        draw_ui_opponent_hand()
        draw_ui_player_discarded_tiles()
        draw_ui_opponent_discarded_tiles()


    start_new_game()

    draw_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 100), (150, 35)),
                                        text='Draw Tile',
                                        manager=ui_manager,
                                        tool_tip_text = "Some tool tip text.")
    discard_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 140), (150, 35)),
                                        text='Discard Tile',
                                        manager=ui_manager,
                                        tool_tip_text = "Some tool tip text.")
    new_game_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((1100, 180), (150, 35)),
                                                   text='Start New Game',
                                                   manager=ui_manager,
                                                   tool_tip_text="Some tool tip text.",
                                                   command=start_new_game)

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
                    if image_slot.rect.collidepoint(event.pos) and env.current_actor == 0 and not image_slot.set_temp_marked:
                        # discard the tile
                        env.player_discard_tile(index)
                        draw_ui_player_hand()
                        draw_ui_player_discarded_tiles()
                        


            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                pass

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
        display_surface.fill(light_yellow)
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