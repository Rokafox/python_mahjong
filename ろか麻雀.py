from collections import Counter
from copy import deepcopy
from itertools import combinations, permutations
import linecache
import math
import random
import re

class 麻雀牌:
    def __init__(self, 何者: str, その上の数字: int, 赤ドラ: bool = False, 副露: bool = False):
        self.何者: str = 何者
        self.その上の数字: int = その上の数字
        self.赤ドラ: bool = 赤ドラ
        self.固有状態: list[str] = []
        self.アクティブ状態: list[str] = []
        self.ドラ: bool = False
        self.副露: bool = 副露
        self.exposed_state: str = ""
        self.marked_a: bool = False # mark the tile for temporary use
        self.marked_b: bool = False
        self.marked_c: bool = False
        self.marked_as_removed: bool = False
        if not self.固有状態追加と検証():
            raise Exception(f"不正な牌: {何者}, {その上の数字}")
        self.sort_order = self.set_sort_order()
        self.init最後の処理()

    def __str__(self):
        表示名 = ""
        表示名 += f"{self.何者}"
        if self.その上の数字 > 0:
            表示名 += f"{self.その上の数字}"
        if self.赤ドラ:
            表示名 += "赤ドラ"
        return 表示名

    def 固有状態追加と検証(self) -> bool:
        if self.何者 in ["東風", "南風", "西風", "北風"]:
            self.固有状態.extend(["四風牌", "字牌", "么九牌"])
            return self.その上の数字 == 0
        elif self.何者 in ["筒子", "萬子", "索子"]:
            self.固有状態.extend(["数牌"])
            # 老頭牌1,9
            if self.その上の数字 in [1, 9]:
                self.固有状態.extend(["老頭牌", "么九牌"])
            # 中張牌2,3,4,5,6,7,8
            elif self.その上の数字 in [2, 3, 4, 5, 6, 7, 8]:
                self.固有状態.extend(["中張牌"])
            return 1 <= self.その上の数字 <= 9
        elif self.何者 in ["白ちゃん", "發ちゃん", "中ちゃん"]:
            self.固有状態.extend(["三元牌", "字牌", "么九牌"])
            return self.その上の数字 == 0
        else:
            return False
        
    def init最後の処理(self):
        self.固有状態.sort()
        return None

    def set_sort_order(self):
        if self.何者 == "萬子":
            return 0
        elif self.何者 == "筒子":
            return 1
        elif self.何者 == "索子":
            return 2
        elif self.何者 == "東風":
            return 3
        elif self.何者 == "南風":
            return 4
        elif self.何者 == "西風":
            return 5
        elif self.何者 == "北風":
            return 6
        elif self.何者 == "白ちゃん":
            return 7
        elif self.何者 == "發ちゃん":
            return 8
        elif self.何者 == "中ちゃん":
            return 9
        else:
            raise Exception(f"不正な牌: {self.何者}")
        
    def mark_as_exposed(self, es: str):
        """
        es:ぽん？それともちー？
        """
        assert es in ["pon", "chii"]
        if self.副露:
            raise Exception(f"牌はすでに副露されています: {self.何者}, {self.その上の数字}")
        self.副露 = True
        self.exposed_state = es
        return None

    def get_exposed_status(self):
        if not self.副露:
            return 0
        else:
            if self.exposed_state == "pon":
                return 1
            elif self.exposed_state == "chii":
                return 2
        raise Exception

    def get_asset(self, yoko: bool = False) -> str:
        """
        牌に対応する PNG ファイルのパスを返す。
        yoko=Trueで横向き画像 (-yoko 付き) を返す。
        赤ドラ (`self.赤ドラ=True`) のときは aka1/2/3 を優先して返す。
        """
        base_dir = "asset/tiles"
        suffix   = "-yoko" if yoko else ""
        
        # ── 赤ドラ牌 ───────────────────────────────
        if self.赤ドラ and self.その上の数字 == 5:
            aka_map = {"筒子": "aka1",  # pin5
                       "索子": "aka2",  # sou5
                       "萬子": "aka3"}  # man5
            aka_code = aka_map.get(self.何者)
            if aka_code:  # 想定外の組み合わせは通常牌扱い
                return f"{base_dir}/{aka_code}-66-90-s{suffix}.png"

        # ── 数牌 (萬子・筒子・索子) ─────────────────
        if self.何者 in {"萬子", "筒子", "索子"}:
            suit_map = {"萬子": "man", "筒子": "pin", "索子": "sou"}
            suit_code = suit_map[self.何者]
            return f"{base_dir}/{suit_code}{self.その上の数字}-66-90-s{suffix}.png"

        # ── 字牌 (風牌・三元牌) ─────────────────────
        honor_map = {
            "東風": 1, "南風": 2, "西風": 3, "北風": 4,
            "發ちゃん": 5, "白ちゃん": 6, "中ちゃん": 7
        }
        honor_idx = honor_map.get(self.何者)
        if honor_idx:
            return f"{base_dir}/ji{honor_idx}-66-90-s{suffix}.png"

        # ── ここに来る場合は未対応 ─────────────────
        raise ValueError(f"アセットファイルが見つかりません: {self}")



def calculate_weighted_preference_score(tiles_list: list[麻雀牌], input_string: str, agent_score: int) -> float:
    """
    Given a list of Mahjong tiles and a string of preferred tiles with scores,
    calculates a weighted preference score for the tiles_list.
    The input_string is expected to be in the format "Tile Name (Score), Tile Name (Score), ...".
    The score for each tile in the tiles_list that matches a preferred tile
    in the input_string is added to the total weighted score.
    Then, agent have its own score, ie 2274 ie 1542
    """
    agent_score = int(float(agent_score))
    prefered_tiles_with_scores = {}
    # Use regex to find all tile name and score pairs
    matches = re.findall(r"([^,]+?)\s*\((\d+)\)", input_string)

    for match in matches:
        tile_name = match[0].strip()
        score = int(match[1])
        prefered_tiles_with_scores[tile_name] = score

    total_weighted_score = 0
    for tile in tiles_list:
        tile_str = str(tile)
        if tile_str in prefered_tiles_with_scores:
            total_weighted_score += prefered_tiles_with_scores[tile_str]

    # Use square root or other scaling to reduce impact
    scaled_agent_score = math.sqrt(agent_score)
    return total_weighted_score * scaled_agent_score


# prefered_input_string = "筒子2 (9860), 筒子3 (8371), 筒子8 (6842), 筒子9 (6299), 筒子6 (4974), 筒子4 (4690), 筒子1 (4655), 筒子7 (4572), 筒子5 (4256), 索子6 (1644), 筒子5赤ドラ (1316), 索子7 (1074), 南風 (901), 索子5 (876), 索子4 (775), 萬子7 (751), 索子3 (701), 東風 (600), 萬子6 (384), 白ちゃん (381)"
# hand_tiles = [
# 麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False), 麻雀牌("萬子", 3, False),  
# ]
# weighted_score = calculate_weighted_preference_score(hand_tiles, prefered_input_string)
# print(f"The weighted preference score for the hand is: {weighted_score}")


def nicely_print_tiles(tiles: list[麻雀牌], sortit: bool = True) -> str:
    """
    牌のリストを整形して表示する
    """
    output = ""
    exposed_tiles = [t for t in tiles if t.副露]
    if sortit:
        exposed_tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    not_exposed_tiles = [t for t in tiles if not t.副露]
    if sortit:
        not_exposed_tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    r0 = []
    for tile in not_exposed_tiles:
        r0.append(str(tile))
    output += " ".join(r0)
    output += " | "
    r1 = []
    for tile in exposed_tiles:
        r1.append(str(tile))
    output += " ".join(r1)
    return output




# ==========================
# 山
# ==========================

def 山を作成する() -> list[麻雀牌]:
    """
    """
    山: list[麻雀牌] = []
    # 数牌（萬子、筒子、索子）各1〜9を4枚ずつ
    for 何者 in ["萬子", "筒子", "索子"]:
        for 数 in range(1, 5):
            for _ in range(4):
                山.append(麻雀牌(何者, 数, 赤ドラ=False))
        for 数 in range(6, 10):
            for _ in range(4):
                山.append(麻雀牌(何者, 数, 赤ドラ=False))
        for _ in range(3):
            山.append(麻雀牌(何者, 5, 赤ドラ=False))
        山.append(麻雀牌(何者, 5, 赤ドラ=True))

    # 字牌：風牌
    for 何者 in ["東風", "南風", "西風", "北風"]:
        for _ in range(4):
            山.append(麻雀牌(何者, 0, 赤ドラ=False))

    # 字牌：三元牌
    for 何者 in ["白ちゃん", "發ちゃん", "中ちゃん"]:
        for _ in range(4):
            山.append(麻雀牌(何者, 0, 赤ドラ=False))
    
    random.shuffle(山)

    assert len(山) == 136, f"山の長さが不正です: {len(山)}"
    return 山


# ==========================
# 補助関数
# ==========================
def 面子スコア(tiles: list[麻雀牌]) -> int:
    """
    13 枚の手牌から完成面子（順子・刻子）の最大数を求めて
    0面子→0, 1面子→1, 2面子→2, 3面子→4, 4面子→8を返す。雀頭は数えない。
    """
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))

    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    memo: dict[tuple[tuple[tuple[str, int], int], ...], int] = {}
    数牌 = ("萬子", "筒子", "索子")

    def dfs(c: Counter) -> int:
        """残りカウンタ c から作れる最大面子数を返す（メモ化付き）"""
        key = tuple(sorted((k, v) for k, v in c.items() if v))
        if not key:             # 牌が残っていない
            return 0
        if key in memo:         # メモ化
            return memo[key]
        best = 0
        # すべての牌を起点に「刻子」「順子」を試す
        for (suit, num), cnt in list(c.items()):
            if cnt == 0:
                continue

            # ── 刻子 ──
            if cnt >= 3:
                c[(suit, num)] -= 3
                best = max(best, 1 + dfs(c))
                c[(suit, num)] += 3

            # ── 順子 ──
            if suit in 数牌 and num <= 7:
                k1, k2 = (suit, num + 1), (suit, num + 2)
                if c[k1] and c[k2]:
                    c[(suit, num)] -= 1
                    c[k1]          -= 1
                    c[k2]          -= 1
                    best = max(best, 1 + dfs(c))
                    c[(suit, num)] += 1
                    c[k1]          += 1
                    c[k2]          += 1
        memo[key] = best
        return best
    
    melds = dfs(counter)                # 0〜4
    return [0, 1, 2, 4, 8][melds]


def 刻子スコア(tiles: list[麻雀牌], allowed_num: list[int] | None = None,
        score_table : list[int] = [0, 1, 2, 4, 8]) -> int:
    """
    13 枚の手牌から刻子（同じ牌3枚）の最大数を求めて
    0刻子→0, 1刻子→1, 2刻子→2, 3刻子→4, 4刻子→8を返す。雀頭は数えない。

    allowed_num: 
    """
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    
    # 牌の種類ごとのカウントを作成
    if allowed_num:
        counter = Counter((t.何者, t.その上の数字) for t in tiles if t.その上の数字 in allowed_num)
    else:
        counter = Counter((t.何者, t.その上の数字) for t in tiles)
    
    # 刻子の数をカウント
    刻子数 = sum(1 for count in counter.values() if count >= 3)
    
    # スコア変換テーブル
    return score_table[刻子数]


# 手牌 = [ 
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 
#     麻雀牌("萬子", 2, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 2, False),  
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),        
# ]
# print(刻子スコア(手牌, [1, 9, 0]))


def 順子スコア(tiles: list[麻雀牌], allowed_sequences: list[list[int]] | None = None,
        score_table : list[int] = [0, 1, 2, 4, 8]) -> int:
    """
    13 枚の手牌から順子（連続する3枚の数牌）の最大数を求めて
    0順子→0, 1順子→1, 2順子→2, 3順子→4, 4順子→8を返す。雀頭は数えない。
    
    allowed_sequences: 有効な順子の組み合わせを指定する（例：[[1, 2, 3], [4, 5, 6]]）
    指定がない場合は通常の連続する3枚を順子とみなす。
    """
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    memo: dict[tuple[tuple[tuple[str, int], int], ...], int] = {}
    
    数牌 = ("萬子", "筒子", "索子")
    
    def dfs(c: Counter) -> int:
        """残りカウンタ c から作れる最大順子数を返す（メモ化付き）"""
        key = tuple(sorted((k, v) for k, v in c.items() if v))
        if not key:  # 牌が残っていない
            return 0
        if key in memo:  # メモ化
            return memo[key]
        
        best = 0
        
        # すべての牌を起点に「順子」のみを試す
        for (suit, num), cnt in list(c.items()):
            if cnt == 0 or suit not in 数牌:
                continue
            
            if allowed_sequences:
                # 許可された順子のみを考慮
                for seq in allowed_sequences:
                    # この順子の開始数字が現在の牌と一致するかチェック
                    if seq[0] != num:
                        continue
                    
                    # この順子が形成可能かチェック
                    can_form = True
                    for seq_num in seq:
                        if c.get((suit, seq_num), 0) == 0:
                            can_form = False
                            break
                    
                    if can_form:
                        # 牌を使用
                        for seq_num in seq:
                            c[(suit, seq_num)] -= 1
                        
                        best = max(best, 1 + dfs(c))
                        
                        # 牌を戻す
                        for seq_num in seq:
                            c[(suit, seq_num)] += 1
            else:
                # 通常の順子（連続する3枚）のみを考慮
                if num > 7:  # 順子の開始数字が7を超えると無効
                    continue
                
                k1, k2 = (suit, num + 1), (suit, num + 2)
                if c[k1] and c[k2]:
                    c[(suit, num)] -= 1
                    c[k1] -= 1
                    c[k2] -= 1
                    best = max(best, 1 + dfs(c))
                    c[(suit, num)] += 1
                    c[k1] += 1
                    c[k2] += 1
        
        memo[key] = best
        return best
    
    順子数 = dfs(counter)  # 0〜4
    return score_table[順子数]


# 手牌 = [ 
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),        
# ]
# print(順子スコア(手牌, [[1,2,3],[4,5,6]]))


def 対子スコア(tiles: list[麻雀牌]) -> int:
    """
    13 枚の手牌から完成対子の最大数を求めて
    0対子→0, 1対子→1, 2対子→2, 3対子→4, 4対子→8, 5対子→16, 6対子→32を返す。副露は数えない。
    """
    tiles = [t for t in tiles if not t.副露]
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    pairs_count = 0
    for (_, _), cnt in counter.items():
        pairs_count += cnt // 2
    if pairs_count > 6:
        pairs_count = 6
    return [0, 1, 2, 4, 8, 16, 32][pairs_count]


def 搭子スコア(tiles: list[麻雀牌]) -> int:
    """
    搭子：例：(萬子4,萬子5),(索子1,索子2), same suit, with num diff 1
    13 枚の手牌から完成搭子の最大数を求めて
    0→0, 1→1, 2→2, 3→4, 4→8, 5→16, 6→32を返す。13枚ですので7+は有り得ない。
    副露は数えない。
    """
    # 副露牌を除外
    tiles = [t for t in tiles if not t.副露]
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    
    # 牌の種類と数字ごとのカウンタを作成
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    memo: dict[tuple[tuple[tuple[str, int], int], ...], int] = {}
    数牌 = ("萬子", "筒子", "索子")
    
    def dfs(c: Counter) -> int:
        """残りカウンタ c から作れる最大搭子数を返す（メモ化付き）"""
        key = tuple(sorted((k, v) for k, v in c.items() if v))
        if not key:             # 牌が残っていない
            return 0
        if key in memo:         # メモ化
            return memo[key]
        
        best = 0
        # すべての牌を起点に「搭子」を試す
        for (suit, num), cnt in list(c.items()):
            if cnt == 0:
                continue
            
            # 搭子を試す (例: 1-2, 2-3, etc.)
            if suit in 数牌 and num <= 8:
                next_tile = (suit, num + 1)
                if c[next_tile] > 0:
                    # 搭子を取る
                    c[(suit, num)] -= 1
                    c[next_tile] -= 1
                    
                    # 再帰的に残りの牌から最大搭子数を求める
                    best = max(best, 1 + dfs(c))
                    
                    # 搭子を戻す（バックトラック）
                    c[(suit, num)] += 1
                    c[next_tile] += 1
        
        memo[key] = best
        return best
    
    # 搭子の最大数を計算
    tanki_count = dfs(counter)
    
    # 0〜6の搭子数を対応するスコアに変換
    tanki_count = min(tanki_count, 6)  # 6以上は6として扱う
    return [0, 1, 2, 4, 8, 16, 32][tanki_count]


# 手牌 = [ 
#     麻雀牌("萬子", 5, False), 麻雀牌("萬子", 4, False), 麻雀牌("萬子", 3, False),  
# ]
# print(搭子スコア(手牌)) # 1
# 手牌 = [ 
#     麻雀牌("萬子", 5, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 5, False),  
# ]
# print(搭子スコア(手牌)) # 0
# 手牌 = [ 
#     麻雀牌("萬子", 5, False), 麻雀牌("萬子", 4, False), 麻雀牌("萬子", 3, False),  麻雀牌("萬子", 2, False)
# ]
# print(搭子スコア(手牌)) # 2

def 上がり形(tiles, process_marked_as_removed=False) -> bool:
    """
    Check if the given tiles can form a valid winning hand.
    A winning hand consists of sets (面子) and 1 pair (雀頭).
    Exposed tiles (副露=True) are removed before checking.
    Args:
        tiles: A list of 麻雀牌 objects
    Returns:
        bool: True if the tiles can form a valid winning hand, False otherwise
    """
    # Remove exposed tiles
    if process_marked_as_removed:
        remaining_tiles = [t for t in tiles if not t.marked_as_removed]
    else:
        remaining_tiles = [t for t in tiles if not t.副露]
    # Check if we have at least 14 tiles (including exposed)
    if len(tiles) < 14:
        return False
    # Sort the remaining tiles
    remaining_tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    
    def can_make_sets(c):
        """
        Check if the remaining tiles can form sets.
        Args:
            c: Counter of (suit, number) tuples
        Returns:
            bool: True if the remaining tiles can form sets, False otherwise
        """
        # Find the smallest remaining tile
        for key, cnt in c.items():
            if cnt:  # If there's at least one tile remaining
                suit, num = key
                break
        else:
            return True  # All tiles used, success!
            
        # Try to form a triplet (刻子)
        if cnt >= 3:
            c[key] -= 3
            if can_make_sets(c):
                return True
            c[key] += 3  # Backtrack
            
        # Try to form a sequence (順子) - only for numbered suits
        if suit in ("萬子", "筒子", "索子") and 1 <= num <= 7:
            k1, k2 = (suit, num + 1), (suit, num + 2)
            if c.get(k1, 0) > 0 and c.get(k2, 0) > 0:
                c[key] -= 1
                c[k1] -= 1
                c[k2] -= 1
                if can_make_sets(c):
                    return True
                c[key] += 1  # Backtrack
                c[k1] += 1
                c[k2] += 1
        return False  # Can't form sets
    
    # Create a counter of (suit, number) tuples
    counter = Counter((t.何者, t.その上の数字) for t in remaining_tiles)
    # Try each possible pair (雀頭)
    for key, cnt in list(counter.items()):
        if cnt >= 2:
            c = deepcopy(counter)
            c[key] -= 2  # Remove the pair
            if can_make_sets(c):
                return True
    return False


# 手牌 = [ 
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False),  
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),        
# ]
# print(上がり形(手牌)) # Must be True


# ====================================================
# 無役
# ====================================================

def 発(tiles_counter) -> bool:
    return tiles_counter.get("發ちゃん", 0) >= 3

def 中(tiles_counter) -> bool:
    return tiles_counter.get("中ちゃん", 0) >= 3

def 白(tiles_counter) -> bool:
    return tiles_counter.get("白ちゃん", 0) >= 3

def 東(tiles_counter) -> bool:
    return tiles_counter.get("東風", 0) >= 3

def 南(tiles_counter) -> bool:
    return tiles_counter.get("南風", 0) >= 3

def 西(tiles_counter) -> bool:
    return tiles_counter.get("西風", 0) >= 3

def 北(tiles_counter) -> bool:
    return tiles_counter.get("北風", 0) >= 3

def 赤ドラの数(tiles: list[麻雀牌]) -> int:
    return len([t for t in tiles if t.赤ドラ])

def 断么九(tiles: list[麻雀牌]) -> bool:
    if any("么九牌" in t.固有状態 for t in tiles):
        return False
    return True

# ====================================================
# 和了形
# ====================================================

def 混全帯么九(tiles: list[麻雀牌]) -> bool:
    """
    使用できるのは123の順子と789の順子、および一九字牌の対子と刻子である。
    """
    if all("字牌" in t.固有状態 for t in tiles):
        return False
    if all("数牌" in t.固有状態 for t in tiles):
        return False
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # print(counter.items())
    # dict_items([(('西風', 0), 3), (('索子', 6), 4), (('萬子', 6), 2), (('筒子', 6), 3), (('白ちゃん', 0), 2)])
    # 1. any 456 will result in false
    for key, cnt in counter.items():
        if key[1] in [4, 5, 6]:
            return False
    # 2. If there is an 2, the number count of 3 of the same suit must be the same, the same for 8 and 7
    for key, cnt in counter.items():
        if key[1] == 2:
            if counter[(key[0], 3)] != cnt or counter[(key[0], 1)] < cnt:
                return False
        if key[1] == 3:
            if counter[(key[0], 2)] != cnt or counter[(key[0], 1)] < cnt:
                return False
        if key[1] == 8:
            if counter[(key[0], 7)] != cnt or counter[(key[0], 9)] < cnt:
                return False
        if key[1] == 7:
            if counter[(key[0], 8)] != cnt or counter[(key[0], 9)] < cnt:
                return False
    return True



def 純全帯么九(tiles: list[麻雀牌]) -> bool:
    if any("字牌" in t.固有状態 for t in tiles):
        return False
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # 1. any 456 will result in false
    for key, cnt in counter.items():
        if key[1] in [4, 5, 6]:
            return False
    # 2. If there is an 2, the number count of 3 of the same suit must be the same, the same for 8 and 7
    for key, cnt in counter.items():
        if key[1] == 2:
            if counter[(key[0], 3)] != cnt or counter[(key[0], 1)] < cnt:
                return False
        if key[1] == 3:
            if counter[(key[0], 2)] != cnt or counter[(key[0], 1)] < cnt:
                return False
        if key[1] == 8:
            if counter[(key[0], 7)] != cnt or counter[(key[0], 9)] < cnt:
                return False
        if key[1] == 7:
            if counter[(key[0], 8)] != cnt or counter[(key[0], 9)] < cnt:
                return False
    return True


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 
#     麻雀牌("萬子", 7, False), 麻雀牌("萬子", 8, False), 麻雀牌("萬子", 9, False),  
#     麻雀牌("索子", 9, False), 麻雀牌("索子", 9, False), 麻雀牌("索子", 9, False),  
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 8, False), 麻雀牌("筒子", 9, False), 
#     麻雀牌("筒子", 9, False),
#     麻雀牌("筒子", 9, False)           
# ]
# print(純全帯么九(手牌))


def 混一色(tiles: list[麻雀牌]) -> bool:
    if all("字牌" in t.固有状態 for t in tiles):
        return False
    if all("数牌" in t.固有状態 for t in tiles):
        return False
    temp_tiles = [t.何者 for t in tiles if "数牌" in t.固有状態]
    temp_tiles = list(set(temp_tiles))
    if len(temp_tiles) != 1:
        return False
    return True


def 清一色(tiles: list[麻雀牌]) -> bool:
    if any("字牌" in t.固有状態 for t in tiles):
        return False
    temp_tiles = [t.何者 for t in tiles]
    temp_tiles = list(set(temp_tiles))
    if len(temp_tiles) != 1:
        return False
    return True


def 混老頭(tiles: list[麻雀牌]) -> bool:
    if all("字牌" in t.固有状態 for t in tiles):
        return False
    if all("数牌" in t.固有状態 for t in tiles):
        return False
    if any("中張牌" in t.固有状態 for t in tiles):
        return False
    return True


def 清老頭(tiles: list[麻雀牌]) -> bool:
    if any("字牌" in t.固有状態 for t in tiles):
        return False
    if any("中張牌" in t.固有状態 for t in tiles):
        return False
    return True


def 字一色(tiles: list[麻雀牌]) -> bool:
    if all("字牌" in t.固有状態 for t in tiles):
        return True
    return False


def 緑一色(tiles: list[麻雀牌]) -> bool:
    """
    23468発
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    for key, cnt in counter.items():
        if key[0] == "索子":
            if key[1] not in [2, 3, 4, 6, 8]:
                return False
        elif key[0] == "發ちゃん":
            continue
        else:
            return False
    return True


def 黒一色(tiles: list[麻雀牌]) -> bool:
    """
    筒子の黒丸のみの牌248と風牌で構成された和了形
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    for key, cnt in counter.items():
        if key[0] == "筒子":
            if key[1] not in [2, 4, 8]:
                return False
        elif key[0] in ["東風", "南風", "西風", "北風"]:
            continue
        else:
            return False
    return True


# 手牌 = [
#     麻雀牌("南風", 0, False), 麻雀牌("南風", 0, False), 麻雀牌("南風", 0, False),  
#     麻雀牌("筒子", 8, False), 麻雀牌("筒子", 8, False), 麻雀牌("筒子", 8, False),  
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(黒一色(手牌))


def 五門斉(tiles: list[麻雀牌]) -> bool:
    """
    萬子・筒子・索子・風牌・三元牌を全て使った和了形を作った時に成立する役。
    """
    has_wind = False
    has_dragon = False
    has_pinzu = False
    has_manzu = False
    has_souzu = False
    for tile in tiles:
        if "四風牌" in tile.固有状態:
            has_wind = True
        if "三元牌" in tile.固有状態:
            has_dragon = True
        if "筒子" in tile.何者:
            has_pinzu = True
        if "萬子" in tile.何者:
            has_manzu = True
        if "索子" in tile.何者:
            has_souzu = True
        # Early termination if all categories found
        if has_wind and has_dragon and has_pinzu and has_manzu and has_souzu:
            return True
    return has_wind and has_dragon and has_pinzu and has_manzu and has_souzu


def 対々和(tiles: list[麻雀牌]) -> bool:
    """
    すべての面子が刻子で構成されている和了形。
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # v4: fix 萬子7 萬子7 萬子7 萬子8 萬子9 筒子1 筒子1 筒子1 | 萬子8 萬子8 萬子8 筒子3 筒子3 筒子3
    pairs = 0
    triplets = 0
    for count in counter.values():
        if count == 2:
            pairs += 1
        elif count >= 3:
            triplets += 1
        else:
            return False
    return pairs == 1 and triplets == 4


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False),  
#     麻雀牌("索子", 3, False), 麻雀牌("索子", 3, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 4, False), 麻雀牌("萬子", 4, False), 
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 7, False), 麻雀牌("筒子", 7, False),  
#     麻雀牌("索子", 4, False),
#     麻雀牌("索子", 4, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(対々和(手牌))


def 三暗刻(tiles: list[麻雀牌]) -> bool:
    """
    condition 2: if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    for t in tiles:
        if t.副露:
            return False
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    triplets = []
    
    # Find all triplets (condition 1 part 2)
    for key, cnt in counter.items():
        if cnt >= 3:
            triplets.append(key)
            
    if len(triplets) >= 3:
        # Condition 1 is met - found at least 3 concealed triplets
        # For condition 2, try all combinations of 3 triplets
        for three_triplets in combinations(triplets, 3):
            temp_tiles = deepcopy(tiles)
            # Mark the triplets as removed
            for suit, number in three_triplets:
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 3:  # Mark exactly 3 tiles of each triplet
                            break
            
            # Check if remaining tiles can form a valid hand
            if 上がり形(temp_tiles, process_marked_as_removed=True):
                return True
    return False


def 四暗刻(tiles: list[麻雀牌]) -> bool:
    """
    condition 2: if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    for t in tiles:
        if t.副露:
            return False
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    triplets = []
    
    # Find all triplets (condition 1 part 2)
    for key, cnt in counter.items():
        if cnt >= 3:
            triplets.append(key)

    if len(triplets) >= 4:
        # Condition 1 is met - found at least 4 concealed triplets
        # For condition 2, try all combinations of 4 triplets
        for four_triplets in combinations(triplets, 4):
            temp_tiles = deepcopy(tiles)
            # Mark the triplets as removed
            for suit, number in four_triplets:
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 3:  # Mark exactly 3 tiles of each triplet
                            break
            # Check if remaining tiles can form a valid hand
            if 上がり形(temp_tiles, process_marked_as_removed=True):
                return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 6, False), 麻雀牌("索子", 6, False), 麻雀牌("索子", 6, False),  
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 
 
#     麻雀牌("萬子", 6, False),
#     麻雀牌("萬子", 6, False)           
# ]
# print(三暗刻(手牌))
# print(四暗刻(手牌))


def 三色同刻(tiles: list[麻雀牌]) -> bool:
    """
    2. condition 2: if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    for key, cnt in counter.items():
        if cnt >= 3:
            if all(
                counter[(suit, key[1])] >= 3
                for suit in ("萬子", "筒子", "索子")
            ):
                # Condition 1 is met - found triplets of the same number in all three suits
                # For condition 2, mark the triplets as removed
                temp_tiles = deepcopy(tiles)
                # Mark exactly 3 tiles of each suit with the same number as removed
                number = key[1]
                for suit in ("萬子", "筒子", "索子"):
                    count = 0
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                            t.marked_as_removed = True
                            count += 1
                            if count == 3:  # Mark exactly 3 tiles of each suit
                                break
                # Check if remaining tiles can form a valid hand
                if 上がり形(temp_tiles, process_marked_as_removed=True):
                    return True
    return False



def 三色小同刻(tiles: list[麻雀牌]) -> bool:
    """
    三色小同刻: For suit in ("萬子", "筒子", "索子"), 2 of them must form triplets,
    1 of them must form a pair. 3 triplets will return false.
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # Iterate through the possible numbers
    for number in set(t.その上の数字 for t in tiles):
        # Check if we have sufficient tiles of this number in different suits
        suit_counts = {
            suit: counter.get((suit, number), 0)
            for suit in ("萬子", "筒子", "索子")
        }
        
        # We need suits with 3+ tiles and possibly 1 suit with exactly 2 tiles
        triplet_suits = [suit for suit, count in suit_counts.items() if count >= 3]
        pair_suits = [suit for suit, count in suit_counts.items() if count == 2]
        
        # Handle the case where we have 3 suits with 3+ tiles
        if len(triplet_suits) == 3:
            # Try all combinations of 2 triplets and 1 pair
            for pair_suit in triplet_suits:
                # Create a copy of tiles to mark as removed
                temp_tiles = deepcopy(tiles)
                
                # Mark 3 tiles for each of the 2 triplet suits
                for suit in triplet_suits:
                    if suit != pair_suit:  # Skip the one we're using as a pair
                        count = 0
                        for t in temp_tiles:
                            if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                                t.marked_as_removed = True
                                count += 1
                                if count == 3:  # Mark exactly 3 tiles of each triplet suit
                                    break
                
                # Mark 2 tiles for the pair suit
                count = 0
                for t in temp_tiles:
                    if t.何者 == pair_suit and t.その上の数字 == number and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 2:  # Mark exactly 2 tiles of the pair suit
                            break
                
                # Check if remaining tiles can form a valid hand
                if 面子スコア([t for t in temp_tiles if not t.marked_as_removed]) == 2:
                    return True
        
        # Handle the standard case: exactly 2 triplets and 1 pair
        elif len(triplet_suits) == 2 and len(pair_suits) == 1:
            # Create a copy of tiles to mark as removed
            temp_tiles = deepcopy(tiles)
            
            # Mark 3 tiles for each of the 2 triplet suits
            for suit in triplet_suits:
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 3:  # Mark exactly 3 tiles of each triplet suit
                            break
            
            # Mark 2 tiles for the pair suit
            for suit in pair_suits:
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 2:  # Mark exactly 2 tiles of the pair suit
                            break
            
            # Check if remaining tiles can form a valid hand
            if 面子スコア([t for t in temp_tiles if not t.marked_as_removed]) == 2:
                return True
    
    return False



# 手牌 = [
#     麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False),  
#     麻雀牌("索子", 6, False), 麻雀牌("索子", 6, False), 麻雀牌("索子", 6, False),  
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 
 
#     麻雀牌("萬子", 4, False),
#     麻雀牌("萬子", 5, False)           
# ]
# print(三色同刻(手牌))
# print(三色小同刻(手牌))


# 手牌 = [
#     麻雀牌("萬子", 6, False), 麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False),  
#     麻雀牌("索子", 6, False), 麻雀牌("索子", 6, False), 麻雀牌("索子", 6, False),  
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 
 
#     麻雀牌("萬子", 6, False),
#     麻雀牌("萬子", 6, False)           
# ]
# print(三色小同刻(手牌))


def 三連刻(tiles: list[麻雀牌]) -> bool:
    """
    1. example :萬子333, 萬子444, 萬子555
    2. if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    suits = ("萬子", "筒子", "索子")
    for suit in suits:
        for start_num in range(1, 8):  # We need 3 consecutive numbers, so we can start from 1 to 7
            if all(counter.get((suit, start_num + i), 0) >= 3 for i in range(3)):
                # Condition 1 is met - found 3 consecutive triplets in the same suit
                # For condition 2, mark the triplets as removed
                temp_tiles = deepcopy(tiles)
                # Mark exactly 3 of each number in the triplet sequence as removed
                for i in range(3):
                    number = start_num + i
                    count = 0
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                            t.marked_as_removed = True
                            count += 1
                            if count == 3:  # Mark exactly 3 tiles of each number
                                break
                # Check if remaining tiles can form a valid hand
                if 上がり形(temp_tiles, process_marked_as_removed=True):
                    return True
    return False

# 手牌 = [
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False),  
#     麻雀牌("萬子", 7, False), 麻雀牌("萬子", 7, False), 麻雀牌("萬子", 7, False),  
#     麻雀牌("萬子", 8, False), 麻雀牌("萬子", 8, False), 麻雀牌("萬子", 8, False), 
 
#     麻雀牌("萬子", 6, False),
#     麻雀牌("萬子", 6, False)           
# ]
# print(三連刻(手牌))


def 四連刻(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    suits = ("萬子", "筒子", "索子")
    for suit in suits:
        for start_num in range(1, 7):  # We need 4 consecutive numbers, so we can start from 1 to 6
            if all(counter.get((suit, start_num + i), 0) >= 3 for i in range(4)):
                # Condition 1 is met - found 3 consecutive triplets in the same suit
                # For condition 2, mark the triplets as removed
                temp_tiles = deepcopy(tiles)
                for i in range(4):
                    number = start_num + i
                    count = 0
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == number and not t.marked_as_removed:
                            t.marked_as_removed = True
                            count += 1
                            if count == 3:  # Mark exactly 3 tiles of each number
                                break
                if 上がり形(temp_tiles, process_marked_as_removed=True):
                    return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 9, False), 麻雀牌("萬子", 9, False), 麻雀牌("萬子", 9, False), 
#     麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False),  
#     麻雀牌("萬子", 7, False), 麻雀牌("萬子", 7, False), 麻雀牌("萬子", 7, False),  
#     麻雀牌("萬子", 8, False), 麻雀牌("萬子", 8, False), 麻雀牌("萬子", 8, False), 
 
#     麻雀牌("萬子", 2, False),
#     麻雀牌("萬子", 2, False)           
# ]
# print(四連刻(手牌))


def 三色連刻(tiles: list[麻雀牌]) -> bool:
    """
    example :萬子333, 筒子444, 索子555
    2. if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    suits = ("萬子", "筒子", "索子")
    
    for start_num in range(1, 8):  # Need 3 consecutive numbers starting from 1 up to 7
        # Try all possible arrangements of suits
        for suit_order in permutations(suits):
            # Check if each suit has a triplet of the required consecutive number
            if (counter.get((suit_order[0], start_num), 0) >= 3 and
                counter.get((suit_order[1], start_num + 1), 0) >= 3 and
                counter.get((suit_order[2], start_num + 2), 0) >= 3):
                
                # Condition 1 is met - found triplets of consecutive numbers across three suits
                # For condition 2, mark the triplets as removed
                temp_tiles = deepcopy(tiles)
                
                # Mark exactly 3 tiles of first suit with start_num
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit_order[0] and t.その上の数字 == start_num and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 3:  # Mark exactly 3 tiles
                            break
                
                # Mark exactly 3 tiles of second suit with start_num + 1
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit_order[1] and t.その上の数字 == start_num + 1 and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 3:  # Mark exactly 3 tiles
                            break
                
                # Mark exactly 3 tiles of third suit with start_num + 2
                count = 0
                for t in temp_tiles:
                    if t.何者 == suit_order[2] and t.その上の数字 == start_num + 2 and not t.marked_as_removed:
                        t.marked_as_removed = True
                        count += 1
                        if count == 3:  # Mark exactly 3 tiles
                            break
                
                # Check if remaining tiles can form a valid hand
                if 上がり形(temp_tiles, process_marked_as_removed=True):
                    return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 9, False), 麻雀牌("萬子", 9, False), 麻雀牌("萬子", 9, False), 
#     麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False), 麻雀牌("萬子", 6, False),  
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 7, False), 麻雀牌("筒子", 7, False),  
#     麻雀牌("索子", 8, False), 麻雀牌("索子", 8, False), 麻雀牌("索子", 8, False), 
 
#     麻雀牌("萬子", 2, False),
#     麻雀牌("萬子", 2, False)           
# ]
# print(三色連刻(手牌))


def 小三風(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles if t.何者 in ["東風", "南風", "西風", "北風"])
    if len(counter) == 3:
        if any(
            cnt == 2 for key, cnt in counter.items()
        ):
            return True
    return False


def 三風刻(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles if t.何者 in ["東風", "南風", "西風", "北風"])
    if len(counter) == 3:
        if all(
            cnt >= 3 for key, cnt in counter.items()
        ):
            return True
    return False


def 客風三刻(tiles: list[麻雀牌], seat: int) -> bool:
    dic = {0: "東風", 1: "南風", 2: "西風", 3: "北風"}
    to_remove = dic.get(seat, "")
    abcd = ["東風", "南風", "西風", "北風"]
    abcd.remove(to_remove)
    assert len(abcd) == 3
    counter = Counter((t.何者, t.その上の数字) for t in tiles if t.何者 in abcd)
    if len(counter) == 3:
        if all(
            cnt >= 3 for key, cnt in counter.items()
        ):
            return True
    return False

# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 2, False),  
#     麻雀牌("南風", 0, False), 麻雀牌("南風", 0, False), 麻雀牌("南風", 0, False), 
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False),  
#     麻雀牌("東風", 0, False),
#     麻雀牌("東風", 0, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# for t in 手牌:
#     print(t.何者, t.その上の数字, t.副露)
# print(小三風(手牌))
# print(三風刻(手牌))


def 四喜和(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles if t.何者 in ["東風", "南風", "西風", "北風"])
    if len(counter) == 4:
        return True
    return False


def 小三元(tiles: list[麻雀牌]) -> bool:
    dragons = ("白ちゃん", "發ちゃん", "中ちゃん")
    counter = Counter(t.何者 for t in tiles if t.何者 in dragons)
    if set(counter.keys()) != set(dragons):
        return False
    pair_cnt = sum(1 for c in counter.values() if c >= 2)
    triple_cnt = sum(1 for c in counter.values() if c >= 3)
    if triple_cnt == 3:
        return False
    return pair_cnt == 3


def 大三元(tiles: list[麻雀牌]) -> bool:
    dragons = ("白ちゃん", "發ちゃん", "中ちゃん")
    counter = Counter(t.何者 for t in tiles if t.何者 in dragons)
    if set(counter.keys()) != set(dragons):
        return False
    triple_cnt = sum(1 for c in counter.values() if c >= 3)
    return triple_cnt == 3


# 手牌 = [ 
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 
#     麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False), 麻雀牌("筒子", 6, False),
#     麻雀牌("發ちゃん", 0, False, 副露=True),
#     麻雀牌("發ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("中ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),        
# ]
# print(小三元(手牌))


def 平和(tiles: list[麻雀牌]) -> bool:
    """
    すべての面子が順子で構成されている和了形。
    """
    return 順子スコア(tiles) >= 8


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False), 麻雀牌("索子", 4, False),  
#     麻雀牌("萬子", 6, False), 麻雀牌("萬子", 7, False), 麻雀牌("萬子", 8, False), 
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 8, False), 麻雀牌("筒子", 9, False),  
#     麻雀牌("索子", 5, False),
#     麻雀牌("索子", 6, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(平和(手牌) and 上がり形(手牌))


def 一盃口(tiles: list[麻雀牌]) -> bool:
    """
    condition 1: No exposed tiles
    condition 2: form n, n+1, n+2 2 times for same n, for same suit. For example 萬子223344.
    condition 3: if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    for t in tiles:
        if t.副露:
            return False     
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    suits = ("萬子", "筒子", "索子")
    for suit in suits:
        for n in range(1, 8):  # Valid starting numbers for sequences
            # Check if we have at least 2 of each tile needed for two sequences
            if (counter[(suit, n)] >= 2 and 
                counter[(suit, n + 1)] >= 2 and 
                counter[(suit, n + 2)] >= 2):
                
                # Condition 2 is met: we have two possible identical sequences
                # Now check condition 3 by marking these tiles as removed
                temp_tiles = deepcopy(tiles)
                
                # Mark two sequences: find and mark 2 of each number (n, n+1, n+2)
                for num in range(n, n+3):
                    count = 0
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == num and count < 2 and not t.marked_as_removed:
                            t.marked_as_removed = True
                            count += 1
                    
                    # If we couldn't mark 2 of each number, we don't have a valid 一盃口
                    if count < 2:
                        break
                else:  # This else belongs to the for loop, executes if no break occurred
                    # Check if remaining tiles can form a valid hand
                    condition2 = 上がり形(temp_tiles, process_marked_as_removed=True)
                    if condition2:
                        return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False), 
#     麻雀牌("筒子", 2, False), 麻雀牌("筒子", 4, False), 麻雀牌("筒子", 3, False),  
#     麻雀牌("筒子", 1, False),
#     麻雀牌("筒子", 1, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(一盃口(手牌))


def 二盃口(tiles: list[麻雀牌]) -> bool:
    for t in tiles:
        if t.副露:
            return False     
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    suits = ("萬子", "筒子", "索子")
    p2_found = 0
    for suit in suits:
        for n in range(1, 8):  # Valid starting numbers for sequences
            # Check if we have at least 2 of each tile needed for two sequences
            if (counter[(suit, n)] >= 2 and 
                counter[(suit, n + 1)] >= 2 and 
                counter[(suit, n + 2)] >= 2):
                
                if (counter[(suit, n)] >= 4 and 
                    counter[(suit, n + 1)] >= 4 and 
                    counter[(suit, n + 2)] >= 4):
                    return True

                # Condition 2 is met: we have two possible identical sequences
                # Now check condition 3 by marking these tiles as removed
                temp_tiles = deepcopy(tiles)
                
                # Mark two sequences: find and mark 2 of each number (n, n+1, n+2)
                for num in range(n, n+3):
                    count = 0
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == num and count < 2 and not t.marked_as_removed:
                            t.marked_as_removed = True
                            count += 1
                    
                    # If we couldn't mark 2 of each number, we don't have a valid 一盃口
                    if count < 2:
                        break
                else:
                    condition2 = 上がり形(temp_tiles, process_marked_as_removed=True)
                    if condition2:
                        if p2_found == 1:
                            return True
                        elif p2_found == 0:
                            p2_found += 1
                        else:
                            raise Exception
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("筒子", 2, False), 麻雀牌("筒子", 3, False), 麻雀牌("筒子", 4, False),  
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False), 
#     麻雀牌("筒子", 2, False), 麻雀牌("筒子", 3, False), 麻雀牌("筒子", 4, False),  
#     麻雀牌("筒子", 1, False),
#     麻雀牌("筒子", 1, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(二盃口(手牌)) # True

# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False), 
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("筒子", 1, False),
#     麻雀牌("筒子", 1, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(二盃口(手牌)) # True


def 三色同順(tiles: list[麻雀牌]) -> bool:
    """
    condition 1: have n, n+1, n+2 in the 3 suit for same n
    condition 2: if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    suits = ("萬子", "筒子", "索子")
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    for n in range(1, 8):
        if all(
            counter[(suit, n)] >= 1 and
            counter[(suit, n + 1)] >= 1 and
            counter[(suit, n + 2)] >= 1
            for suit in suits
        ):
            # Condition 1 is met
            # For condition 2, mark the triplets in all three suits as removed
            temp_tiles = deepcopy(tiles)
            # Mark one triplet from each suit as removed
            for suit in suits:
                for i in range(n, n + 3):
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == i and not t.marked_as_removed:
                            t.marked_as_removed = True
                            break
            # Check if remaining tiles can form a valid hand
            condition2 = 上がり形(temp_tiles, process_marked_as_removed=True)
            if condition2:
                return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("筒子", 2, False), 麻雀牌("筒子", 4, False), 麻雀牌("筒子", 3, False),  
#     麻雀牌("筒子", 1, False),
#     麻雀牌("筒子", 1, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(三色同順(手牌))


def 三色三步(tiles: list[麻雀牌]) -> bool:
    """
    condition 1: have [n, n+1, n+2] [n+1, n+2, n+3] [n+2, n+3, n+4] in the 3 suit for same n
    for example: 萬子234, 筒子345, 索子456
    condition 2: if the checked tiles is marked as removed, the tiles can still form 上がり形.
    """
    suits = ("萬子", "筒子", "索子")
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    
    for n in range(1, 6):  # Since we need up to n+4, n can be 1 to 5
        # Check all possible arrangements of suits
        for suit_order in permutations(suits):
            # Check if we have the stepped sequences
            if (all(counter[(suit_order[0], n + i)] >= 1 for i in range(3)) and       # First sequence: n, n+1, n+2
                all(counter[(suit_order[1], n + 1 + i)] >= 1 for i in range(3)) and   # Second sequence: n+1, n+2, n+3
                all(counter[(suit_order[2], n + 2 + i)] >= 1 for i in range(3))):     # Third sequence: n+2, n+3, n+4
                
                # Condition 1 is met, now check condition 2
                temp_tiles = deepcopy(tiles)
                
                # Mark the tiles in each sequence as removed
                # First sequence in first suit
                for i in range(3):
                    for t in temp_tiles:
                        if t.何者 == suit_order[0] and t.その上の数字 == n + i and not t.marked_as_removed:
                            t.marked_as_removed = True
                            break
                
                # Second sequence in second suit
                for i in range(3):
                    for t in temp_tiles:
                        if t.何者 == suit_order[1] and t.その上の数字 == n + 1 + i and not t.marked_as_removed:
                            t.marked_as_removed = True
                            break
                
                # Third sequence in third suit
                for i in range(3):
                    for t in temp_tiles:
                        if t.何者 == suit_order[2] and t.その上の数字 == n + 2 + i and not t.marked_as_removed:
                            t.marked_as_removed = True
                            break
                
                # Check if remaining tiles can form a valid hand
                if 上がり形(temp_tiles, process_marked_as_removed=True):
                    return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False), 麻雀牌("索子", 4, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("筒子", 3, False), 麻雀牌("筒子", 4, False), 麻雀牌("筒子", 5, False),  
#     麻雀牌("筒子", 1, False),
#     麻雀牌("筒子", 1, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(三色三步(手牌))


def 一気通貫(tiles: list[麻雀牌]) -> bool:
    """
    Condition 1 : have 123456789 of the same suit
    Condition 2 : If 123456789 of the same suit is temperarily marked as removed, the tiles can still form 上がり形.
    tile.marked_as_removed, 上がり形(tiles, process_marked_as_removed=True)
    """
    suits = ("萬子", "筒子", "索子")
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    condition1 = False
    condition2 = False
    
    for suit in suits:
        if all(counter[(suit, n)] >= 1 for n in range(1, 10)):
            condition1 = True
            
            # For condition 2, mark the 123456789 tiles as removed
            temp_tiles = deepcopy(tiles)
            n = 1
            for t in temp_tiles:
                if t.何者 == suit and t.その上の数字 == n:
                    t.marked_as_removed = True
                    n += 1
            assert n == 10
            # Check if remaining tiles can form a valid hand
            condition2 = 上がり形(temp_tiles, process_marked_as_removed=True)
            
            # If both conditions are met, return True
            if condition1 and condition2:
                return True
    
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("萬子", 7, False), 麻雀牌("萬子", 8, False), 麻雀牌("萬子", 9, False),  
#     麻雀牌("萬子", 9, False),
#     麻雀牌("萬子", 9, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(一気通貫(手牌))



def 三色通貫(tiles: list[麻雀牌]) -> bool:
    """
    由萬筒索三門三組順子組成的一氣通貫的牌型。Valid as long as:
    1. 123456789 all exists in tiles
    2. "萬子", "筒子", "索子" must all exist in tiles
    3. The full seq is [(123), (456), (789)]. A suit claim one of them, then another claim another, then another claim the last one.
    4. After the successful claim of all seq, copy the tiles, mark the tiles who has claimed. The full tiles must still form 上がり形. 
    """
    suits = ("萬子", "筒子", "索子")
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    
    # Check if all suits have at least one tile
    if not all(any(counter[(suit, n)] >= 1 for n in range(1, 10)) for suit in suits):
        return False
    
    # Check if all numbers 1-9 exist in the tiles
    if not all(any(counter[(suit, n)] >= 1 for suit in suits) for n in range(1, 10)):
        return False
    
    # Define the three sequence groups
    sequences = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    
    # For rule 3: Check if we can assign one sequence to each 萬子
    # Then remove that sequence, check if we can assign 1 to 筒子,
    # then, check if we can assign the last one to 索子.
    # We also need to consider there could be mulitiple seq to assign to each suit, try all of them.
    # Check if each suit can claim at least one complete sequence
    suit_sequences = {}
    for suit in suits:
        suit_sequences[suit] = []
        for seq in sequences:
            # Check if this suit has all numbers in this sequence
            if all(counter.get((suit, n), 0) >= 1 for n in seq):
                suit_sequences[suit].append(seq)
    
    # For each possible ordering of the sequences
    for seq_ordering in permutations(sequences):
        # Try to assign sequences to suits in this order
        assignment_works = True
        used_suits = set()
        suit_to_seq = {}  # Map suits to their assigned sequences
        
        for seq in seq_ordering:
            # Find a suit that can claim this sequence and hasn't been used
            assigned = False
            for suit in suits:
                if suit not in used_suits and seq in suit_sequences[suit]:
                    used_suits.add(suit)
                    suit_to_seq[suit] = seq  # Store the assignment
                    assigned = True
                    break
            
            if not assigned:
                assignment_works = False
                break
        
        # If we found a valid assignment, return True
        if assignment_works and len(used_suits) == len(suits):
            # Implement condition 4: check if remaining tiles form 上がり形
            temp_tiles = deepcopy(tiles)
            
            # Mark the assigned tiles as removed
            for suit, seq in suit_to_seq.items():
                for n in seq:
                    # Mark exactly one tile of each number in each assigned sequence
                    for t in temp_tiles:
                        if t.何者 == suit and t.その上の数字 == n and not t.marked_as_removed:
                            t.marked_as_removed = True
                            break
            
            # Check if remaining tiles can form a valid hand
            if 上がり形(temp_tiles, process_marked_as_removed=True):
                return True

    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 8, False), 麻雀牌("筒子", 9, False),  
#     麻雀牌("筒子", 9, False),
#     麻雀牌("筒子", 9, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(三色通貫(手牌))


def 鏡同和(tiles: list[麻雀牌]) -> bool:
    """
    m1 m2 m3 m5 m5 m5 s1 s2 s3 s5 s5 s5 + 2east
    """
    suits = ("萬子", "筒子", "索子")
    for suit_order in permutations(suits):
        l = [t.その上の数字 for t in tiles if t.何者 == suit_order[0]]
        r = [t.その上の数字 for t in tiles if t.何者 == suit_order[1]]
        if l == r and len(l) > 0:
            return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("索子", 4, False), 麻雀牌("索子", 5, False), 麻雀牌("索子", 6, False),  
#     麻雀牌("筒子", 9, False),
#     麻雀牌("筒子", 9, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(鏡同和(手牌))


def 七対子(tiles: list[麻雀牌]) -> bool:
    for t in tiles:
        if t.副露:
            return False
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # print(counter)
    # Counter({('萬子', 1): 2, ('萬子', 2): 2, ('萬子', 3): 2, ('萬子', 4): 2, ('萬子', 5): 2, ('白ちゃん', 0): 2, ('中ちゃん', 0): 2})
    # True if all counter are 2 or 4
    for key, cnt in counter.items():
        if cnt != 2 and cnt != 4:
            return False
    return True


def 二槓子(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    k = 0
    for key, cnt in counter.items():
        if cnt == 4:
            k += 1
    return k >= 2


def 三槓子(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    k = 0
    for key, cnt in counter.items():
        if cnt == 4:
            k += 1
    return k >= 3


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False), 
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("筒子", 1, False),
#     麻雀牌("筒子", 1, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# print(三槓子(手牌)) # True


def 国士無双(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles if "么九牌" in t.固有状態)
    if len(counter) == 13:
        return True
    return False


# 国士無双手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 9, False), 
#     麻雀牌("筒子", 1, False), 麻雀牌("筒子", 9, False), 
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 9, False), 
#     麻雀牌("東風", 0, False), 麻雀牌("南風", 0, False), 
#     麻雀牌("西風", 0, False), 麻雀牌("北風", 0, False), 
#     麻雀牌("白ちゃん", 0, False), 麻雀牌("發ちゃん", 0, False),
#     麻雀牌("中ちゃん", 0, False)
# ]
# print(国士無双(国士無双手牌))


# ==========================
# 点数計算
# ==========================
def 点数計算(tiles: list[麻雀牌], seat: int) -> tuple[int, list[str], bool]:
    """
    seat: 0:東 1:南 2:西 3:北
    return: (score, yaku, win)
    """
    if len(tiles) != 14:
        raise ValueError(f"手牌は 14 枚である必要があります: 今{len(tiles)}枚。")
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    tiles_counter = Counter((t.何者, t.その上の数字) for t in tiles)
    score = 0
    yaku = []
    win = False

    if 上がり形(tiles):

        if seat == 0 and 東(tiles_counter):
            score += 1000
            yaku.append("東")
        elif seat == 1 and 南(tiles_counter):
            score += 1000
            yaku.append("南")
        elif seat == 2 and 西(tiles_counter):
            score += 1000
            yaku.append("西")
        elif seat == 3 and 北(tiles_counter):
            score += 1000
            yaku.append("北")

        if 発(tiles_counter):
            score += 1000
            yaku.append("發")
        if 中(tiles_counter):
            score += 1000
            yaku.append("中")
        if 白(tiles_counter):
            score += 1000
            yaku.append("白")

        if 赤ドラの数(tiles) > 0:
            score += 1000 * 赤ドラの数(tiles)
            yaku.append(f"赤ドラ{赤ドラの数(tiles)}")

        if 二槓子(tiles):
            score += 3000
            yaku.append("二槓子")
        if 三槓子(tiles):
            score += 3000
            yaku.append("三槓子")


        if 断么九(tiles):
            score += 1000
            yaku.append("断么九")
        if 平和(tiles):
            score += 1000
            yaku.append("平和")

        if 混全帯么九(tiles):
            score += 3000
            yaku.append("混全帯么九")
            win = True
        if 純全帯么九(tiles):
            score += 6000
            yaku.append("純全帯么九")
            win = True
        if 五門斉(tiles):
            score += 3000
            yaku.append("五門斉")
            win = True
        if 対々和(tiles):
            score += 3000
            yaku.append("対々和")
            win = True
        if 清一色(tiles):
            score += 6000
            yaku.append("清一色")
            win = True
        if 混一色(tiles):
            score += 3000
            yaku.append("混一色")
            win = True
        if 混老頭(tiles):
            score += 6000
            yaku.append("混老頭")
            win = True
        if 三暗刻(tiles):
            score += 6000
            yaku.append("三暗刻")
            win = True
        if 小三風(tiles):
            score += 3000
            yaku.append("小三風")
            win = True
        if 三風刻(tiles):
            score += 6000
            yaku.append("三風刻")
            win = True
        if 一盃口(tiles):
            score += 3000
            yaku.append("一盃口")
            win = True
        if 二盃口(tiles):
            score += 6000
            yaku.append("二盃口")
            win = True
        if 三色同順(tiles):
            score += 6000
            yaku.append("三色同順")
            win = True
        if 三色三步(tiles):
            score += 6000
            yaku.append("三色三步")
            win = True
        if 一気通貫(tiles):
            score += 3000
            yaku.append("一気通貫")
            win = True
        if 三色通貫(tiles):
            score += 6000
            yaku.append("三色通貫")
            win = True
        if 鏡同和(tiles):
            score += 6000
            yaku.append("鏡同和")
            win = True
        if 小三元(tiles):
            score += 6000
            yaku.append("小三元")
            win = True
        if 三色小同刻(tiles):
            score += 6000
            yaku.append("三色小同刻")
            win = True
        if 大三元(tiles):
            score += 32000
            yaku.append("大三元")
            win = True
        if 清老頭(tiles):
            score += 32000
            yaku.append("清老頭")
            win = True
        if 字一色(tiles):
            score += 32000
            yaku.append("字一色")
            win = True
        if 緑一色(tiles):
            score += 32000
            yaku.append("緑一色")
            win = True
        if 黒一色(tiles):
            score += 32000
            yaku.append("黒一色")
            win = True
        if 四暗刻(tiles):
            score += 32000
            yaku.append("四暗刻")
            win = True
        if 三連刻(tiles):
            score += 3000
            yaku.append("三連刻")
            win = True
        if 四連刻(tiles):
            score += 32000
            yaku.append("四連刻")
            win = True
        if 三色連刻(tiles):
            score += 32000
            yaku.append("三色連刻")
            win = True
        if 三色同刻(tiles):
            score += 32000
            yaku.append("三色同刻")
            win = True
        if 客風三刻(tiles, seat):
            score += 32000
            yaku.append("客風三刻")
        if 四喜和(tiles):
            score += 32000
            yaku.append("四喜和")
            win = True


    if 七対子(tiles):
        score += 3000
        yaku.append("七対子")
        win = True

        if seat == 0 and 東(tiles_counter):
            score += 1000
            yaku.append("東")
        elif seat == 1 and 南(tiles_counter):
            score += 1000
            yaku.append("南")
        elif seat == 2 and 西(tiles_counter):
            score += 1000
            yaku.append("西")
        elif seat == 3 and 北(tiles_counter):
            score += 1000
            yaku.append("北")

        if 発(tiles_counter):
            score += 1000
            yaku.append("發")
        if 中(tiles_counter):
            score += 1000
            yaku.append("中")
        if 白(tiles_counter):
            score += 1000
            yaku.append("白")

        if 赤ドラの数(tiles) > 0:
            score += 1000 * 赤ドラの数(tiles)
            yaku.append(f"赤ドラ{赤ドラの数(tiles)}")

        if 二槓子(tiles):
            score += 3000
            yaku.append("二槓子")
        if 三槓子(tiles):
            score += 3000
            yaku.append("三槓子")

        if 断么九(tiles):
            score += 1000
            yaku.append("断么九")
        if 五門斉(tiles):
            score += 3000
            yaku.append("五門斉")
        if 清一色(tiles):
            score += 6000
            yaku.append("清一色")
        if 混一色(tiles):
            score += 3000
            yaku.append("混一色")
        if 混老頭(tiles):
            score += 6000
            yaku.append("混老頭")
        if 鏡同和(tiles):
            score += 6000
            yaku.append("鏡同和")
        if 清老頭(tiles):
            score += 32000
            yaku.append("清老頭")
        if 小三元(tiles):
            score += 6000
            yaku.append("小三元")
        if 大三元(tiles):
            score += 32000
            yaku.append("大三元")
        if 字一色(tiles):
            score += 32000
            yaku.append("大七星")
        if 黒一色(tiles):
            score += 32000
            yaku.append("黒一色")
            win = True

    if 国士無双(tiles):
        score += 32000
        yaku.append("国士無双")
        win = True

    return score, yaku, win



def 聴牌ですか(tiles: list[麻雀牌], seat: int) -> tuple[bool, list[麻雀牌]]:
    """
    return (whether is tenpaing, list of tiles to tenpai)
    """
    if len([t for t in tiles if not t.副露]) > 13:
        raise ValueError(f"手牌は13枚に超えることはない")
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    counter = Counter((t.何者, t.その上の数字) for t in tiles)  # 赤ドラを区別しない
    待ち牌: list[麻雀牌] = []
    全候補: list[tuple[str, int]] = []
    for suit in ("萬子", "筒子", "索子"):
        for num in range(1, 10):
            全候補.append((suit, num))
    全候補 += [(honor, 0) for honor in
               ("東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん")]

    for 何者, 数字 in 全候補:
        # 4 tiles in hand, skip
        if counter[(何者, 数字)] >= 4:
            continue

        仮手牌 = tiles + [麻雀牌(何者, 数字, False)]
        a, b, c = 点数計算(仮手牌, seat)
        if c:
            待ち牌.append(麻雀牌(何者, 数字, False))

    # 重複除去（同種同数の牌が複数回入るのを防ぐ）
    unique = {(p.何者, p.その上の数字) for p in 待ち牌}
    待ち牌 = [麻雀牌(s, n, False) for (s, n) in unique]

    return bool(待ち牌), 待ち牌


if __name__ == "__main__":
    pass