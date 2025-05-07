from collections import Counter
from copy import deepcopy
from itertools import combinations, permutations
import linecache
import random

class 麻雀牌:
    def __init__(self, 何者: str, その上の数字: int, 赤ドラ: bool = False, 副露: bool = False):
        self.何者: str = 何者
        self.その上の数字: int = その上の数字
        self.赤ドラ: bool = 赤ドラ
        self.固有状態: list[str] = []
        self.アクティブ状態: list[str] = []
        self.ドラ: bool = False
        self.副露: bool = 副露
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
        # if self.副露:
        #     表示名 += "副露"
        表示名 += f"{self.何者}"
        if self.その上の数字 > 0:
            表示名 += f"{self.その上の数字}"
        if self.赤ドラ:
            表示名 += "赤ドラ"
        # add passive state
        # if self.固有状態:
        #     表示名 += f" {self.固有状態}"
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
        
    def mark_as_exposed(self):
        if self.副露:
            raise Exception(f"牌はすでに副露されています: {self.何者}, {self.その上の数字}")
        self.副露 = True
        return None

    def get_asset(self, yoko: bool = False) -> str:
        """
        牌に対応する PNG ファイルのパスを返す。
        `yoko=True` で横向き画像 (-yoko 付き) を返す。
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



def 基礎訓練山を作成する() -> list[麻雀牌]:
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

    random_line = linecache.getline('tenpai_hands.txt', random.randint(0, 28645)).strip()
    fake_tiles = create_mahjong_tiles_from_line(random_line)
    for ft in fake_tiles:
        for t in 山:
            if ft.何者 == t.何者 and ft.その上の数字 == t.その上の数字 and not t.marked_a:
                t.marked_a = True
                break
    # remove all marked_a tiles in 山
    山 = [t for t in 山 if not t.marked_a]
    assert len(山) == 136 - 13
    山 = fake_tiles + 山

    assert len(山) == 136, f"山の長さが不正です: {len(山)}" # TypeError: object of type 'NoneType' has no len()
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
        
        # 搭子を作らない選択肢も考慮
        for (suit, num), cnt in list(c.items()):
            if cnt == 0:
                continue
            
            # この牌を使わずに次へ
            c[(suit, num)] -= 1
            best = max(best, dfs(c))
            c[(suit, num)] += 1
            
            # 一つ試せば十分（すべての牌について同じ操作を繰り返すと無駄なので）
            break
            
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
#     麻雀牌("中ちゃん", 0, False, 副露=True),  
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),
#     麻雀牌("白ちゃん", 0, False, 副露=True),           
# ]
# print(上がり形(手牌)) # Must be True


# ====================================================
# 無役
# ====================================================

def 発(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "發ちゃん"]) >= 3

def 中(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "中ちゃん"]) >= 3

def 白(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "白ちゃん"]) >= 3

def 東(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "東風"]) >= 3

def 南(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "南風"]) >= 3

def 西(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "西風"]) >= 3

def 北(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "北風"]) >= 3

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
    2. condition 2: if the checked tiles is marked as removed, the tiles can still form 上がり形.
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
    4. NEW: After the successful claim of all seq, copy the tiles, mark the tiles who has claimed. The full tiles must still form 上がり形. 
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


def 五門斉(tiles: list[麻雀牌]) -> bool:
    """
    萬子・筒子・索子・風牌・三元牌を全て使った和了形を作った時に成立する役。
    """
    if any("四風牌" in t.固有状態 for t in tiles):
        if any("三元牌" in t.固有状態 for t in tiles):
            if any("筒子" in t.何者 for t in tiles):
                if any("萬子" in t.何者 for t in tiles):
                    if any("索子" in t.何者 for t in tiles):
                        return True
    return False


def 対々和(tiles: list[麻雀牌]) -> bool:
    """
    すべての面子が刻子または槓子で構成されている和了形。
    """
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    c = 0
    if len(counter) <= 5:
        for key, cnt in counter.items():
            if cnt >= 3:
                c += 1
        if c >= 4:
            return True
    return False


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False),  
#     麻雀牌("索子", 3, False), 麻雀牌("索子", 3, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 4, False), 麻雀牌("萬子", 4, False), 
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 7, False), 麻雀牌("筒子", 7, False),  
#     麻雀牌("索子", 2, False),
#     麻雀牌("索子", 3, False) 
# ]
# 手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
# for t in 手牌:
#     print(t.何者, t.その上の数字, t.副露)
# print(対々和(手牌))


def 小三元(tiles: list[麻雀牌]) -> bool:
    dragons = ("白ちゃん", "發ちゃん", "中ちゃん")
    counter = Counter(t.何者 for t in tiles if t.何者 in dragons)
    # ３種すべて揃っていなければ不成立
    if set(counter.keys()) != set(dragons):
        return False
    # ペア１つ・刻子／槓子２つを確認
    pair_cnt   = sum(1 for c in counter.values() if c == 2)
    triple_cnt = sum(1 for c in counter.values() if c >= 3)
    return pair_cnt == 1 and triple_cnt == 2

def 大三元(tiles: list[麻雀牌]) -> bool:
    dragons = ("白ちゃん", "發ちゃん", "中ちゃん")
    counter = Counter(t.何者 for t in tiles if t.何者 in dragons)
    # ３種すべて揃っていなければ不成立
    if set(counter.keys()) != set(dragons):
        return False
    triple_cnt = sum(1 for c in counter.values() if c >= 3)
    return triple_cnt == 3


def 国士無双(tiles: list[麻雀牌]) -> bool:
    counter = Counter((t.何者, t.その上の数字) for t in tiles if "么九牌" in t.固有状態)
    if len(counter) == 13:
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
    score = 0
    yaku = []
    win = False

    if seat == 0 and 東(tiles):
        score += 1000
        yaku.append("東")
    if seat == 1 and 南(tiles):
        score += 1000
        yaku.append("南")
    if seat == 2 and 西(tiles):
        score += 1000
        yaku.append("西")
    if seat == 3 and 北(tiles):
        score += 1000
        yaku.append("北")

    if 発(tiles):
        score += 1000
        yaku.append("發")
    if 中(tiles):
        score += 1000
        yaku.append("中")
    if 白(tiles):
        score += 1000
        yaku.append("白")

    if 赤ドラの数(tiles) > 0:
        score += 1000 * 赤ドラの数(tiles)
        yaku.append(f"赤ドラ{赤ドラの数(tiles)}")

    if 上がり形(tiles):
        if 断么九(tiles):
            score += 1000
            yaku.append("断么九")
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
        if 三色同順(tiles):
            score += 3000
            yaku.append("三色同順")
            win = True
        if 一気通貫(tiles):
            score += 3000
            yaku.append("一気通貫")
            win = True
        if 三色通貫(tiles):
            score += 3000
            yaku.append("三色通貫")
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
        if 四暗刻(tiles):
            score += 32000
            yaku.append("四暗刻")
            win = True
        if 四連刻(tiles):
            score += 32000
            yaku.append("四連刻")
            win = True
        if 三連刻(tiles):
            score += 3000
            yaku.append("三連刻")
            win = True
        if 三色同刻(tiles):
            score += 32000
            yaku.append("三色同刻")
            win = True
        if 四喜和(tiles):
            score += 32000
            yaku.append("四喜和")
            win = True
        if 国士無双(tiles):
            score += 32000
            yaku.append("国士無双")
            win = True
        if 緑一色(tiles):
            score += 32000
            yaku.append("緑一色")
            win = True
    elif 七対子(tiles):
        score += 3000
        yaku.append("七対子")
        win = True
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
        if 清老頭(tiles):
            score += 32000
            yaku.append("清老頭")
        if 字一色(tiles):
            score += 32000
            yaku.append("大七星")

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


# 手牌 = [
#     麻雀牌("萬子", 1, False), 麻雀牌("萬子", 2, False), 麻雀牌("萬子", 3, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 2, False), 麻雀牌("索子", 3, False),  
#     麻雀牌("萬子", 4, False), 麻雀牌("萬子", 5, False), 麻雀牌("萬子", 6, False), 
#     麻雀牌("筒子", 7, False), 麻雀牌("筒子", 8, False), 麻雀牌("筒子", 9, False),  
#     麻雀牌("筒子", 9, False),
#     麻雀牌("筒子", 9, False) 
# ]


def generate_random_meld():
    is_triplet = random.choice([True, False])
    if is_triplet:
        # Generate a triplet (three identical tiles)
        suits = ["萬子", "筒子", "索子", "東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん"]
        weights = [4.5, 4.5, 4.5, 1, 1, 1, 1, 1, 1, 1]
        suit = random.choices(suits, weights=weights, k=1)[0]
        if suit in ["萬子", "筒子", "索子"]:
            num = random.randint(1, 9)
        else:
            num = 0
        return [麻雀牌(suit, num, False), 麻雀牌(suit, num, False), 麻雀牌(suit, num, False)]
    else:
        # Generate a sequence (three consecutive numbers in the same suit)
        suit = random.choice(["萬子", "筒子", "索子"])  # Only numbered suits can form sequences
        # Can only start a sequence with 1-7
        start_num = random.randint(1, 7)
        return [麻雀牌(suit, start_num, False), 麻雀牌(suit, start_num+1, False), 麻雀牌(suit, start_num+2, False)]

def generate_random_tile():
    suits = ["萬子", "筒子", "索子", "東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん"]
    weights = [9, 9, 9, 1, 1, 1, 1, 1, 1, 1]
    suit = random.choices(suits, weights=weights, k=1)[0]
    if suit in ["萬子", "筒子", "索子"]:
        num = random.randint(1, 9)
    else:
        num = 0
    return 麻雀牌(suit, num, False)


def generate_random_41_13_hand():
    hand = []
    for _ in range(4):
        meld = generate_random_meld()
        hand.extend(meld)
    hand.append(generate_random_tile())
    return hand


def generate_tenpai(max_attempts):
    # Generate hands until we find one in tenpai
    total_attempts = 0

    while total_attempts < max_attempts:
        total_attempts += 1
        hand = generate_random_41_13_hand()
        counter = Counter((t.何者, t.その上の数字) for t in hand)
        # check valid hand
        hand_is_valid = True
        for key, cnt in counter.items():
            if cnt > 4:
                hand_is_valid = False
                break
        if not hand_is_valid:
            continue
        is_tenpai, waiting_tiles = 聴牌ですか(hand.copy(), 0)  # Passing seat as 0
        if is_tenpai:
            with open("tenpai_hands.txt", "a", encoding="utf-8") as f:
                f.write(f"{nicely_print_tiles(hand)}\n")



def create_mahjong_tiles_from_line(line: str) -> list[麻雀牌]:
    if line.endswith(" |"):
        line = line[:-2].strip()
    tiles = []
    tile_specs = line.split()
    for tile_spec in tile_specs:
        if " " in tile_spec:
            # Handle multi-tile input, which is not implemented.
            raise ValueError("Multi-tile input is not supported.")
        if tile_spec in {"東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん"}:
            tiles.append(麻雀牌(tile_spec, 0))
        else:
            牌名 = tile_spec[:-1]
            数字 = int(tile_spec[-1])
            tiles.append(麻雀牌(牌名, 数字))
    return tiles


if __name__ == "__main__":
    pass
    # generate_tenpai(100000)