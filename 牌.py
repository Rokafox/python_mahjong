from collections import Counter
from copy import deepcopy
import random

class 麻雀牌:
    """
    麻雀牌。例：
    ("索子", 1, False)
    ("萬子", 5, True)
    ("東風", 0, False)
    ("白ちゃん", 0, False)
    """
    def __init__(self, 何者, その上の数字, 赤ドラ):
        self.何者: str = 何者
        self.その上の数字: int = その上の数字
        self.赤ドラ: bool = 赤ドラ
        self.固有状態: list[str] = []
        self.アクティブ状態: list[str] = []
        self.ドラ: bool = False
        self.副露: bool = False
        if not self.固有状態追加と検証():
            raise Exception(f"不正な牌: {何者}, {その上の数字}")
        self.sort_order = self.set_sort_order()
        self.init最後の処理()

    def __str__(self):
        表示名 = f"{self.何者}"
        if self.その上の数字 > 0:
            表示名 += f"{self.その上の数字}"
        if self.赤ドラ:
            表示名 += "赤ドラ"
        # add passive state
        # if self.固有状態:
        #     表示名 += f" {self.固有状態}"
        return 表示名

    def 固有状態追加と検証(self) -> bool:
        """
        四風牌->その上の数字が0
        筒子->その上の数字が1から9
        萬子->その上の数字が1から9
        索子->その上の数字が1から9
        白ちゃん->その上の数字が0
        發ちゃん->その上の数字が0
        中ちゃん->その上の数字が0
        """
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
        """
        """
        self.固有状態.sort()
        return None

    def アクティブ状態追加(self, 状態名: str):
        """
        """
        if 状態名 not in ["場風牌", "自風牌"]:
            raise Exception(f"不正な状態名: {状態名}")
        if 状態名 not in self.アクティブ状態:
            self.アクティブ状態.append(状態名)
        return None

    def アクティブ状態削除(self, 状態名: str):
        """
        """
        if 状態名 in self.アクティブ状態:
            self.アクティブ状態.remove(状態名)
        else:
            raise Exception(f"状態名が見つかりません: {状態名}")
        return None

    def set_sort_order(self):
        """
        """
        if self.何者 == "萬子":
            return 0
        elif self.何者 == "筒子":
            return 1
        elif self.何者 == "索子":
            return 2
        elif self.何者 in ["東風", "南風", "西風", "北風"]:
            return 3
        elif self.何者 in ["白ちゃん", "發ちゃん", "中ちゃん"]:
            return 4
        else:
            raise Exception(f"不正な牌: {self.何者}")
        

# ==========================
# 山
# ==========================

def 山を作成する() -> list[麻雀牌]:
    """
    """
    山: list[麻雀牌] = []
    # 数牌（萬子、筒子、索子）各1〜9を4枚ずつ
    for 何者 in ["萬子", "筒子", "索子"]:
        for 数 in range(1, 10):
            for _ in range(4):
                山.append(麻雀牌(何者, 数, 赤ドラ=False))
        # 赤ドラ：5に1枚だけ赤ドラとして追加（各色1枚ずつ）
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

    return 山


# ==========================
# 補助関数
# ==========================
def 面子スコア(tiles: list[麻雀牌]) -> int:
    """
    13 枚の手牌から完成面子（順子・刻子）の最大数を求めて
    0面子→0, 1面子→1, 2面子→2, 3面子→4, 4面子→8を返す。雀頭は数えない。
    """
    if len(tiles) != 13:
        raise ValueError("手牌は 13 枚である必要があります")
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
    0対子→0, 1対子→1, 2対子→2, 3対子→4, 4対子→8, 5対子→16, 6対子→32を返す。
    """
    if len(tiles) != 13:
        raise ValueError("手牌は 13 枚である必要があります")
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    pairs_count = 0
    for (_, _), cnt in counter.items():
        pairs_count += cnt // 2
    if pairs_count > 6:
        pairs_count = 6
    return [0, 1, 2, 4, 8, 16, 32][pairs_count]


def 四面子一雀頭ですか(tiles: list[麻雀牌]) -> bool:
    """
    """
    if len(tiles) != 14:
        raise ValueError("手牌は 14 枚である必要があります")
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    def _can_make_sets(c: dict[tuple[str, int], int]) -> bool:
        # まだ残っている最小の牌を探す
        for key, cnt in c.items():
            if cnt:                       # 1枚でも残っていたら処理開始
                suit, num = key
                break
        else:
            return True                  # 全部 0 なら成功

        # ① 刻子を取る
        if cnt >= 3:
            c[key] -= 3
            if _can_make_sets(c):
                return True
            c[key] += 3                  # バックトラック

        # ② 順子を取る（数牌のみ）
        if suit in ("萬子", "筒子", "索子") and 1 <= num <= 7:
            k1, k2 = (suit, num + 1), (suit, num + 2)
            if c.get(k1, 0) and c.get(k2, 0):
                c[key]  -= 1
                c[k1]   -= 1
                c[k2]   -= 1
                if _can_make_sets(c):
                    return True
                c[key]  += 1              # バックトラック
                c[k1]   += 1
                c[k2]   += 1

        return False                     # どちらも無理

    tiles = list(tiles)
    if len(tiles) != 14:
        return False

    counter = Counter((t.何者, t.その上の数字) for t in tiles)

    # まず雀頭候補をひとつ選び、残りを面子に分解できるか調べる
    for key, cnt in list(counter.items()):
        if cnt >= 2:
            c = deepcopy(counter)
            c[key] -= 2              # 雀頭として抜く
            if _can_make_sets(c):
                return True
    return False

def 聴牌ですか(tiles: list[麻雀牌],) -> tuple[bool, list[麻雀牌]]:
    """
    13 枚の手牌がチャンタ一向聴かどうかを判定し、
    待ち牌（重複なし）も返す。

    Returns
    -------
    bool              : テンパイなら True
    list[麻雀牌]      : 待ち牌オブジェクトのリスト
    """
    if len(tiles) != 13:
        raise ValueError("手牌は 13 枚である必要があります")
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    counter = Counter((t.何者, t.その上の数字) for t in tiles)  # 赤ドラを区別しない
    待ち牌: list[麻雀牌] = []

    # 全候補を列挙（ここでは赤ドラ無視）
    全候補: list[tuple[str, int]] = []
    for suit in ("萬子", "筒子", "索子"):
        for num in range(1, 10):
            全候補.append((suit, num))
    全候補 += [(honor, 0) for honor in
               ("東風", "南風", "西風", "北風", "白ちゃん", "發ちゃん", "中ちゃん")]

    for 何者, 数字 in 全候補:
        # 4 枚使い切っている牌はスキップ
        if counter[(何者, 数字)] >= 4:
            continue

        仮手牌 = tiles + [麻雀牌(何者, 数字, False)]
        if 純全帯么九(仮手牌):
            待ち牌.append(麻雀牌(何者, 数字, False))
        if 混全帯么九(仮手牌):
            待ち牌.append(麻雀牌(何者, 数字, False))
        if 四面子一雀頭ですか(仮手牌):
            if 混一色(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 混老頭(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 字一色(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 三色同順(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 一気通貫(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 五門斉(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 対々和(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 小三元(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
            if 大三元(仮手牌):
                待ち牌.append(麻雀牌(何者, 数字, False))
        if 七対子(仮手牌):
            待ち牌.append(麻雀牌(何者, 数字, False))

    # 重複除去（同種同数の牌が複数回入るのを防ぐ）
    unique = {(p.何者, p.その上の数字) for p in 待ち牌}
    待ち牌 = [麻雀牌(s, n, False) for (s, n) in unique]

    return bool(待ち牌), 待ち牌

# ====================================================
# 無役
# ====================================================

def 発(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "發ちゃん"]) == 3

def 中(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "中ちゃん"]) == 3

def 白(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "白ちゃん"]) == 3

def 東(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "東風"]) == 3

def 南(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "南風"]) == 3

def 西(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "西風"]) == 3

def 北(tiles: list[麻雀牌]) -> bool:
    return len([t for t in tiles if t.何者 == "北風"]) == 3

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
    counter = Counter((t.何者, t.その上の数字) for t in tiles)

    def _can_make_sets_chanta(c: dict[tuple[str, int], int]) -> bool:
        for key, cnt in c.items():
            if cnt:
                suit, num = key
                break
        else:
            return True

        if cnt >= 3:
            if (suit not in ("萬子", "筒子", "索子")) or num in (1, 9):
                c[key] -= 3
                if _can_make_sets_chanta(c):
                    return True
                c[key] += 3

        if suit in ("萬子", "筒子", "索子") and num in (1, 7):
            k1, k2 = (suit, num + 1), (suit, num + 2)
            if c.get(k1, 0) and c.get(k2, 0):
                c[key]  -= 1
                c[k1]   -= 1
                c[k2]   -= 1
                if _can_make_sets_chanta(c):
                    return True
                c[key]  += 1
                c[k1]   += 1
                c[k2]   += 1

        return False

    for key, cnt in list(counter.items()):
        if cnt >= 2:
            suit, num = key
            if (suit not in ("萬子", "筒子", "索子")) or num in (1, 9):
                c = deepcopy(counter)
                c[key] -= 2
                if _can_make_sets_chanta(c):
                    return True
    return False


def 純全帯么九(tiles: list[麻雀牌]) -> bool:
    if any("字牌" in t.固有状態 for t in tiles):
        return False
    return 混全帯么九(tiles)


def 混一色(tiles: list[麻雀牌]) -> bool:
    if all("字牌" in t.固有状態 for t in tiles):
        return False
    temp_tiles = [t.何者 for t in tiles if "数牌" in t.固有状態]
    temp_tiles = list(set(temp_tiles))
    if len(temp_tiles) != 1:
        return False
    return True


def 清一色(tiles: list[麻雀牌]) -> bool:
    if any("字牌" in t.固有状態 for t in tiles):
        return False
    if not 混一色(tiles):
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
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # print(counter)
    # Counter({('萬子', 1): 2, ('萬子', 2): 2, ('萬子', 3): 2, ('萬子', 4): 2, ('萬子', 5): 2, ('白ちゃん', 0): 2, ('中ちゃん', 0): 2})
    # True if all counter are 2 or 4
    for key, cnt in counter.items():
        if cnt != 2 and cnt != 4:
            return False
    return True


def 三色同順(tiles: list[麻雀牌]) -> bool:
    """
    1. 手牌は和了形（四面子一雀頭）である  
    2. 同じ数値 (n, n+1, n+2) で構成される順子が、萬子・筒子・索子の 3 色すべてに 1 組ずつ存在する
    """
    suits = ("萬子", "筒子", "索子")
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # 1〜7 から始まる順子まで調査（7‑8‑9 が最大）
    for n in range(1, 8):
        # 各色に n,n+1,n+2 が 1 枚ずつ揃っているか
        if all(
            counter[(suit, n)] >= 1 and
            counter[(suit, n + 1)] >= 1 and
            counter[(suit, n + 2)] >= 1
            for suit in suits
        ):
            return True
    return False


def 一気通貫(tiles: list[麻雀牌]) -> bool:
    """
    同種の数牌で123・456・789と揃えると成立する。
    """
    suits = ("萬子", "筒子", "索子")
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    for suit in suits:
        if all(counter[(suit, n)] >= 1 for n in range(1, 10)):
            return True
    return False


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
    # すべての牌が刻子または槓子であることを確認
    if len(counter) <= 5:
        return True
    return False


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



# 手牌 = [
#     麻雀牌("萬子", 7, False), 麻雀牌("萬子", 7, False), 麻雀牌("萬子", 7, False),  
#     麻雀牌("索子", 1, False), 麻雀牌("索子", 1, False), 麻雀牌("索子", 1, False),  
#     麻雀牌("筒子", 8, False), 麻雀牌("筒子", 8, False), 麻雀牌("筒子", 8, False), 
#     麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False), 麻雀牌("西風", 0, False),  
#     麻雀牌("白ちゃん", 0, False),
#     麻雀牌("白ちゃん", 0, False)           
# ]
# print(対々和(手牌))
# a, b = 聴牌ですか(手牌)
# print(a)
# for _ in b:
#     print(f"聴牌: {len(b)}")
#     print(_.何者, _.その上の数字)


# ==========================
# 点数計算
# ==========================
def 点数計算(tiles: list[麻雀牌], seat: int) -> tuple[int, list[str], bool]:
    """
    seat: 0:東 1:南 2:西 3:北
    """
    if len(tiles) != 14:
        raise ValueError(f"手牌は 14 枚である必要があります: 今{len(tiles)}枚。")
    tiles.sort(key=lambda x: (x.sort_order, x.その上の数字))
    score = 0
    yaku = []
    win = False
    chanta = False
    clearwater = False

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

    # 混全帯么九
    if 純全帯么九(tiles):
        score += 6000
        yaku.append("純全帯么九")
        win = True
        chanta = True
    if 混全帯么九(tiles) and not chanta:
        score += 3000
        yaku.append("混全帯么九")
        win = True

    if 四面子一雀頭ですか(tiles):
        if 断么九(tiles):
            score += 1000
            yaku.append("断么九")
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
            clearwater = True
        if 混一色(tiles) and not clearwater:
            score += 3000
            yaku.append("混一色")
            win = True
        if 混老頭(tiles):
            score += 6000
            yaku.append("混老頭")
            win = True
        if 三色同順(tiles):
            score += 3000
            yaku.append("三色同順")
            win = True
        if 一気通貫(tiles):
            score += 3000
            yaku.append("一気通貫")
            win = True
        if 小三元(tiles):
            score += 6000
            yaku.append("小三元")
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
    elif 七対子(tiles):
        score += 3000
        yaku.append("七対子")
        win = True
        if 五門斉(tiles):
            score += 3000
            yaku.append("五門斉")
            win = True
        if 混一色(tiles) and not clearwater:
            score += 3000
            yaku.append("混一色")
            win = True
        if 混老頭(tiles):
            score += 6000
            yaku.append("混老頭")
            win = True
        if 清老頭(tiles):
            score += 32000
            yaku.append("清老頭")
            win = True
        if 字一色(tiles):
            score += 32000
            yaku.append("字一色")
            win = True

    return score, yaku, win
