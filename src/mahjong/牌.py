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
        if self.固有状態:
            表示名 += f" {self.固有状態}"
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
# 役の判定
# ==========================

def 四面子一雀頭ですか(tiles: list[麻雀牌]) -> bool:
    """
    """
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


def 混全帯么九ですか(tiles: list[麻雀牌]) -> bool:
    """
    """
    if len(tiles) != 14:
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
                print(c)
                if _can_make_sets_chanta(c):
                    return True
    return False


def 純全帯么九ですか(tiles: list[麻雀牌]) -> bool:
    """
    """
    if len(tiles) != 14:
        return False

    counter = Counter((t.何者, t.その上の数字) for t in tiles)

    def _can_make_sets_pure_chanta(c: dict[tuple[str, int], int]) -> bool:
        for key, cnt in c.items():
            if cnt:
                suit, num = key
                break
        else:
            return True

        if cnt >= 3:
            if num in (1, 9):
                c[key] -= 3
                if _can_make_sets_pure_chanta(c):
                    return True
                c[key] += 3

        if suit in ("萬子", "筒子", "索子") and num in (1, 7):
            k1, k2 = (suit, num + 1), (suit, num + 2)
            if c.get(k1, 0) and c.get(k2, 0):
                c[key]  -= 1
                c[k1]   -= 1
                c[k2]   -= 1
                if _can_make_sets_pure_chanta(c):
                    return True
                c[key]  += 1
                c[k1]   += 1
                c[k2]   += 1

        return False

    for key, cnt in list(counter.items()):
        if cnt >= 2:
            suit, num = key
            if num in (1, 9):
                c = deepcopy(counter)
                c[key] -= 2
                print(c)
                if _can_make_sets_pure_chanta(c):
                    return True
    return False


手牌 = [
    麻雀牌("索子", 8, False), 麻雀牌("索子", 7, False), 麻雀牌("索子", 9, False),  
    麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False), 麻雀牌("萬子", 1, False),  
    麻雀牌("筒子", 8, False), 麻雀牌("筒子", 7, False), 麻雀牌("筒子", 9, False), 
    麻雀牌("萬子", 9, False), 麻雀牌("萬子", 9, False), 麻雀牌("萬子", 9, False),  
    麻雀牌("筒子", 9, False), 麻雀牌("筒子", 9, False)           
]
手牌.sort(key=lambda x: (x.sort_order, x.その上の数字))
for _ in 手牌:
    print(_)

print(純全帯么九ですか(手牌))   # True


def 七対子ですか(tiles: list[麻雀牌]) -> bool:
    if len(tiles) != 14:
        return False
    counter = Counter((t.何者, t.その上の数字) for t in tiles)
    # print(counter)
    # Counter({('萬子', 1): 2, ('萬子', 2): 2, ('萬子', 3): 2, ('萬子', 4): 2, ('萬子', 5): 2, ('白ちゃん', 0): 2, ('中ちゃん', 0): 2})
    # True if all counter are 2 or 4
    for key, cnt in counter.items():
        if cnt != 2 and cnt != 4:
            return False
    return True