class 麻雀牌:
    """
    麻雀牌。例：
    "東風", 0
    "南風", 0
    "西風", 0
    "北風", 0
    ”萬子", 5
    "筒子", 5
    "索子", 1
    ”白ちゃん", 0
    "發ちゃん", 0
    "中ちゃん", 0
    """
    def __init__(self, 何者, その上の数字, 赤ドラ):
        self.何者: str = 何者
        self.その上の数字: int = その上の数字
        self.赤ドラ: bool = 赤ドラ
        self.牌の状態: list[str] = [] # 例: ["四風牌"]
        self.おいらはドラ: bool = False

        if not self.検証():
            raise ValueError(f"不正な牌: {何者}, {その上の数字}")

    def __str__(self):
        表示名 = f"{self.何者}"
        if self.その上の数字 > 0:
            表示名 += f"{self.その上の数字}"
        if self.赤ドラ:
            表示名 += "（赤ドラ）"
        return 表示名


    def 検証(self) -> bool:
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
            self.牌の状態.append("四風牌")
            return self.その上の数字 == 0
        elif self.何者 in ["筒子", "萬子", "索子"]:
            return 1 <= self.その上の数字 <= 9
        elif self.何者 in ["白ちゃん", "發ちゃん", "中ちゃん"]:
            return self.その上の数字 == 0
        else:
            return False
        

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

# 枚数確認
print(f"合計牌数: {len(山)}")  # 136 + 赤ドラ3 = 139

for _ in 山:
    print(_)
