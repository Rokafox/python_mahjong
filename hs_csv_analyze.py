import csv

def average_turn_from_csv(filename):
    turns = []
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                turn = int(row["Turn"])
                turns.append(turn)
            except ValueError:
                continue  # 空欄や不正なデータをスキップ
    if not turns:
        print("Turn データが見つかりませんでした。")
        return
    avg_turn = sum(turns) / len(turns)
    print(f"平均 Turn: {avg_turn:.2f}")

# 実行（例: ファイル名が data.csv の場合）
average_turn_from_csv("data.csv")
