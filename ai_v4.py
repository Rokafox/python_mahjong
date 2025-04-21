import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from 牌 import 聴牌ですか, 麻雀牌, 山を作成する, 面子スコア, 点数計算


class MahjongEnvironment:
    """
    強化学習用の麻雀環境
    """

    N_TILE_TYPES = 34               # 萬・筒・索 (9*3) + 字牌 7
    STATE_BITS = 5                 # 0,1,2,3,4 枚の one‑hot

    def __init__(self):
        self.reset()

    # --------------------------------------------------
    # ゲーム管理
    # --------------------------------------------------
    def reset(self):
        self.山 = 山を作成する()
        self.手牌 = self.山[:13]
        # give tiles to the remaining 3 sandbags, these sandbags is used for simulating 4 player environment,
        # they behave randomly and their only purpose is to train the agent. We should not consider them as real players.
        self.sandbag_a_tiles = self.山[13:26]
        self.sandbag_b_tiles = self.山[26:39]
        self.sandbag_c_tiles = self.山[39:52]
        self.山 = self.山[52:]  # Remaining tiles
        self.discard_pile = []  # Record what tiles are discarded in current game.
        self.seat = random.randint(0, 3)  # 0=東, 1=南, 2=西, 3=北
        self.score = 0
        self.turn  = 0
        self.penalty_456 = 0
        self.tennpai = 0
        self.mz_score = 0
        self.完成した役 = []
        self.hand_when_complete = []
        return self._get_state()


    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS, dtype=np.float32)
        counter = Counter(self._tile_to_index(t) for t in self.手牌)
        for idx in range(self.N_TILE_TYPES):
            cnt = counter.get(idx, 0)
            state[idx * self.STATE_BITS + cnt] = 1.0  # 0〜4
        seat = np.zeros(4, dtype=np.float32) # 東南西北
        seat[self.seat] = 1.0
        # 捨て牌の枚数（連続値）← 最大136枚なので数値として問題なし
        discard_counts = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        for tile in self.discard_pile:
            discard_counts[self._tile_to_index(tile)] += 1.0
        discard_counts /= 4.0  # 正規化（最大4枚）
        return np.concatenate([state, seat, discard_counts]) # → 170 + 4 + 34 = 208 次元


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


    def get_valid_actions(self) -> list[int]:
        return sorted({self._tile_to_index(t) for t in self.手牌})


    def step(self, action: int):
        reward = 0
        if not self.山:
            raise Exception("Trying to call step() when deck is empty.")

        # ① ツモ
        self.手牌.append(self.山.pop(0))
        a, b, c = 点数計算(self.手牌, self.seat) # This func returns 点数int, 完成した役list, 和了形bool
        if c:
            print(f"ツモ！{b}")
            self.完成した役 = b
            reward += int(a)
            done = True
            remaining_turns = 126 - self.turn
            assert remaining_turns >= 0
            reward += int((400 * remaining_turns) * 0.5)
            self.mz_score = 16
            self.score += reward 
            self.hand_when_complete = [str(t) for t in self.手牌]
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_456": self.penalty_456,
                                                    "completeyaku": self.完成した役,
                                                    "mz_score": self.mz_score,
                                                    "hand_when_complete": self.hand_when_complete}
            
        # ② 打牌
        # reward -= max(1, int(self.turn * 0.8)) # Will greatly hinder learning process
        reward -= 2  # base penalty
        tile_type = self._index_to_tile_type(action)
        for i, t in enumerate(self.手牌):
            if (t.何者, t.その上の数字) == tile_type:
                discarded_tile = self.手牌.pop(i)
                self.discard_pile.append(discarded_tile)
                break
        else:
            raise ValueError(f"Invalid action: {action} (tile not in hand)")

        # 聴牌の場合、大量の報酬を与える
        聴牌, 何の牌 = 聴牌ですか(self.手牌)
        if 聴牌:
            reward += 400
            # print(f"聴牌:")
            # for p in 何の牌:
            #     print(f"{p.何者} {p.その上の数字}")
            reward += len(何の牌) * 100

        done = not self.山
        
        # 面子スコア: 13 枚の手牌から完成面子（順子・刻子）の最大数を求めて
        # 0面子→0, 1面子→1, 2面子→2, 3面子→4, 4面子→8を返す。雀頭は数えない。
        self.mz_score = 面子スコア(self.手牌)
        reward += self.mz_score * 8

        # 内部状態更新
        self.score += reward
        self.turn  += 1
        return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_456": self.penalty_456,
                                                "completeyaku": self.完成した役,
                                                "mz_score": self.mz_score,
                                                "hand_when_complete": self.hand_when_complete}



class DQNAgent:

    TARGET_UPDATE_EVERY = 2000

    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size  = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.memory = deque(maxlen=50_000)
        self.gamma   = 0.99
        self.epsilon = 1.0
        self.epsilon_min   = 0.04
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-3

        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self._step_count = 0

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, self.action_size)
        )

    # --------------------------------------------------
    # 行動選択
    # --------------------------------------------------
    def act(self, state: np.ndarray, valid_actions: list[int]) -> int:
        # ε‐greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.model(state_t)[0]          # shape: (34,)
            q_valid = q_vals[valid_actions]          # マスク
            best_idx = q_valid.argmax().item()
            return valid_actions[best_idx]           # 元のタイル番号を返す

    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    def memorize(self, *transition):
        self.memory.append(transition)

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        B = state_batch.size(0)
        tile_state = state_batch[:, :170] # Only the first 170 is for tiles, the last 4 are for seat
        counts = tile_state.view(B, 34, 5).argmax(dim=2)
        return counts > 0

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states      = torch.as_tensor(np.array(states),      dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions,               dtype=torch.int64 , device=self.device).unsqueeze(1)
        rewards     = torch.as_tensor(rewards,               dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(dones,                 dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.model(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_model(next_states)
            valid_mask = self._valid_mask_from_state(next_states)
            q_next[~valid_mask] = -1e9
            q_next_max = q_next.max(1, keepdim=True)[0]
            y = rewards + self.gamma * q_next_max * (1 - dones)

        loss = self.criterion(q_sa, y)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.TARGET_UPDATE_EVERY == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay



def train_agent(episodes: int = 5000, pretrained: str | None = None, device: str = "cpu") -> DQNAgent:
    env = MahjongEnvironment()
    agent = DQNAgent(env.N_TILE_TYPES * env.STATE_BITS + 4 + env.N_TILE_TYPES, env.N_TILE_TYPES, device=device)

    if pretrained and os.path.exists(pretrained):
        agent.model.load_state_dict(torch.load(pretrained, map_location=device))
        agent.update_target_model()
        agent.epsilon = 0.005
        print(f"[INFO] 既存モデル {pretrained} をロードしました。")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            valid = env.get_valid_actions()
            action = agent.act(state, valid)
            next_state, reward, done, info = env.step(action)
            agent.memorize(state, action, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            ep_reward += reward
            if done:
                agent.decay_epsilon()
                if env.seat == 0:
                    envseat = "East"
                elif env.seat == 1:
                    envseat = "South"
                elif env.seat == 2:
                    envseat = "West"
                elif env.seat == 3:
                    envseat = "North"
                with open("training_logv4_0.csv", "a") as f:
                    if ep == 0:  # Write header only once
                        f.write("Episode,Seat,Score,Turn,Epsilon,Penalty456,Yaku,MZScore, HandComplete\n")
                    f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},{agent.epsilon:.3f},{info['penalty_456']},{' '.join(str(x) for x in info['completeyaku'])},{info['mz_score']}, {' '.join(info['hand_when_complete'])}\n")
                break

    return agent


if __name__ == "__main__":
    trained_agent = train_agent(pretrained="model.pth")
    torch.save(trained_agent.model.state_dict(), "model.pth")
