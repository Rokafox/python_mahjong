import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from 牌 import 対子スコア, 聴牌ですか, 麻雀牌, 山を作成する, 面子スコア, 点数計算


class MahjongEnvironment:
    """
    強化学習用の麻雀環境
    """

    N_TILE_TYPES = 34               # 萬・筒・索 (9*3) + 字牌 7
    STATE_BITS = 5                 # 0,1,2,3,4 枚の one‑hot

    def __init__(self):
        self.agent_tile_count = 13
        self.sandbag_tile_count = 6 # slightly reduced difficulty
        self.reset()

    # --------------------------------------------------
    # ゲーム管理
    # --------------------------------------------------

    def 配牌を割り当てる(self, 山: list, agent_count: int, sandbag_count: int):
        assert len(山) >= agent_count + 3 * sandbag_count, "山に十分な牌がありません"

        agent_tiles = 山[:agent_count]
        sandbag_a = 山[agent_count:agent_count + sandbag_count]
        sandbag_b = 山[agent_count + sandbag_count:agent_count + 2 * sandbag_count]
        sandbag_c = 山[agent_count + 2 * sandbag_count:agent_count + 3 * sandbag_count]
        remaining = 山[agent_count + 3 * sandbag_count:]

        return agent_tiles, sandbag_a, sandbag_b, sandbag_c, remaining
    

    def reset(self):
        self.山 = 山を作成する()
        (
            self.手牌,
            self.sandbag_a_tiles,
            self.sandbag_b_tiles,
            self.sandbag_c_tiles,
            self.山
        ) = self.配牌を割り当てる(
            self.山,
            agent_count=self.agent_tile_count,
            sandbag_count=self.sandbag_tile_count
        )
        self.current_actor = 0  # 0=agent, 1=sandbag_a, 2=sandbag_b, 3=sandbag_c
        self.discard_pile = []  # Record what tiles are discarded in current game.
        self.seat = random.randint(0, 3)  # 0=東, 1=南, 2=西, 3=北
        self.score = 0
        self.turn  = 0
        self.penalty_A = 0
        self.total_tennpai = 0
        self.mz_score = 0
        self.tuiz_score = 0
        self.pon = 0
        self.完成した役 = []
        self.hand_when_complete = []
        self.agent_after_pon = False
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
        return np.concatenate([state, seat, discard_counts]) # → 170 + 4 + 34 = 208


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
        if self.current_actor == 0:
            return sorted({self._tile_to_index(t) for t in self.手牌 if not t.副露})
        else:
            hand_counts = Counter(self._tile_to_index(t) for t in self.手牌 if not t.副露)
            pon_actions = []
            for tile_idx, cnt in hand_counts.items():
                if cnt >= 2:
                    pon_actions.append(34 + tile_idx) # action 34 - 67: pon true
                    pon_actions.append(68 + tile_idx) # action 68 - 102: pon false
            return pon_actions


    def _agent_extra_reward(self, 手牌: list[麻雀牌]) -> int:
        reward_extra = 0
        # 聴牌の場合、大量の報酬を与える
        聴牌, 何の牌 = 聴牌ですか(self.手牌)
        if 聴牌:
            reward_extra += 600
            # print(f"聴牌:")
            # for p in 何の牌:
            #     print(f"{p.何者} {p.その上の数字}")
            self.total_tennpai += len(何の牌)
            reward_extra += len(何の牌) * 200
        
        # 面子スコア: 13 枚の手牌から完成面子（順子・刻子）の最大数を求めて
        # 0面子→0, 1面子→1, 2面子→2, 3面子→4, 4面子→8を返す。雀頭は数えない。
        mz_score = 面子スコア(self.手牌)
        self.mz_score += mz_score * 10
        reward_extra += mz_score * 10

        # 対子スコア: 13 枚の手牌から完成対子の最大数を求めて
        # 0対子→0, 1対子→1, 2対子→2, 3対子→4, 4対子→8, 5対子→16, 6対子→32を返す。
        # tuiz_score = 対子スコア(self.手牌)
        # self.tuiz_score += tuiz_score * 2
        # reward_extra += tuiz_score * 2

        # Give penalty for having 4,5,6, the more the agent has, the more penalty it gets.
        # for t in self.手牌:
        #     if t.何者 in {"萬子", "筒子", "索子"} and t.その上の数字 in {4,5,6}:
        #         reward_extra -= 4
        #         self.penalty_A += 4

        return reward_extra


    def _step_agent(self, action: int, after_pon: bool):
        reward = 0

        # ① ツモ
        if not after_pon:
            self.手牌.append(self.山.pop(0))
        a, b, c = 点数計算(self.手牌, self.seat) # This func returns 点数int, 完成した役list, 和了形bool
        if c:
            print(f"ツモ！{b}")
            self.完成した役 = b
            reward += int(a)
            done = True
            remaining_turns = 140 - self.turn
            assert remaining_turns >= 0
            reward += int((300 * remaining_turns) * 0.5)
            self.mz_score = 16
            self.score += reward 
            self.hand_when_complete = [str(t) for t in self.手牌]
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "completeyaku": self.完成した役,
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

        reward += self._agent_extra_reward(self.手牌)

        done = not self.山
        self.score += reward
        self.turn  += 1
        self.current_actor += 1  # Next actor is sandbag_a
        # should be a cycle, but agent is always 0
        return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}


    def _step_sandbag(self, action: int, full_action_value_dict: dict):
        # The sandbag cannot do anything but randomly discard a tile,
        # and the discarded tile is then added to the discard pile.
        # during sandbag's turn, the agent can "ロン", "ポン" the tile.
        # "ロン" is automatically checked like ツモ.
        # "ロン": try add the tile to the hand and check if it is a ツモ,
        # if so, it is a "ロン".
        # "ポン": agent, if having the 2 same tiles as the discarded tile,
        # can "ポン" the tile and add it to the hand, if so, mark the 3 tiles as 副露
        # by setting 牌.副露 = True and remove the tile in pile. 
        # After ポン, current_actor is set to the agent but
        # the agent cannot draw a tile this turn.
        # To decide whether to "ポン", use the action value dict, pon
        # if 34 + tile_idx > 68 + tile_idx, otherwise do not pon.


        # Determine which sandbag is currently active
        sandbag_tiles = None
        if self.current_actor == 1:
            sandbag_tiles = self.sandbag_a_tiles
        elif self.current_actor == 2:
            sandbag_tiles = self.sandbag_b_tiles
        elif self.current_actor == 3:
            sandbag_tiles = self.sandbag_c_tiles
        
        # Draw a tile for the sandbag if there are tiles left in the mountain
        new_tile = self.山.pop(0)
        sandbag_tiles.append(new_tile)
        done = not self.山
    
        # Sandbag randomly discards a tile
        discard_idx = random.randint(0, len(sandbag_tiles) - 1)
        discarded_tile = sandbag_tiles.pop(discard_idx)
        self.discard_pile.append(discarded_tile)

        # Check if agent can ロン (win) with this tile
        temp_hand = self.手牌.copy()
        temp_hand.append(discarded_tile)
        点数, 役, 和了形 = 点数計算(temp_hand, self.seat)
        
        if 和了形:
            # Agent can ロン
            print(f"ロン！{役}")
            self.完成した役 = 役
            reward = int(点数)
            done = True
            remaining_turns = 140 - self.turn
            assert remaining_turns >= 0
            reward += int((300 * remaining_turns) * 0.5)
            self.score += reward
            self.hand_when_complete = [str(t) for t in temp_hand]
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}
    
        if done: # 注意：これで最後はポンできない。これは一般的なルールと違う。
            return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}
    
        # Check if agent can ポン (pong)
        # Count how many of the discarded tile type the agent has
        same_tile_count = 0
        same_tiles = []
        discarded_tile_idx = self._tile_to_index(discarded_tile)
        
        for i, tile in enumerate(self.手牌):
            if (tile.何者, tile.その上の数字) == (discarded_tile.何者, discarded_tile.その上の数字) and not tile.副露:
                same_tile_count += 1
                same_tiles.append(i)
        
        # If agent has at least 2 of the same tile, they can pon
        if same_tile_count >= 2:
            # The agent will decide to pon or not based on the action value dictionary
            pon_action = 34 + discarded_tile_idx
            no_pon_action = 68 + discarded_tile_idx
            
            if full_action_value_dict and pon_action in full_action_value_dict and no_pon_action in full_action_value_dict:
                if full_action_value_dict[pon_action] > full_action_value_dict[no_pon_action]:
                    # print(f"ポン！{discarded_tile.何者} {discarded_tile.その上の数字}")
                    # Agent decides to pon
                    # Mark the 2 tiles in agent's hand as 副露
                    for i in sorted(same_tiles[:2], reverse=True):
                        self.手牌[i].副露 = True
                    
                    # Add the discarded tile to agent's hand and mark it as 副露
                    discarded_tile.副露 = True
                    self.手牌.append(discarded_tile)
                    
                    # Remove the discarded tile from the discard pile
                    self.discard_pile.pop()
                    
                    # Set current actor to agent for immediate play
                    self.current_actor = 0
                    self.agent_after_pon = True
                    # Return state with after_pon=True for next step
                    return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}

        # Move to next actor (cycle through 0-3)
        self.current_actor = (self.current_actor + 1) % 4
        self.turn += 1
        
        # Return state, no reward for agent during sandbag turns
        return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                            "penalty_A": self.penalty_A,
                                            "tenpai": self.total_tennpai,
                                            "mz_score": self.mz_score,
                                            "tuiz_score": self.tuiz_score,
                                            "completeyaku": self.完成した役,
                                            "hand_when_complete": self.hand_when_complete}


    def step(self, action: int, after_pon: bool = False, full_action_value_dict: dict = None):
        if not self.山 and not after_pon:
            raise Exception("Trying to call step() when deck is empty.")
        self.agent_after_pon = False
        if self.current_actor == 0:
            return self._step_agent(action, after_pon)
        else:
            return self._step_sandbag(action, full_action_value_dict)



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
    def act(self, state: np.ndarray, valid_actions: list[int]):
        # ε‐greedy
        if random.random() < self.epsilon:
            return random.choice(valid_actions), dict()

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.model(state_t)[0]  # shape: [102]
            # 有効アクションの中から最大Q値のアクションを選ぶ
            q_valid = q_vals[valid_actions]
            best_idx = q_valid.argmax().item()
            # すべてのアクションに対するQ値の辞書を返す
            q_dict = {i: q_vals[i].item() for i in range(len(q_vals))}
            return valid_actions[best_idx], q_dict


    # --------------------------------------------------
    # 学習
    # --------------------------------------------------
    def memorize(self, *transition):
        self.memory.append(transition)

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        B = state_batch.size(0)
        tile_state = state_batch[:, :170]  # [B, 170]
        counts = tile_state.view(B, 34, 5).argmax(dim=2)  # [B, 34]
        mask34 = counts > 0  # [B, 34]
        mask_rest = torch.ones((B, 68), dtype=torch.bool, device=state_batch.device)
        full_mask = torch.cat([mask34, mask_rest], dim=1)  # [B, 102]
        return full_mask

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

        # Filter out invalid actions (-1)
        valid_indices = actions.squeeze(1) != -1
        states = states[valid_indices]
        actions = actions[valid_indices]
        rewards = rewards[valid_indices]
        next_states = next_states[valid_indices]
        dones = dones[valid_indices]

        if len(states) == 0:  # Skip if no valid actions in the batch
            return

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



def train_agent(episodes: int = 5000, pretrained: str | None = None, device: str = "cuda") -> DQNAgent:
    env = MahjongEnvironment()
    agent = DQNAgent(208, env.N_TILE_TYPES * 3, device=device)

    if pretrained and os.path.exists(pretrained):
        agent.model.load_state_dict(torch.load(pretrained, map_location=device))
        agent.update_target_model()
        agent.epsilon = 0.035
        print(f"[INFO] 既存モデル {pretrained} をロードしました。")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            valid: list = env.get_valid_actions()
            if valid:
                action, avd = agent.act(state, valid)
            else:
                action = -1
                avd = None
            # if exist info["after_pon"],
            next_state, reward, done, info = env.step(action, env.agent_after_pon, full_action_value_dict=avd)
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
                with open("training_logv4_1.csv", "a") as f:
                    if ep == 0:  # Write header only once
                        f.write("Episode,Seat,Score,Turn,Epsilon,penalty_A,Yaku,MZ_Score,TZ_Score,Tenpai,HandComplete\n")
                    f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},{agent.epsilon:.3f},{info['penalty_A']},{' '.join(str(x) for x in info['completeyaku'])},{info['mz_score']},{info['tuiz_score']},{info['tenpai']},{' '.join(info['hand_when_complete'])}\n")
                break

    return agent


if __name__ == "__main__":
    trained_agent = train_agent(pretrained="model.pth")
    torch.save(trained_agent.model.state_dict(), "model.pth")
