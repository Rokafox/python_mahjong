import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ろか麻雀 import nicely_print_tiles, 対子スコア, 聴牌ですか, 麻雀牌, 山を作成する, 面子スコア, 点数計算


class MahjongEnvironment:
    """
    強化学習用の麻雀環境
    """

    N_TILE_TYPES = 34               # 萬・筒・索 (9*3) + 字牌 7
    STATE_BITS = 5                 # 0,1,2,3,4 枚の one‑hot

    def __init__(self):
        self.agent_tile_count = 13
        self.sandbag_tile_count = 13 # slightly reduced difficulty
        self.reset()
        self.add_tenpai_score = True

    # --------------------------------------------------
    # ゲーム管理
    # --------------------------------------------------

    def 配牌を割り当てる(self, 山: list, agent_count: int, sandbag_count: int):
        assert len(山) >= agent_count + 3 * sandbag_count, "山に十分な牌がありません"

        agent_tiles = 山[:agent_count]
        sandbag_a = 山[agent_count:agent_count + sandbag_count]
        sandbag_b = 山[agent_count + sandbag_count:agent_count + 2 * sandbag_count]
        # sandbag_c = 山[agent_count + 2 * sandbag_count:agent_count + 3 * sandbag_count]
        remaining = 山[agent_count + 2 * sandbag_count:]

        return agent_tiles, sandbag_a, sandbag_b, remaining
    

    def reset(self):
        self.手牌: list[麻雀牌]
        self.山 = 山を作成する()
        (
            self.手牌,
            self.sandbag_a_tiles,
            self.sandbag_b_tiles,
            # self.sandbag_c_tiles,
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


    def _get_state(self, ndtidx: int = -1) -> np.ndarray:
        """
        ntidx: 他家がさき捨てた牌を記録したい時用
        """
        state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS, dtype=np.float32)
        counter = Counter(self._tile_to_index(t) for t in self.手牌)
        # Counter({22: 2, 0: 1, 1: 1, 2: 1, 9: 1})
        for idx in range(self.N_TILE_TYPES):
            cnt = counter.get(idx, 0)
            state[idx * self.STATE_BITS + cnt] = 1.0  # 0〜4
        seat = np.zeros(4, dtype=np.float32) # 東南西北
        seat[self.seat] = 1.0
        discarded_tiles_counts = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        for tile in self.discard_pile:
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
        if self.current_actor == 0:
            # 行動：牌を捨てる
            # 聴牌の時、もらった牌を捨てるだけにしとこう
            if tenpai and extra_tile_idx >= 0:
                return [extra_tile_idx]
            hand_tiles = [self._tile_to_index(t) for t in self.手牌 if not t.副露]
            if extra_tile_idx >= 0:
                hand_tiles.append(extra_tile_idx)
            hand_tiles = sorted(set(hand_tiles))
            return hand_tiles
        else:
            # 行動：何もしない・ポン・チー
            # 34, 35, 36
            return [34, 35, 36]


    def _agent_extra_reward(self, 手牌: list[麻雀牌]) -> int:
        reward_extra = 0
        if self.add_tenpai_score:
            # 聴牌の場合、大量の報酬を与える
            聴牌, 何の牌 = 聴牌ですか(self.手牌, self.seat)
            if 聴牌:
                reward_extra += 800
                # print(f"聴牌:")
                # for p in 何の牌:
                #     print(f"{p.何者} {p.その上の数字}")
                self.total_tennpai += len(何の牌)
                reward_extra += len(何の牌) * 100
            pass
        
        # 面子スコア: 13 枚の手牌から完成面子（順子・刻子）の最大数を求めて
        # 0面子→0, 1面子→1, 2面子→2, 3面子→4, 4面子→8を返す。雀頭は数えない。
        # mz_score = 面子スコア(self.手牌)
        # self.mz_score += mz_score * 9
        # reward_extra += mz_score * 9

        # 対子スコア: 13 枚の手牌から完成対子の最大数を求めて
        # 0対子→0, 1対子→1, 2対子→2, 3対子→4, 4対子→8, 5対子→16, 6対子→32を返す。副露は数えない。
        # tuiz_score = 対子スコア(self.手牌)
        # self.tuiz_score += tuiz_score * 3
        # reward_extra += tuiz_score * 3

        # Big penalty for having 4,5,6.
        # for t in self.手牌:
        #     if t.何者 in {"萬子", "筒子", "索子"} and t.その上の数字 in {4,5,6}:
        #         reward_extra -= 40
        #         self.penalty_A += 40

        # Big reward for having 1,2,3 and 7,8,9 and 字牌
        # for t in self.手牌:
        #     if t.何者 in {"白ちゃん", "發ちゃん", "中ちゃん"}:
        #         reward_extra += 40

        return reward_extra


    def _step_agent(self, actor: "DQNAgent", after_pon: bool):
        reward = 0
        assert len([t for t in self.手牌 if not t.副露]) > 0, "Agent's hand is empty"

        newly_drawn_tile_index: int = -1
        tenpai_before_draw = False

        if not after_pon:
            tenpai_before_draw, list_of_tanpai = 聴牌ですか(self.手牌, self.seat)
            newly_drawn_tile = self.山.pop(0)
            newly_drawn_tile_index = self._tile_to_index(newly_drawn_tile)
            self.手牌.append(newly_drawn_tile)

        # ツモ
        a, b, c = 点数計算(self.手牌, self.seat) # This func returns 点数int, 完成した役list, 和了形bool
        if c:
            print(f"ツモ！{b}")
            self.完成した役 = b
            reward += int(a)
            done = True
            self.score += reward 
            self.hand_when_complete = nicely_print_tiles(self.手牌)
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}, -1
            
        # 牌を捨てる
        valid_actions: list = self.get_valid_actions(extra_tile_idx=newly_drawn_tile_index,
                                                     tenpai=tenpai_before_draw)
        assert len(valid_actions) > 0
        action: int
        action, full_dict = actor.act(self._get_state(), valid_actions)
        if self.add_tenpai_score:
            reward -= 2  # base penalty
        tile_type = self._index_to_tile_type(action)
        # print(tile_type) # ('筒子', 5)
        target_tile = None
        for t in [t for t in self.手牌 if not t.副露]:
            if (t.何者, t.その上の数字) == tile_type:
                target_tile = t
                break
        if target_tile is None:
            raise ValueError(f"Invalid action: {self._index_to_tile_type(action)} not in hand.")
        # Remove the tile from the hand and add it to the discard pile
        assert target_tile.副露 == False, f"Exposed tile can not be discarded! {target_tile.何者}{target_tile.その上の数字}"
        self.手牌.remove(target_tile)
        self.discard_pile.append(target_tile)
    
        # Reward by how good the agent has formed the hand
        reward += self._agent_extra_reward(self.手牌)

        done = not self.山
        self.score += reward
        self.turn  += 1
        self.current_actor += 1  # Next actor is sandbag_a
        # should be a cycle, but agent is always 0
        # if done, add hand
        if done:
            self.hand_when_complete = nicely_print_tiles(self.手牌)
        return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, action


    def _step_sandbag(self, actor: "DQNAgent"):
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
        # elif self.current_actor == 3: # The player is removed
        #     sandbag_tiles = self.sandbag_c_tiles
        
        # Draw tile
        new_tile = self.山.pop(0)
        sandbag_tiles.append(new_tile)
        done = not self.山
    
        # Randomly discards a tile
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
            self.score += reward
            self.hand_when_complete = nicely_print_tiles(temp_hand)
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, -1
    
        if done: # 注意：これで最後はポンできない。これは一般的なルールと違う。
            self.hand_when_complete = nicely_print_tiles(self.手牌)
            return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, -1

        agent_did_nothing = False

        # Only if the agent has more than 2 tiles in hand, pon is possible
        if len([t for t in self.手牌 if not t.副露]) > 1:
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
                valid_actions: list = self.get_valid_actions(-1)
                valid_actions.remove(36)
                assert 34 in valid_actions and 35 in valid_actions, "34 do nothing 35 pon 36 chii"
                action: int
                action, full_dict = actor.act(self._get_state(ndtidx=discarded_tile_idx), valid_actions)
                if action == 35:
                    # print(f"ポン！{discarded_tile.何者} {discarded_tile.その上の数字}")
                    agent_did_nothing = False
                    # Mark the 2 tiles in agent's hand as 副露
                    for i in sorted(same_tiles[:2], reverse=True):
                        self.手牌[i].mark_as_exposed()
                    
                    # Add the discarded tile to agent's hand and mark it as 副露
                    discarded_tile.mark_as_exposed()
                    self.手牌.append(discarded_tile)
                    
                    # Remove the discarded tile from the discard pile
                    self.discard_pile.pop()
                    
                    # Set current actor to agent for immediate play
                    self.current_actor = 0
                    self.agent_after_pon = True
                    return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}, 35
                else: # action 34
                    agent_did_nothing = True

        # Check if agent can Chii
        # Agent can only Chii if they have two consecutive tiles that can form a sequence with the discarded tile
        # Custom rule: agent can only chii if the discarded tile is 中張牌 (2-8), and the only allowed sequence is (n-1, n, n+1)
        # Custom rule: any player can chii from any other player
        if len([t for t in self.手牌 if not t.副露]) > 2:
            can_chii = False
            chii_tiles = []
            
            if "中張牌" in discarded_tile.固有状態:
                dt_num = discarded_tile.その上の数字
                discarded_tile_idx = self._tile_to_index(discarded_tile)
                possible_sequence = [(discarded_tile.何者, dt_num-1), (discarded_tile.何者, dt_num+1)] # [n-1,n,n+1]
                
                # print(f"Checking sequence: {possible_sequence}")
                # Checking sequence: [('筒子', 6), ('筒子', 7)]
                check = 0
                # check if both these tiles are in the hand
                for combo in possible_sequence:
                    # combo: ('筒子', 6) then ('筒子', 7)
                    for p in self.手牌:
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
                    valid_actions: list = self.get_valid_actions(-1)
                    valid_actions.remove(35)
                    assert 34 in valid_actions and 36 in valid_actions, "34 do nothing 35 pon 36 chii"
                    action: int
                    action, full_dict = actor.act(self._get_state(ndtidx=discarded_tile_idx), valid_actions)
                    if action == 36:
                        # print(f"チー！{discarded_tile.何者} {discarded_tile.その上の数字}")
                        agent_did_nothing = False
                        # Mark the 2 tiles in hand that is the chii_tiles to be 副露
                        ct_t_check = 0
                        for ctn in chii_tiles:
                            assert ctn.副露 == False
                        for t in self.手牌:
                            if t.何者 == chii_tiles[0].何者 and t.その上の数字 == chii_tiles[0].その上の数字 and t.赤ドラ == chii_tiles[0].赤ドラ and t.副露 == chii_tiles[0].副露:
                                t.mark_as_exposed()
                                ct_t_check += 1
                                break
                        for t in self.手牌:
                            if t.何者 == chii_tiles[1].何者 and t.その上の数字 == chii_tiles[1].その上の数字 and t.赤ドラ == chii_tiles[1].赤ドラ and t.副露 == chii_tiles[1].副露:
                                t.mark_as_exposed()
                                ct_t_check += 1
                                break
                        assert ct_t_check == 2, "No, not happening"

                        discarded_tile.mark_as_exposed()

                        self.手牌.append(discarded_tile)
                        self.discard_pile.pop()

                        self.current_actor = 0
                        self.agent_after_pon = True  # Reuse the same flag for simplicity
                        return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                        "penalty_A": self.penalty_A,
                                                        "tenpai": self.total_tennpai,
                                                        "mz_score": self.mz_score,
                                                        "tuiz_score": self.tuiz_score,
                                                        "completeyaku": self.完成した役,
                                                        "hand_when_complete": self.hand_when_complete}, 36
                    else: # action 34
                        agent_did_nothing = True



        # Move to next actor (cycle through 0-2) prev: 0-3
        self.current_actor = (self.current_actor + 1) % 3
        self.turn += 1
        final_action = -1
        if agent_did_nothing:
            # print("The agent did not pon or chii.")
            final_action = 34
        # Return state, no reward for agent during sandbag turns
        return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                            "penalty_A": self.penalty_A,
                                            "tenpai": self.total_tennpai,
                                            "mz_score": self.mz_score,
                                            "tuiz_score": self.tuiz_score,
                                            "completeyaku": self.完成した役,
                                            "hand_when_complete": self.hand_when_complete}, final_action


    def step(self, actor: "DQNAgent", after_pon: bool = False):
        # print(self.current_actor) 012 012 012
        if not self.山 and not after_pon:
            raise Exception("Trying to call step() when deck is empty.")
        self.agent_after_pon = False
        if self.current_actor == 0:
            return self._step_agent(actor, after_pon)
        else:
            return self._step_sandbag(actor)


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # determines how much prioritization is used
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        
    def beta_by_frame(self):
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
    
    def __len__(self):
        return len(self.buffer)
        
    def add(self, *transition, priority=None):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if priority is None:
            priority = max_priority
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
            
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta_by_frame())
        weights /= weights.max()
        
        self.frame += 1
        
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority.item()


class EpisodicMemory:
    def __init__(self, capacity=50):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        
    def add_episode(self, episode, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(episode)
            self.priorities.append(reward)
        else:
            # Replace lowest priority episode if this one is better
            min_idx = np.argmin(self.priorities)
            if reward > self.priorities[min_idx]:
                self.memory[min_idx] = episode
                self.priorities[min_idx] = reward
                
    def sample(self, n=1):
        if not self.memory:
            return []
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.memory), min(n, len(self.memory)), p=probs, replace=False)
        return [self.memory[i] for i in indices]


class NewDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        
        # For hand representation (34 tile types)
        self.hand_encoder = nn.Sequential(
            nn.Linear(34*5, 256),  # 34 tile types with counts 0-4
            nn.ReLU()
        )
        
        # For other game state features
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size - 34*5, 128),
            nn.ReLU()
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        # Split input into hand and other features
        hand = x[:, :170]  # First 170 features (34 tiles × 5 possible counts)
        other = x[:, 170:]  # Remaining features
        
        # Process separately
        hand_features = self.hand_encoder(hand)
        state_features = self.state_encoder(other)
        
        # Combine and produce action values
        combined = torch.cat([hand_features, state_features], dim=1)
        return self.combined(combined)


class DQNAgent:
    TARGET_UPDATE_EVERY = 2000

    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        # Use prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=50_000)
        # Add episodic memory for valuable game sequences
        self.episodic_memory = EpisodicMemory(capacity=50)

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998
        self.learning_rate = 1e-3

        # Use the Mahjong-specific model instead of the sequential model
        self.model = NewDQN(state_size, action_size).to(self.device)
        self.target_model = NewDQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        self._step_count = 0

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
        # Use add method of prioritized replay buffer
        self.memory.add(*transition)
        
    def add_episode_to_memory(self, episode, reward):
        # Add complete episodes with high rewards to episodic memory
        self.episodic_memory.add_episode(episode, reward)

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        B = state_batch.size(0)
        tile_state = state_batch[:, :170]  # [B, 170]
        counts = tile_state.view(B, 34, 5).argmax(dim=2)  # [B, 34]
        mask34 = counts > 0  # [B, 34]
        mask_other_action = torch.ones((B, 3), dtype=torch.bool, device=state_batch.device)
        full_mask = torch.cat([mask34, mask_other_action], dim=1)  # [B, 37]
        return full_mask

    def replay(self, batch_size: int = 64):
        if len(self.memory) < batch_size:
            return
            
        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # Convert to tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Filter out invalid actions (-1)
        valid_indices = actions.squeeze(1) != -1
        states = states[valid_indices]
        actions = actions[valid_indices]
        rewards = rewards[valid_indices]
        next_states = next_states[valid_indices]
        dones = dones[valid_indices]
        weights = weights[valid_indices]
        
        # Update indices to match filtered tensors
        valid_indices_list = valid_indices.cpu().numpy()
        valid_idx_map = [i for i, valid in enumerate(valid_indices_list) if valid]
        filtered_indices = [indices[i] for i in valid_idx_map]

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

        # TD error for prioritization
        td_error = torch.abs(q_sa - y).detach().cpu().numpy()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(filtered_indices, td_error + 1e-6)  # Small constant to ensure all priorities > 0
        
        # Weighted MSE loss
        loss = (weights * (q_sa - y) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self._step_count += 1
        if self._step_count % self.TARGET_UPDATE_EVERY == 0:
            self.update_target_model()
            
        # Occasionally sample from episodic memory to reinforce learning from successful episodes
        if self._step_count % 500 == 0 and self.episodic_memory.memory:
            self._replay_from_episodic_memory()
            
    def _replay_from_episodic_memory(self, num_episodes=1):
        # Sample successful episodes and learn from them
        sampled_episodes = self.episodic_memory.sample(num_episodes)
        for episode in sampled_episodes:
            for transition in episode:
                self.memory.add(*transition, priority=2.0)  # Higher priority for good episodes

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes: int = 1200000, pretrained: str | None = None,
                device: str = "cpu", log_save_path: str = None) -> DQNAgent:
    env = MahjongEnvironment()
    agent = DQNAgent(242, 34 + 3, device=device)
    if os.path.exists(log_save_path):
        raise FileExistsError(f"Log file {log_save_path} already exists. Please choose a different name.")
    if pretrained and os.path.exists(pretrained):
        agent.model.load_state_dict(torch.load(pretrained, map_location=device))
        agent.update_target_model()
        agent.epsilon = 0.01
        print(f"[INFO] 既存モデル {pretrained} をロードしました。")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            next_state, reward, done, info, action = env.step(agent, env.agent_after_pon)
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
                with open(log_save_path, "a") as f:
                    if ep == 0:  # Write header only once
                        f.write("Episode,Seat,Score,Turn,Epsilon,penalty_A,Yaku,MZ_Score,TZ_Score,Tenpai,HandComplete\n")
                    f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},{agent.epsilon:.3f},{info['penalty_A']},{' '.join(str(x) for x in info['completeyaku'])},{info['mz_score']},{info['tuiz_score']},{info['tenpai']},{info['hand_when_complete']}\n")

                # モデル保存
                if (ep + 1) % 1000 == 0:
                    save_path = f"Kurt_1_v80_ep{ep+1}.pth"
                    torch.save(agent.model.state_dict(), save_path)
                    print(f"[INFO] モデルを保存しました: {save_path}")
                break

    return agent


def test_agent(episodes: int, model_path: str, 
               device: str = "cpu", log_save_path: str = None) -> None:
    env = MahjongEnvironment()
    env.add_tenpai_score = False
    agent = DQNAgent(242, 34 + 3, device=device)
    
    # Load pre-trained model
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] テストモデル {model_path} をロードしました。")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    # Set epsilon to 0 for deterministic policy (no exploration)
    agent.epsilon = 0
    
    # Create log file if specified
    if log_save_path:
        if os.path.exists(log_save_path):
            raise FileExistsError(f"Log file {log_save_path} already exists. Please choose a different name.")
        with open(log_save_path, "a") as f:
            f.write("Episode,Seat,Score,Turn,Yaku,MZ_Score,TZ_Score,Tenpai,HandComplete\n")
    
    # Statistics tracking
    tenpai_count = 0
    agari = 0
    total_score = 0
    
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        
        while True:
            # Get action from agent but don't train (no memorize or replay)
            next_state, reward, done, info, action = env.step(agent, env.agent_after_pon)
            state = next_state
            ep_reward += reward
            
            if done:
                # Convert seat number to compass direction
                seat_names = ["East", "South", "West", "North"]
                envseat = seat_names[env.seat]
                
                # Track statistics
                total_score += info['score']
                # if 'tenpai' in info and info['tenpai'] > 0:
                #     tenpai_count += 1
                if 'completeyaku' in info and info['completeyaku']:
                    agari += 1
                
                # Log results
                if log_save_path:
                    with open(log_save_path, "a") as f:
                        f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},"
                                f"{' '.join(str(x) for x in info['completeyaku'])},"
                                f"{info['mz_score']},{info['tuiz_score']},"
                                f"{info['tenpai']},{info['hand_when_complete']}\n")
                
                # Print progress
                if (ep + 1) % 500 == 0:
                    print(f"[TEST] Episode {ep+1}/{episodes} completed. "
                          f"Avg score: {total_score/(ep+1):.2f}, "
                        #   f"Tenpai rate: {tenpai_count/(ep+1)*100:.2f}%, "
                          f"Agari rate: {agari/(ep+1)*100:.2f}%")
                break
    
    # Final statistics
    print(f"\n[TEST COMPLETE] Total episodes: {episodes}")
    print(f"Average score: {total_score/episodes:.2f}")
    # print(f"Tenpai rate: {tenpai_count/episodes*100:.3f}%")
    print(f"Agari rate: {agari/episodes*100:.3f}%")




if __name__ == "__main__":
    # trained_agent = train_agent(pretrained="Kurt_v1.pth", device="cuda",
    #                             log_save_path="./log/kurt_2_logv80.csv")
    # torch.save(trained_agent.model.state_dict(), "modelv6.pth")
    test_agent(episodes=5000, model_path="Brett.pth", device="cuda",
               log_save_path="./log/testing_Brett.csv")
