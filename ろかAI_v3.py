import collections
import csv
from itertools import permutations
import os
import random
from collections import Counter, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ろか麻雀 import calculate_weighted_preference_score, nicely_print_tiles, 対子スコア, 搭子スコア, 聴牌ですか, 麻雀牌, 山を作成する, 面子スコア, 点数計算


class MahjongEnvironment:
    """
    強化学習用の麻雀環境
    """

    N_TILE_TYPES = 34               # 萬・筒・索 (9*3) + 字牌 7
    STATE_BITS = 5                 # 0,1,2,3,4 枚の one‑hot

    def __init__(self):
        self.agent_tile_count = 13
        self.sandbag_tile_count = 13
        self.reset()
        # test environment does not give penalty nor hand reward
        self.is_test_environment = False

    # --------------------------------------------------
    # ゲーム管理
    # --------------------------------------------------

    def 配牌を割り当てる(self, 山: list, agent_count: int, sandbag_count: int):
        assert len(山) >= agent_count + 3 * sandbag_count, "山に十分な牌がありません"

        agent_tiles = 山[:agent_count]
        sandbag_a = 山[agent_count:agent_count + sandbag_count]
        # sandbag_b = 山[agent_count + sandbag_count:agent_count + 2 * sandbag_count]
        # sandbag_c = 山[agent_count + 2 * sandbag_count:agent_count + 3 * sandbag_count]
        remaining = 山[agent_count + sandbag_count:]

        return agent_tiles, sandbag_a, remaining
    

    def reset(self):
        self.手牌: list[麻雀牌]
        self.山 = 山を作成する()
        # self.山 = 基礎訓練山を作成する()
        (
            self.手牌,
            self.sandbag_a_tiles,
            # self.sandbag_b_tiles,
            # self.sandbag_c_tiles, # Sandbag removed
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
        self.tatsu_score = 0
        self.pon = 0
        self.完成した役 = []
        self.hand_when_complete = []
        self.agent_after_pon = False

        return self._get_state()


    def _get_state(self, ndtidx: int = -1) -> np.ndarray:
        """
        ntidx: 他家がさき捨てた牌を記録したい時用
        """
        hand_state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS, dtype=np.float32)
        hand_exposed_state = np.zeros(self.N_TILE_TYPES * self.STATE_BITS, dtype=np.float32)
        counter_hand = Counter(self._tile_to_index(t) for t in self.手牌 if not t.副露)
        counter_hand_exposed = Counter(self._tile_to_index(t) for t in self.手牌 if t.副露)
        # Counter({22: 2, 0: 1, 1: 1, 2: 1, 9: 1})
        for idx in range(self.N_TILE_TYPES):
            cnt = counter_hand.get(idx, 0)
            cnt_1 = counter_hand_exposed.get(idx, 0)
            hand_state[idx * self.STATE_BITS + cnt] = 1.0  # 0〜4
            hand_exposed_state[idx * self.STATE_BITS + cnt_1] = 1.0  # 0〜4
        seat = np.zeros(4, dtype=np.float32) # 東南西北
        seat[self.seat] = 1.0
        discarded_tiles_counts = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        for tile in self.discard_pile:
            discarded_tiles_counts[self._tile_to_index(tile)] += 1.0
        discarded_tiles_counts /= 4.0  # 正規化（最大4枚）
        # NOTE: 実戦中他家の副露も捨て牌として考えればよい
        new_discarded_tile_by_others = np.zeros(self.N_TILE_TYPES, dtype=np.float32)
        if ndtidx >= 0:
            new_discarded_tile_by_others[ndtidx] = 1.0
        # 170 + 170 + 4 + 34 + 34 = 412
        return np.concatenate([hand_state, hand_exposed_state, seat, discarded_tiles_counts, new_discarded_tile_by_others])


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

        # 聴牌の場合、大量の報酬を与える
        # NOTE: Check tenpai every turn is slow
        # 聴牌, 何の牌 = 聴牌ですか(self.手牌, self.seat)
        # if 聴牌:
        #     reward_extra += 300
        #     # print(f"聴牌:")
        #     # for p in 何の牌:
        #     #     print(f"{p.何者} {p.その上の数字}")
        #     self.total_tennpai += len(何の牌)
        #     reward_extra += len(何の牌) * 50
        
        mz_score = int(面子スコア(self.手牌) * 8)
        self.mz_score += mz_score
        reward_extra += mz_score

        tuiz_score = int(対子スコア(self.手牌) * 4)
        self.tuiz_score += tuiz_score
        reward_extra += tuiz_score

        tatsu_score = int(搭子スコア(self.手牌) * 4)
        self.tatsu_score += tatsu_score
        reward_extra += tatsu_score

        ht_counter = Counter((t.何者, t.その上の数字) for t in self.手牌)

        # Penalty for having 4,5,6.
        # for t in self.手牌:
        #     if t.何者 in {"萬子", "筒子", "索子"} and t.その上の数字 in {4,5,6}:
        #         reward_extra -= 6
        #         self.penalty_A += 6
        #     else:
        #         reward_extra += 3
        #         self.penalty_A -= 3

        # for key, cnt in ht_counter.items():
        #     if key[1] == 2:
        #         if ht_counter[(key[0], 3)] != cnt or ht_counter[(key[0], 1)] < cnt:
        #             reward_extra -= 6
        #             self.penalty_A += 6
        #     if key[1] == 3:
        #         if ht_counter[(key[0], 2)] != cnt or ht_counter[(key[0], 1)] < cnt:
        #             reward_extra -= 6
        #             self.penalty_A += 6
        #     if key[1] == 8:
        #         if ht_counter[(key[0], 7)] != cnt or ht_counter[(key[0], 9)] < cnt:
        #             reward_extra -= 6
        #             self.penalty_A += 6
        #     if key[1] == 7:
        #         if ht_counter[(key[0], 8)] != cnt or ht_counter[(key[0], 9)] < cnt:
        #             reward_extra -= 6
        #             self.penalty_A += 6


        # Penalty for having exposed tiles
        # the_exposed = [t for t in self.手牌 if t.副露]
        # punishment_of_the_exposed = len(the_exposed) * 30
        # reward_extra -= punishment_of_the_exposed
        # self.penalty_A += punishment_of_the_exposed

        # Custom Penalty
        for t in self.手牌:
            if "字牌" in t.固有状態:
                reward_extra += 15
                self.penalty_A -= 15

            # if t.何者 in {"萬子"} and t.その上の数字 in {1,2,3}:
            #     reward_extra += 3
            #     self.penalty_A -= 3
            # elif t.何者 in {"索子"} and t.その上の数字 in {4,5,6}:
            #     reward_extra += 3
            #     self.penalty_A -= 3
            # elif t.何者 in {"筒子"} and t.その上の数字 in {7,8,9}:
            #     reward_extra += 3
            #     self.penalty_A -= 3
            else:
                reward_extra -= 15
                self.penalty_A += 15


        # suits = ("萬子", "筒子", "索子")
        # # Define the three sequence groups
        # sequences = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
        
        # # For rule 3: Check if we can assign one sequence to each 萬子
        # # Then remove that sequence, check if we can assign 1 to 筒子,
        # # then, check if we can assign the last one to 索子.
        # # We also need to consider there could be mulitiple seq to assign to each suit, try all of them.
        # # Check if each suit can claim at least one complete sequence
        # suit_sequences = {}
        # for suit in suits:
        #     suit_sequences[suit] = []
        #     for seq in sequences:
        #         # Check if this suit has all numbers in this sequence
        #         if all(ht_counter.get((suit, n), 0) >= 1 for n in seq):
        #             suit_sequences[suit].append(seq)
        #             reward_extra += 6
        #             self.penalty_A -= 6
        
        # # For each possible ordering of the sequences
        # for seq_ordering in permutations(sequences):
        #     # Try to assign sequences to suits in this order
        #     assignment_works = True
        #     used_suits = set()
        #     suit_to_seq = {}  # Map suits to their assigned sequences
            
        #     for seq in seq_ordering:
        #         # Find a suit that can claim this sequence and hasn't been used
        #         assigned = False
        #         for suit in suits:
        #             if suit not in used_suits and seq in suit_sequences[suit]:
        #                 used_suits.add(suit)
        #                 suit_to_seq[suit] = seq  # Store the assignment
        #                 assigned = True
        #                 break
                
        #         if not assigned:
        #             assignment_works = False
        #             break
            
        #     # If we found a valid assignment, return True
        #     if assignment_works and len(used_suits) == len(suits):
        #         reward_extra += 80
        #         self.penalty_A -= 80

        # Custom 7 tui penalty
        # counter = Counter((t.何者, t.その上の数字) for t in self.手牌)
        # for key, cnt in counter.items():
        #     if cnt == 2 or cnt == 4:
        #         reward_extra += 9
        #         self.penalty_A -= 9

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
                                                    "tatsu_score": self.tatsu_score,
                                                    "completeyaku": self.完成した役,
                                                    "hand_when_complete": self.hand_when_complete}, -1
            
        # 牌を捨てる
        valid_actions: list = self.get_valid_actions(extra_tile_idx=newly_drawn_tile_index,
                                                     tenpai=tenpai_before_draw)
        assert len(valid_actions) > 0
        action: int
        action, full_dict = actor.act(self._get_state(), valid_actions)
        if not self.is_test_environment:
            # reward -= 2  # base penalty
            reward -= int(self.turn)
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

        if not self.is_test_environment:
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
                                                "tatsu_score": self.tatsu_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, action


    def _step_sandbag(self, actor: "DQNAgent"):
        reward = 0
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
        # elif self.current_actor == 2: # removed
        #     sandbag_tiles = self.sandbag_b_tiles 
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
            reward += int(点数)
            done = True
            self.score += reward
            self.hand_when_complete = nicely_print_tiles(temp_hand)
            return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "tatsu_score": self.tatsu_score,
                                                "completeyaku": self.完成した役,
                                                "hand_when_complete": self.hand_when_complete}, -1
    
        if done: # 注意：これで最後はポンできない。これは一般的なルールと違う。
            self.hand_when_complete = nicely_print_tiles(self.手牌)
            return self._get_state(), 0, done, {"score": self.score, "turn": self.turn,
                                                "penalty_A": self.penalty_A,
                                                "tenpai": self.total_tennpai,
                                                "mz_score": self.mz_score,
                                                "tuiz_score": self.tuiz_score,
                                                "tatsu_score": self.tatsu_score,
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
                        self.手牌[i].mark_as_exposed("pon")
                    
                    # Add the discarded tile to agent's hand and mark it as 副露
                    discarded_tile.mark_as_exposed("pon")
                    self.手牌.append(discarded_tile)
                    
                    # Remove the discarded tile from the discard pile
                    self.discard_pile.pop()
                    
                    # Set current actor to agent for immediate play
                    self.current_actor = 0
                    self.agent_after_pon = True

                    if not self.is_test_environment:
                        reward += self._agent_extra_reward(self.手牌)
                    self.score += reward

                    return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                    "penalty_A": self.penalty_A,
                                                    "tenpai": self.total_tennpai,
                                                    "mz_score": self.mz_score,
                                                    "tuiz_score": self.tuiz_score,
                                                    "tatsu_score": self.tatsu_score,
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
                                t.mark_as_exposed("chii")
                                ct_t_check += 1
                                break
                        for t in self.手牌:
                            if t.何者 == chii_tiles[1].何者 and t.その上の数字 == chii_tiles[1].その上の数字 and t.赤ドラ == chii_tiles[1].赤ドラ and t.副露 == chii_tiles[1].副露:
                                t.mark_as_exposed("chii")
                                ct_t_check += 1
                                break
                        assert ct_t_check == 2, "No, not happening"

                        discarded_tile.mark_as_exposed("chii")

                        self.手牌.append(discarded_tile)
                        self.discard_pile.pop()

                        self.current_actor = 0
                        self.agent_after_pon = True  # Reuse the same flag for simplicity

                        if not self.is_test_environment:
                            reward += self._agent_extra_reward(self.手牌)
                        self.score += reward

                        return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                                        "penalty_A": self.penalty_A,
                                                        "tenpai": self.total_tennpai,
                                                        "mz_score": self.mz_score,
                                                        "tuiz_score": self.tuiz_score,
                                                        "tatsu_score": self.tatsu_score,
                                                        "completeyaku": self.完成した役,
                                                        "hand_when_complete": self.hand_when_complete}, 36
                    else: # action 34
                        agent_did_nothing = True



        # Move to next actor (cycle through 0-2) prev: 0-3
        # 2025-04-25 NEW: only 2 players left
        self.current_actor = (self.current_actor + 1) % 2
        self.turn += 1
        final_action = -1
        if agent_did_nothing:
            # print("The agent did not pon or chii.")
            final_action = 34

        if not self.is_test_environment:
            reward += self._agent_extra_reward(self.手牌)
        self.score += reward

        # Return state
        return self._get_state(), reward, done, {"score": self.score, "turn": self.turn,
                                            "penalty_A": self.penalty_A,
                                            "tenpai": self.total_tennpai,
                                            "mz_score": self.mz_score,
                                            "tuiz_score": self.tuiz_score,
                                            "tatsu_score": self.tatsu_score,
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
    def __init__(self, capacity=300):
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
        # probs = np.array(self.priorities) / sum(self.priorities)
        # Random sample
        indices = np.random.choice(len(self.memory), min(n, len(self.memory)), p=None, replace=False)
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
        # self.combined = nn.Sequential(
        #     nn.Linear(256 + 128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, action_size)
        # )
        self.combined = nn.Sequential(
            nn.Linear(256 + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
    EPISODIC_MEMORY_REPLAY_FREQ = 500
    SUCCESS_REWARD_THRESHOLD = 3000

    def __init__(self, state_size: int, action_size: int, device: str = "cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)

        self.memory = PrioritizedReplayBuffer(capacity=50_000)
        self.episodic_memory = EpisodicMemory(capacity=3333)
        self.current_episode_transitions = []

        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 1e-3

        self.model = NewDQN(state_size, action_size).to(self.device)
        self.target_model = NewDQN(state_size, action_size).to(self.device)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self._step_count = 0

    def act(self, state: np.ndarray, valid_actions: list[int]):
        # ε‐greedy
        if random.random() < self.epsilon:
            # Return a random action from valid_actions
            # If valid_actions is empty, this will raise an error, need to handle that
            if not valid_actions:
                 return -1, {} # Return an invalid action if no valid actions (shouldn't happen in Mahjong normally)
            return random.choice(valid_actions), dict()

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_vals = self.model(state_t)[0]  # shape: [action_size]
            # Mask invalid actions with a very low value
            q_vals_masked = q_vals.clone()
            invalid_action_indices = [i for i in range(self.action_size) if i not in valid_actions]
            q_vals_masked[invalid_action_indices] = -float('inf') # Use -inf to ensure they are not chosen

            # Select the action with the highest Q value among valid actions
            best_action_idx = torch.argmax(q_vals_masked).item()

            # Return Q values for all actions (for logging/debugging if needed)
            q_dict = {i: q_vals[i].item() for i in range(self.action_size)}
            return best_action_idx, q_dict

    def memorize(self, *transition):
        # Add transition to the main replay buffer
        self.memory.add(*transition)
        # Also append to the current episode's transitions
        self.current_episode_transitions.append(transition)

    def add_episode_to_memory(self, episode: list, reward: float):
        # Add complete episodes with high rewards to episodic memory
        # This method is called externally after an episode finishes
        self.episodic_memory.add_episode(episode, reward)
        print("Episode memorized.")

    def _valid_mask_from_state(self, state_batch: torch.Tensor) -> torch.Tensor:
        B = state_batch.size(0)
        tile_state = state_batch[:, :170]  # [B, 170]
        counts = tile_state.view(B, 34, 5).argmax(dim=2)  # [B, 34]
        mask34 = counts > 0  # [B, 34]
        mask_other_action = torch.ones((B, 3), dtype=torch.bool, device=state_batch.device)
        full_mask = torch.cat([mask34, mask_other_action], dim=1)  # [B, 37]
        return full_mask

    def replay(self, batch_size: int = 64):
        # Periodically sample from episodic memory and add to main buffer
        if self._step_count % self.EPISODIC_MEMORY_REPLAY_FREQ == 0 and len(self.episodic_memory.memory) > 0:
            self._replay_from_episodic_memory()
            # print("Successful episode replayed.")

        if len(self.memory) < batch_size:
            return

        # Sample from prioritized replay buffer
        samples, indices, weights = self.memory.sample(batch_size)

        # Check if sampling returned anything
        if not samples:
             return

        states, actions, rewards, next_states, dones = zip(*samples)

        # Convert to tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Filter out invalid actions (-1) if any were stored.
        # Note: It's better practice to avoid storing invalid actions in the first place
        valid_transition_mask = actions.squeeze(1) != -1
        # Apply mask to all tensors
        states = states[valid_transition_mask]
        actions = actions[valid_transition_mask]
        rewards = rewards[valid_transition_mask]
        next_states = next_states[valid_transition_mask]
        dones = dones[valid_transition_mask]
        weights = weights[valid_transition_mask]
        # Need to also filter the original indices to match the filtered tensors
        filtered_indices = np.array(indices)[valid_transition_mask.cpu().numpy()]


        if len(states) == 0:  # Skip if no valid transitions in the batch after filtering
            # print("Warning: Batch contained only invalid transitions after filtering.") # Debugging
            return

        # Q(s,a)
        q_sa = self.model(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.target_model(next_states) # [B', action_size] (B' <= batch_size)
            valid_mask = self._valid_mask_from_state(next_states) # [B', action_size]
            q_next[~valid_mask] = -1e9 # Mask out invalid actions
            q_next_max = q_next.max(1, keepdim=True)[0] # [B', 1]
            y = rewards + self.gamma * q_next_max * (1 - dones) # [B', 1]

        td_error = torch.abs(q_sa - y).detach().cpu().numpy() # [B', 1] -> [B',]

        self.memory.update_priorities(filtered_indices, td_error) # td_error is already abs and 1e-6 + alpha is applied inside update_priorities

        # Weighted MSE loss
        # Ensure weights match the filtered batch size
        loss = (weights * (q_sa - y.detach()) ** 2).mean() # Use y.detach() here as target is fixed

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()

        self._step_count += 1
        # Update target model periodically
        if self._step_count % self.TARGET_UPDATE_EVERY == 0:
            self.update_target_model()

        # Episodic memory replay happens at the start of replay() now.

    def _replay_from_episodic_memory(self, num_episodes=1):
        # print(f"Replaying from episodic memory. Steps: {self._step_count}") # Debugging
        sampled_episodes = self.episodic_memory.sample(num_episodes)
        added_count = 0
        for episode in sampled_episodes:
            high_priority_base = 2.0
            for transition in episode:
                 # Add transition to main buffer with boosted priority
                 # The add method calculates priority^alpha
                self.memory.add(*transition, priority=high_priority_base)
                added_count += 1
        # if added_count > 0: # Debugging
        #     print(f"Added {added_count} transitions from episodic memory to main buffer.")


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_agent(episodes: int = 2999, name: str = "agent",
                device: str = "cuda", save_every_this_ep: int = 1000,
                save_after_this_ep: int = 900) -> DQNAgent:
    env = MahjongEnvironment()
    state_size = env.observation_space.shape[0] if hasattr(env, 'observation_space') and env.observation_space.shape else 412 # Use env spec if available
    action_size = env.action_space.n if hasattr(env, 'action_space') else 34 + 3 # Use env spec if available
    agent = DQNAgent(state_size, action_size, device=device)
    assert save_every_this_ep >= 99, "Reasonable saving interval must be no less than 99."
    log_save_path = f"./log/train_{name}.csv"

    if os.path.exists(log_save_path):
         print(f"Warning: Log file {log_save_path} already exists. Appending to it.")
         # raise FileExistsError(f"Log file {log_save_path} already exists.")
         # If appending, make sure header is not written again
         write_header = False
    else:
         write_header = True
         # Ensure the log directory exists
         os.makedirs(os.path.dirname(log_save_path), exist_ok=True)


    # Load agent if name is provided and file exists
    model_save_dir = "./DQN_agents_candidates/"
    formal_agent_save_dir = "./DQN_agents/"
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"{name}.pth")
    formal_model_path = os.path.join(formal_agent_save_dir, f"{name}.pth")

    if name and os.path.exists(formal_model_path):
        try:
            agent.model.load_state_dict(torch.load(formal_model_path, map_location=device))
            agent.update_target_model()
            # Optionally adjust epsilon if loading a trained agent
            agent.epsilon = agent.epsilon_min # fine-tuning
            print(f"[INFO] Agent {name} loaded from {formal_model_path}.")
            # If continuing training, might want to load prior episodic memory if saved
            # (Requires implementing save/load for episodic memory)
        except Exception as e:
            print(f"Error loading model {formal_model_path}: {e}")
            print("Starting training from scratch.")
    else:
        print(f"Training new agent: {name}")

    batch_size = 64
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        # Clear episode transitions at the start of each episode
        agent.current_episode_transitions = []

        while True:
            next_state, reward, done, info, action_taken = env.step(agent, env.agent_after_pon)
            agent.memorize(state, action_taken, reward, next_state, done)
            agent.replay(batch_size)
            state = next_state
            ep_reward += reward

            if done:
                # Episode finished
                agent.decay_epsilon()

                # Add episode to episodic memory if successful
                if ep_reward > agent.SUCCESS_REWARD_THRESHOLD:
                # Or: Add to memory as long as it is a win.
                # yaku_list: list = info.get('completeyaku', [])
                # if len(yaku_list) > 0:
                    # Pass the collected transitions and the total reward
                    agent.add_episode_to_memory(agent.current_episode_transitions, ep_reward)
                    # print(f"Episode {ep+1}: Added to episodic memory with reward {ep_reward}") # Debugging

                # Clear the temporary episode storage for the next episode
                agent.current_episode_transitions = []


                # Log results
                envseat = {0: "East", 1: "South", 2: "West", 3: "North"}.get(env.seat, "Unknown")
                try:
                    with open(log_save_path, "a", encoding="utf-8") as f:
                        if write_header: # Write header only once
                            f.write("Episode,Seat,Score,Turn,Epsilon,penalty_A,Yaku,MZ_Score,TEZ_Score,TAZ_Score,Tenpai,HandComplete\n")
                            write_header = False # Don't write again
                        # Ensure log values are safe to write (e.g., convert None to string)
                        log_values = [
                            ep+1,
                            envseat,
                            info.get('score', 'N/A'),
                            info.get('turn', 'N/A'),
                            f"{agent.epsilon:.3f}",
                            info.get('penalty_A', 'N/A'),
                            ' '.join(str(x) for x in info.get('completeyaku', [])), # Handle potential None/empty list
                            info.get('mz_score', 'N/A'),
                            info.get('tuiz_score', 'N/A'),
                            info.get('tatsu_score', 'N/A'),
                            info.get('tenpai', 'N/A'),
                            info.get('hand_when_complete', 'N/A')
                        ]
                        f.write(",".join(map(str, log_values)) + "\n")

                except Exception as e:
                    print(f"Error writing to log file {log_save_path}: {e}")


                if (ep + 1) % save_every_this_ep == 0 and ep > save_after_this_ep:
                    save_path = os.path.join(model_save_dir, f"{name}_{ep+1}.pth")
                    try:
                        torch.save(agent.model.state_dict(), save_path)
                        print(f"[INFO] Agent saved: {save_path}")
                    except Exception as e:
                        print(f"Error saving model to {save_path}: {e}")
                break

    final_save_path = os.path.join(model_save_dir, f"{name}_final.pth")
    try:
        torch.save(agent.model.state_dict(), final_save_path)
        print(f"[INFO] Final agent saved: {final_save_path}")
    except Exception as e:
        print(f"Error saving final model to {final_save_path}: {e}")

    return agent


def test_agent(episodes: int, model_path: str, device: str = "cpu") -> tuple[float, float]:
    env = MahjongEnvironment()
    env.is_test_environment = True
    agent = DQNAgent(412, 34 + 3, device=device)
    agent.epsilon = 0
    log_save_path = f"./log/test_{model_path.split('/')[-1].split('.')[0]}.csv"
    
    # Load pre-trained model
    if os.path.exists(model_path):
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Model loaded.")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found.")
    
    if log_save_path:
        if os.path.exists(log_save_path):
            print(f"Test file {log_save_path} already exist, skip.")
            return 0, 0
        with open(log_save_path, "a", encoding="utf-8") as f:
            f.write("Episode,Seat,Score,Turn,Yaku,MZ_Score,TEZ_Score,TAZ_Score,Tenpai,HandComplete,AgariRate\n")
    
    # Statistics tracking
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
                
                if 'completeyaku' in info and info['completeyaku']:
                    agari += 1
                
                # Calculate current Agari rate
                current_agari_rate = (agari / (ep + 1)) * 100
                
                # Log results
                if log_save_path:
                    with open(log_save_path, "a", encoding="utf-8") as f:
                        f.write(f"{ep+1},{envseat},{info['score']},{info['turn']},"
                               f"{' '.join(str(x) for x in info['completeyaku'])},"
                               f"{info['mz_score']},{info['tuiz_score']},{info['tatsu_score']},"
                               f"{info['tenpai']},{info['hand_when_complete']},{current_agari_rate:.3f}\n")
                
                # Print progress
                if (ep + 1) % 500 == 0:
                    print(f"[TEST] Episode {ep+1}/{episodes} completed. "
                         f"Avg score: {total_score/(ep+1):.2f}, "
                         f"Agari rate: {current_agari_rate:.2f}%")
                break
    
    # Final statistics
    print(f"\n[TEST COMPLETE] Total episodes: {episodes}")
    print(f"Average score: {total_score/episodes:.2f}")
    print(f"Agari rate: {agari/episodes*100:.3f}%")
    return total_score/episodes, agari/episodes*100


def test_all_agent_candidates(episodes: int, device: str = "cpu") -> None:
    agent_dir = "./DQN_agents_candidates/"
    formal_agent_dir = "./DQN_agents/"
    agent_files = [f for f in os.listdir(agent_dir) if f.endswith(".pth")]
    
    # Track best agent performance
    best_agent_file = None
    best_performance = float('-inf')  # Initialize with worst possible score
    results = []
    
    # Test each agent and record results
    for f in agent_files:
        print(f"Testing agent: {f}")
        avg_score, agari_rate = test_agent(episodes, f"{agent_dir}{f}", device)
        
        # Calculate a combined performance metric (you can adjust this formula)
        performance = avg_score * agari_rate
        results.append((f, avg_score, agari_rate, performance))
        
        # Update best agent if current one performs better
        if performance > best_performance:
            best_performance = performance
            best_agent_file = f
    
    # Keep only the best agent, delete the others
    if best_agent_file:
        print(f"\n[BEST AGENT] {best_agent_file}")
        print(f"Average score: {results[results.index((best_agent_file, *[r for r in results if r[0] == best_agent_file][0][1:]))][1]:.2f}")
        print(f"Agari rate: {results[results.index((best_agent_file, *[r for r in results if r[0] == best_agent_file][0][1:]))][2]:.3f}%")
        
        # Delete all agents except the best one
        for f in agent_files:
            file_path = f"{agent_dir}{f}"
            formal_file_path = f"{formal_agent_dir}{f}"
            log_file_path = f"./log/test_{f.split('.')[0]}.csv"
            
            if f != best_agent_file:
                try:
                    # Delete model file
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                    
                    # Delete corresponding log file
                    if os.path.exists(log_file_path):
                        os.remove(log_file_path)
                        print(f"Deleted: {log_file_path}")
                except Exception as e:
                    print(f"Error deleting files: {e}")
            else:
                try:
                    os.rename(file_path, formal_file_path)
                except Exception as e:
                    print(f"Error moving best agent: {e}")

    else:
        print("No agents were evaluated.")


def test_all_agents(episodes: int, device: str = "cpu") -> None:
    agent_dir = "./DQN_agents/"
    agent_files = [f for f in os.listdir(agent_dir) if f.endswith(".pth")]
    for f in agent_files:
        avg_score, agari_rate = test_agent(episodes, f"{agent_dir}{f}", device)


def test_mixed_agent(episodes: int, device: str = "cpu") -> None:
    """
    Tests multiple DQN agents by loading them and selecting an agent for each
    episode based on the weighted preference score of the initial hand,
    calculated using preference data from agent_summary.csv.
    """
    agent_dir = "./DQN_agents/"
    summary_file = "./log/agent_summary.csv"
    log_save_path = f"./test_mixed.csv"

    # 1. Load agent state dictionaries
    agent_state_dicts: dict[str, collections.OrderedDict] = {}
    if not os.path.exists(agent_dir):
        print(f"Error: Agent directory not found: {agent_dir}")
        return

    agent_files = [f for f in os.listdir(agent_dir) if f.endswith(".pth")]
    if not agent_files:
        print(f"Error: No agent files found in {agent_dir}")
        return

    print(f"Loading {len(agent_files)} agent models from {agent_dir}...")
    for filename in agent_files:
        try:
            # Load the state dictionary
            # Ensure map_location is set to the target device
            state_dict: collections.OrderedDict = torch.load(os.path.join(agent_dir, filename), map_location=device)
            agent_state_dicts[filename] = state_dict
            print(f" - Loaded {filename}")
        except Exception as e:
            print(f"Error loading agent file {filename}: {e}")
            # Decide whether to skip this agent or abort
            continue # Skip malformed files

    if not agent_state_dicts:
        print("Error: No agent state dictionaries were successfully loaded.")
        return

    # 2. Read agent preference data from summary file
    agent_preferred_tiles = {}
    agent_avg_score = {}
    if not os.path.exists(summary_file):
        print(f"Error: Agent summary file not found: {summary_file}")
        # Cannot proceed without preference data for selection
        return

    print(f"Reading agent summary from {summary_file}...")
    try:
        with open(summary_file, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader) # Skip header
            try:
                # Find the column indices for 'Filename' and 'Top Tiles'
                filename_col_idx = header.index("Filename")
                top_tiles_col_idx = header.index("Top Tiles")
                avg_score_col_idx = header.index("Average Score")
            except ValueError:
                print(f"Error: Required columns ('Filename', 'Top Tiles') not found in {summary_file}")
                return

            for row in reader:
                # Ensure the row has enough columns
                if len(row) > max(filename_col_idx, top_tiles_col_idx):
                     filename_in_csv = row[filename_col_idx].strip()
                     # test_5th_hiruchaaru_1200.csv -> 5th_hiruchaaru_1200.pth
                     filename_in_csv = filename_in_csv.replace("test_", "").replace(".csv", ".pth")
                     preferred_tiles_str = row[top_tiles_col_idx].strip()
                     agent_avg_score_int = row[avg_score_col_idx].strip()
                     # Only add preference data for agents whose files were successfully loaded
                     if filename_in_csv in agent_state_dicts:
                         agent_preferred_tiles[filename_in_csv] = preferred_tiles_str
                         agent_avg_score[filename_in_csv] = agent_avg_score_int
                         print(f" - Read preferences for {filename_in_csv}") # Uncomment for detailed read
                     else:
                         print(f"Warning: Summary for agent file '{filename_in_csv}' not found in {agent_dir}. Skipping preference data.") # Uncomment for detailed read
                else:
                     print(f"Warning: Skipping malformed row in {summary_file}: {row}") # Uncomment for detailed read


    except Exception as e:
        print(f"Error reading agent summary file {summary_file}: {e}")
        return

    if not agent_preferred_tiles:
         print("Error: No agent preferences successfully loaded from summary file. Cannot perform agent selection.")
         return # Cannot proceed without preferences linking files to scores

    # Initialize the environment and a single agent instance
    # This agent instance will have its state_dict updated before each episode
    env = MahjongEnvironment()
    env.is_test_environment = True

    agent = DQNAgent(412, 34 + 3, device=device)
    agent.epsilon = 0 # Ensure deterministic action selection during testing

    # Prepare log file
    if log_save_path:
        if os.path.exists(log_save_path):
            # Raise error if the log file already exists, as requested
             raise FileExistsError(f"Log file {log_save_path} already exists. Please choose a different name or delete it.")
        with open(log_save_path, "w", encoding="utf-8") as f: # Use "w" to create a new file
            # Add 'Selected Agent' column to the header
            f.write("Episode,Seat,Score,Turn,Yaku,MZ_Score,TEZ_Score,TAZ_Score,Tenpai,HandComplete,AgariRate,Selected Agent\n")

    agari = 0
    total_score = 0

    print(f"\nStarting mixed agent testing for {episodes} episodes...")
    for ep in range(episodes):
        # --- Agent Selection for the current episode ---
        state = env.reset() # Reset environment to get the initial hand
        the_hand_tiles: list[麻雀牌] = env.手牌 # Assuming env.手牌 is updated by reset()

        best_score = -1
        selected_agent_filename = None

        # Calculate preference score for each agent's preferred tiles against the current hand
        for filename, preferred_tiles_string in agent_preferred_tiles.items():
            # Check if the agent's state dict was actually loaded
            if filename in agent_state_dicts:
                score = calculate_weighted_preference_score(the_hand_tiles, preferred_tiles_string, agent_avg_score[filename])
                if score > best_score:
                    best_score = score
                    selected_agent_filename = filename

        # Load the state dict of the selected agent
        if selected_agent_filename:
            agent.model.load_state_dict(agent_state_dicts[selected_agent_filename])
            # print(f"Episode {ep+1}: Selected agent {selected_agent_filename} (Score: {best_score})")
        elif agent_state_dicts:
             print(f"Warning: No agent had a positive preference score for hand in episode {ep+1}. Selecting the first loaded agent.")
             selected_agent_filename = list(agent_state_dicts.keys())[0]
             agent.model.load_state_dict(agent_state_dicts[selected_agent_filename])
        else:
             # This case should not happen if the initial checks passed, but for safety
             print(f"Error: No agents available to run episode {ep+1}. Aborting.")
             break

        # --- Run the episode with the selected agent ---
        ep_reward = 0
        while True:
            # The env.step method should use the current state of the 'agent' object
            next_state, reward, done, info, action = env.step(agent, env.agent_after_pon)
            state = next_state # Update state for the next step
            ep_reward += reward

            if done:
                seat_names = ["East", "South", "West", "North"]
                envseat = seat_names[env.seat]
                total_score += info.get('score', 0) # Use .get for safety
                if info.get('completeyaku', []): # Use .get for safety
                    agari += 1

                # Calculate cumulative agari rate
                current_agari_rate = (agari / (ep + 1)) * 100 if (ep + 1) > 0 else 0

                # Log the results including the selected agent
                if log_save_path:
                    try:
                        with open(log_save_path, "a", encoding="utf-8") as f: # Use "a" to append
                            # Get yaku list string, handle empty list
                            yaku_str = ' '.join(str(x) for x in info.get('completeyaku', []))
                            f.write(f"{ep+1},{envseat},{info.get('score', 0)},{info.get('turn', 0)},"
                                   f"{yaku_str},"
                                   f"{info.get('mz_score', 0)},{info.get('tuiz_score', 0)},{info.get('tatsu_score', 0)},"
                                   f"{info.get('tenpai', False)},{info.get('hand_when_complete', '')},{current_agari_rate:.3f},"
                                   f"{selected_agent_filename}\n") # Log the selected agent
                    except Exception as log_e:
                         print(f"Error writing to log file {log_save_path}: {log_e}")


                # Print progress update every 500 episodes
                if (ep + 1) % 500 == 0:
                    print(f"[TEST] Episode {ep+1}/{episodes} completed. "
                         f"Avg score: {total_score/(ep+1):.2f}, "
                         f"Agari rate: {current_agari_rate:.2f}%. "
                         f"Selected agent for last episode: {selected_agent_filename}")

                break # End of episode inner while loop

    print(f"\n[TEST COMPLETE] Total episodes: {episodes}")
    print(f"Average score: {total_score/episodes:.2f}")
    print(f"Agari rate: {agari/episodes*100:.3f}%")




def train_and_test_pipeline():
    agent_name = "zi0"
    train_agent(999, name=agent_name, device="cuda", save_every_this_ep=100, save_after_this_ep=99)
    test_all_agent_candidates(1000, "cuda")


if __name__ == "__main__":
    # train_and_test_pipeline()
    # test_all_agent_candidates(1000, "cuda")
    # test_all_agents(1000, 'cuda')
    # test_agent(episodes=5000, model_path=f"./DQN_agents/7tui_hr_a_2800.pth", device="cuda")
    test_mixed_agent(3000, "cuda")
