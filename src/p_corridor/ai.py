#!/usr/bin/env python3
import json
import sys
from typing import List, Tuple, Dict, Optional
from collections import deque
import heapq

class BoardGameAI:
    def __init__(self):
        self.BOARD_SIZE = 17
        self.CENTER = 8  # 0,0 in game coordinates = index 8
        self.MAX_WALLS = 10
        self.WALL_INTERVAL = 10
        
    def game_to_board(self, game_x: int, game_y: int) -> Tuple[int, int]:
        """ゲーム座標を配列インデックスに変換"""
        return (game_x + self.CENTER, game_y + self.CENTER)
    
    def board_to_game(self, board_x: int, board_y: int) -> Tuple[int, int]:
        """配列インデックスをゲーム座標に変換"""
        return (board_x - self.CENTER, board_y - self.CENTER)
    
    def is_valid_position(self, x: int, y: int, player_pos: Tuple[int, int], 
                         opponent_pos: Tuple[int, int]) -> bool:
        """位置が有効かチェック（盤内かつ他のプレイヤーがいない）"""
        return (0 <= x < self.BOARD_SIZE and 0 <= y < self.BOARD_SIZE and 
                (x, y) != player_pos and (x, y) != opponent_pos)
    
    def is_wall_blocking(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], 
                        walls: List[Dict]) -> bool:
        """2つの隣接するセル間に壁があるかチェック"""
        from_x, from_y = from_pos
        to_x, to_y = to_pos
        
        for wall in walls:
            if wall['orientation'] == 'horizontal':
                # 水平な壁は垂直移動をブロック
                if from_x == to_x and abs(from_y - to_y) == 1:
                    wall_y = max(from_y, to_y) - 0.5
                    if (wall['x'] == from_x and 
                        (wall['y'] == wall_y or wall['y'] + 1 == wall_y)):
                        return True
            else:  # vertical
                # 垂直な壁は水平移動をブロック
                if from_y == to_y and abs(from_x - to_x) == 1:
                    wall_x = max(from_x, to_x) - 0.5
                    if (wall['y'] == from_y and 
                        (wall['x'] == wall_x or wall['x'] + 1 == wall_x)):
                        return True
        return False
    
    def get_valid_moves(self, pos: Tuple[int, int], opponent_pos: Tuple[int, int], 
                       walls: List[Dict]) -> List[Tuple[int, int]]:
        """有効な移動先を取得"""
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        x, y = pos
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)
            
            if (self.is_valid_position(new_x, new_y, pos, opponent_pos) and
                not self.is_wall_blocking(pos, new_pos, walls)):
                moves.append(new_pos)
        
        return moves
    
    def find_shortest_path(self, start: Tuple[int, int], goal_side: str, 
                          opponent_pos: Tuple[int, int], walls: List[Dict]) -> Optional[List[Tuple[int, int]]]:
        """A*アルゴリズムを使って最短経路を探索"""
        if goal_side == 'left':
            goal_x = 0
        else:  # right
            goal_x = self.BOARD_SIZE - 1
            
        # A*アルゴリズム
        heap = [(0, 0, start, [start])]  # (f_score, g_score, position, path)
        visited = set()
        
        while heap:
            f_score, g_score, current_pos, path = heapq.heappop(heap)
            
            if current_pos in visited:
                continue
            visited.add(current_pos)
            
            # ゴール到達チェック
            if current_pos[0] == goal_x:
                return path
            
            # 隣接する有効な移動先を探索
            for next_pos in self.get_valid_moves(current_pos, opponent_pos, walls):
                if next_pos not in visited:
                    new_g_score = g_score + 1
                    # ヒューリスティック: ゴールまでのマンハッタン距離
                    h_score = abs(next_pos[0] - goal_x)
                    new_f_score = new_g_score + h_score
                    
                    heapq.heappush(heap, (new_f_score, new_g_score, next_pos, path + [next_pos]))
        
        return None
    
    def can_place_wall(self, wall_pos: Tuple[float, float], orientation: str, 
                      walls: List[Dict]) -> bool:
        """壁を設置できるかチェック"""
        wall_x, wall_y = wall_pos
        
        # 既存の壁と重複していないかチェック
        for existing_wall in walls:
            if existing_wall['orientation'] == orientation:
                if orientation == 'horizontal':
                    if (existing_wall['y'] == wall_y and 
                        abs(existing_wall['x'] - wall_x) < 2):
                        return False
                else:  # vertical
                    if (existing_wall['x'] == wall_x and 
                        abs(existing_wall['y'] - wall_y) < 2):
                        return False
        
        return True
    
    def evaluate_wall_effectiveness(self, wall: Dict, my_pos: Tuple[int, int], 
                                   opponent_pos: Tuple[int, int], walls: List[Dict], 
                                   my_goal: str, opponent_goal: str) -> float:
        """壁の効果を評価（高いほど良い）"""
        test_walls = walls + [wall]
        
        # 壁設置前の経路長を計算
        my_path_before = self.find_shortest_path(my_pos, my_goal, opponent_pos, walls)
        opponent_path_before = self.find_shortest_path(opponent_pos, opponent_goal, my_pos, walls)
        
        # 壁設置後の経路長を計算
        my_path_after = self.find_shortest_path(my_pos, my_goal, opponent_pos, test_walls)
        opponent_path_after = self.find_shortest_path(opponent_pos, opponent_goal, my_pos, test_walls)
        
        # 経路が見つからない場合は無効
        if not my_path_after or not opponent_path_after:
            return -1000.0
        
        # 経路長の変化を計算
        my_length_before = len(my_path_before) if my_path_before else 1000
        opponent_length_before = len(opponent_path_before) if opponent_path_before else 1000
        
        my_length_after = len(my_path_after)
        opponent_length_after = len(opponent_path_after)
        
        # 相手の経路が長くなるほど良い、自分の経路が長くなるほど悪い
        opponent_penalty = opponent_length_after - opponent_length_before
        my_penalty = my_length_after - my_length_before
        
        # 効果スコア：相手への阻害効果 - 自分への悪影響
        score = opponent_penalty - (my_penalty * 0.8)  # 自分への影響を少し軽く評価
        
        return score

    def find_best_wall_placement(self, my_pos: Tuple[int, int], opponent_pos: Tuple[int, int], 
                                walls: List[Dict], my_goal: str, opponent_goal: str) -> Optional[Dict]:
        """最適な壁設置位置を見つける"""
        # 現在の経路を取得
        my_current_path = self.find_shortest_path(my_pos, my_goal, opponent_pos, walls)
        opponent_current_path = self.find_shortest_path(opponent_pos, opponent_goal, my_pos, walls)
        
        if not opponent_current_path or len(opponent_current_path) < 2:
            return None
        
        best_wall = None
        best_score = -float('inf')
        
        # 相手の経路上の複数の位置で壁設置を検討
        for i in range(min(3, len(opponent_current_path) - 1)):  # 先頭3つの移動を検討
            current_pos = opponent_current_path[i]
            next_pos = opponent_current_path[i + 1]
            
            # 各方向の壁を試す
            wall_candidates = []
            
            if current_pos[0] == next_pos[0]:  # 垂直移動
                # 水平な壁で阻害
                wall_x = current_pos[0]
                wall_y = min(current_pos[1], next_pos[1]) - 0.5
                if 0 <= wall_x < self.BOARD_SIZE and 0 <= wall_y < self.BOARD_SIZE - 1:
                    wall_candidates.append({
                        'x': wall_x, 'y': wall_y, 'orientation': 'horizontal'
                    })
                    # 隣接位置にも設置を検討
                    if wall_x > 0:
                        wall_candidates.append({
                            'x': wall_x - 1, 'y': wall_y, 'orientation': 'horizontal'
                        })
                    if wall_x < self.BOARD_SIZE - 1:
                        wall_candidates.append({
                            'x': wall_x + 1, 'y': wall_y, 'orientation': 'horizontal'
                        })
                        
            else:  # 水平移動
                # 垂直な壁で阻害
                wall_x = min(current_pos[0], next_pos[0]) - 0.5
                wall_y = current_pos[1]
                if 0 <= wall_x < self.BOARD_SIZE - 1 and 0 <= wall_y < self.BOARD_SIZE:
                    wall_candidates.append({
                        'x': wall_x, 'y': wall_y, 'orientation': 'vertical'
                    })
                    # 隣接位置にも設置を検討
                    if wall_y > 0:
                        wall_candidates.append({
                            'x': wall_x, 'y': wall_y - 1, 'orientation': 'vertical'
                        })
                    if wall_y < self.BOARD_SIZE - 1:
                        wall_candidates.append({
                            'x': wall_x, 'y': wall_y + 1, 'orientation': 'vertical'
                        })
            
            # 各壁候補を評価
            for wall in wall_candidates:
                if (0 <= wall['x'] < self.BOARD_SIZE - 1 and 
                    0 <= wall['y'] < self.BOARD_SIZE - 1 and
                    self.can_place_wall((wall['x'], wall['y']), wall['orientation'], walls)):
                    
                    score = self.evaluate_wall_effectiveness(
                        wall, my_pos, opponent_pos, walls, my_goal, opponent_goal
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_wall = wall
        
        # 追加的な戦略的位置も検討（相手のゴール近くなど）
        strategic_positions = []
        if opponent_goal == 'left':
            # 左側ゴール付近に垂直な壁
            for y in range(max(0, opponent_pos[1] - 2), min(self.BOARD_SIZE, opponent_pos[1] + 3)):
                for x in range(3):  # 左端付近
                    strategic_positions.append({
                        'x': x - 0.5, 'y': y, 'orientation': 'vertical'
                    })
        else:  # right
            # 右側ゴール付近に垂直な壁
            for y in range(max(0, opponent_pos[1] - 2), min(self.BOARD_SIZE, opponent_pos[1] + 3)):
                for x in range(self.BOARD_SIZE - 3, self.BOARD_SIZE):  # 右端付近
                    strategic_positions.append({
                        'x': x - 0.5, 'y': y, 'orientation': 'vertical'
                    })
        
        # 戦略的位置も評価
        for wall in strategic_positions:
            if (0 <= wall['x'] < self.BOARD_SIZE - 1 and 
                0 <= wall['y'] < self.BOARD_SIZE - 1 and
                self.can_place_wall((wall['x'], wall['y']), wall['orientation'], walls)):
                
                score = self.evaluate_wall_effectiveness(
                    wall, my_pos, opponent_pos, walls, my_goal, opponent_goal
                )
                
                if score > best_score:
                    best_score = score
                    best_wall = wall
        
        # 有効なスコアの壁が見つかった場合のみ返す
        if best_wall and best_score > 0:
            return best_wall
        
        return None
    
    def make_decision(self, game_state: Dict) -> Dict:
        """ゲーム状態を受け取って次の行動を決定"""
        # ゲーム状態の解析
        my_player = game_state['current_player']
        turn = game_state['turn']
        
        if my_player == 'player1':
            my_pos = tuple(game_state['player1_pos'])
            opponent_pos = tuple(game_state['player2_pos'])
            my_walls = game_state['player1_walls']
            my_goal = 'right'
            opponent_goal = 'left'
        else:
            my_pos = tuple(game_state['player2_pos'])
            opponent_pos = tuple(game_state['player1_pos'])
            my_walls = game_state['player2_walls']
            my_goal = 'left'
            opponent_goal = 'right'
        
        walls = game_state['walls']
        
        # 壁設置のターンかどうかチェック
        if turn % self.WALL_INTERVAL == 0 and my_walls > 0:
            # 壁設置を試みる
            wall_placement = self.find_best_wall_placement(my_pos, opponent_pos, walls, 
                                                         my_goal, opponent_goal)
            if wall_placement:
                # ゲーム座標に変換
                game_x, game_y = self.board_to_game(
                    int(wall_placement['x'] + 0.5), 
                    int(wall_placement['y'] + 0.5)
                )
                return {
                    'action': 'place_wall',
                    'x': game_x,
                    'y': game_y,
                    'orientation': wall_placement['orientation']
                }
        
        # 移動する
        path = self.find_shortest_path(my_pos, my_goal, opponent_pos, walls)
        if path and len(path) > 1:
            next_pos = path[1]
            game_x, game_y = self.board_to_game(next_pos[0], next_pos[1])
            return {
                'action': 'move',
                'x': game_x,
                'y': game_y
            }
        
        # 移動できない場合（エラー回避）
        valid_moves = self.get_valid_moves(my_pos, opponent_pos, walls)
        if valid_moves:
            next_pos = valid_moves[0]
            game_x, game_y = self.board_to_game(next_pos[0], next_pos[1])
            return {
                'action': 'move',
                'x': game_x,
                'y': game_y
            }
        
        return {'action': 'move', 'x': 0, 'y': 0}  # フォールバック

def main():
    """標準入力からゲーム状態を読み取り、次の行動をJSON出力"""
    try:
        # 標準入力からJSON読み取り
        input_data = sys.stdin.read().strip()
        game_state = json.loads(input_data)
        
        # AIインスタンス作成
        ai = BoardGameAI()
        
        # 盤座標をボード座標に変換
        game_state['player1_pos'] = ai.game_to_board(
            game_state['player1_pos'][0], 
            game_state['player1_pos'][1]
        )
        game_state['player2_pos'] = ai.game_to_board(
            game_state['player2_pos'][0], 
            game_state['player2_pos'][1]
        )
        
        # 壁の座標も変換
        for wall in game_state['walls']:
            wall_board_pos = ai.game_to_board(
                int(wall['x'] + 0.5), 
                int(wall['y'] + 0.5)
            )
            wall['x'] = wall_board_pos[0] - 0.5
            wall['y'] = wall_board_pos[1] - 0.5
        
        # 次の行動を決定
        decision = ai.make_decision(game_state)
        
        # JSON出力
        print(json.dumps(decision, ensure_ascii=False))
        
    except Exception as e:
        # エラー時のフォールバック
        error_response = {
            'action': 'move',
            'x': 0,
            'y': 0,
            'error': str(e)
        }
        print(json.dumps(error_response, ensure_ascii=False))

if __name__ == "__main__":
    main()