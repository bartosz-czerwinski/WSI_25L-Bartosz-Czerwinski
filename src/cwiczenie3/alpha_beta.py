from two_player_games.games.connect_four import ConnectFour, ConnectFourMove, ConnectFourState
from two_player_games.player import Player
import random


def alphabeta(state, depth, alpha, beta, maximizing_player, player_to_evaluate):

    if depth == 0 or state.is_finished():
        return evaluate(state, player_to_evaluate, depth)

    if maximizing_player:
        value = float('-inf')
        for move in state.get_moves():
            new_state = state.make_move(move)
            value = max(value, alphabeta(new_state, depth - 1, alpha, beta, False, player_to_evaluate))
            if value >= beta:
                break
            alpha = max(alpha, value)
        return value
    else:
        value = float('inf')
        for move in state.get_moves():
            new_state = state.make_move(move)
            value = min(value, alphabeta(new_state, depth - 1, alpha, beta, True, player_to_evaluate))
            if value <= alpha:
                break
            beta = min(beta, value)
        return value

def get_best_move(state, depth, alpha, beta, player):
    best_moves = []
    best_value = float('-inf')

    for move in state.get_moves():
        new_state = state.make_move(move)
        move_value = alphabeta(new_state, depth - 1, alpha, beta, False, player)


        if move_value > best_value:
            best_value = move_value
            best_moves = [move]
        elif move_value == best_value:
            best_moves.append(move)

    # best_move = random.choice(best_moves) if best_moves else None
    best_move = best_moves[0]

    return best_move, best_value


def evaluate(state, player, depth_left):
    if state.is_finished():
        winner = state.get_winner()
        if winner == player:
            return 100000 + depth_left
        elif winner is not None:
            return -100000 + depth_left
        else:
            return 0
    opponent = state._other_player if state._current_player == player else state._current_player
    return (score_position(state, player) - score_position(state, opponent)
    )


def score_position(state, player):
    score = 0
    weights = {2: 1, 3: 10, 4: 1000}
    cols = len(state.fields)
    rows = len(state.fields[0])
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for col in range(cols):
        for row in range(rows):
            for dx, dy in directions:
                tokens = []
                for i in range(4):
                    x = col + i * dx
                    y = row + i * dy
                    if 0 <= x < cols and 0 <= y < rows:
                        tokens.append(state.fields[x][y])
                    else:
                        break
                if len(tokens) == 4:
                    count_player = tokens.count(player)
                    count_empty = tokens.count(None)
                    if count_empty + count_player == 4 and count_player > 1:
                        score += weights.get(count_player, 0)
    return score




def play_against_ai(depth=6, human_start = True):
    print("Rozpoczynamy grę Connect Four!")

    # Stworzenie graczy
    human_player = Player('1')
    ai_player = Player('2')
    if not human_start:
        state = ConnectFourState(size=(7, 6), current_player=ai_player, other_player=human_player)
    else:
        state = ConnectFourState(size=(7, 6), current_player=human_player, other_player=ai_player)

    while not state.is_finished():
        print(state)

        if state.get_current_player() == human_player:
            column = int(input("Wybierz kolumnę (1-7): "))
            move = ConnectFourMove(column-1)

        else:
            print("AI myśli...")
            move, _ = get_best_move(state, depth, float('-inf'), float('inf'), ai_player)
            print(f"AI wybiera kolumnę: {move.column+1}")

        state = state.make_move(move)

    # Gra zakończona
    print(state)
    winner = state.get_winner()
    if winner == human_player:
        print("Wygrałeś!")
    elif winner == ai_player:
        print("AI wygrało!")
    else:
        print("Remis!")


def ai_tournament(num_games=10):
    depth_ai_1 = 5
    depth_ai_2 = 3

    wins_ai1 = 0
    wins_ai2 = 0
    draws = 0

    for game_num in range(1, num_games + 1):
        ai1 = Player('1')
        ai2 = Player('2')

        # Zmienna kto zaczyna – naprzemiennie
        if game_num % 2 == 1:
            state = ConnectFourState(size=(7, 6), current_player=ai1, other_player=ai2)
        else:
            state = ConnectFourState(size=(7, 6), current_player=ai2, other_player=ai1)

        while not state.is_finished():
            current_player = state.get_current_player()
            depth = depth_ai_1 if current_player.char == '1' else depth_ai_2
            move, _ = get_best_move(state, depth, float('-inf'), float('inf'), current_player)
            state = state.make_move(move)

        winner = state.get_winner()
        if winner == ai1:
            wins_ai1 += 1
        elif winner == ai2:
            wins_ai2 += 1
        else:
            draws += 1

        print(f"Gra {game_num}/{num_games}: {winner.char if winner else 'Remis'}")

    print(f"AI1 głębokość {depth_ai_1}: {wins_ai1} zwycięstw")
    print(f"AI2 głębokość {depth_ai_2}: {wins_ai2} zwycięstw")
    print(f"Remisy: {draws}")

# ai_tournament(10)


def compare_ai_strengths(max_depth=5, games_per_pair=10, output_file="wyniki.txt"):
    results = {}
    lines = []

    for depth_ai1 in range(1, max_depth + 1):
        for depth_ai2 in range(depth_ai1, max_depth + 1):
            wins_ai1 = 0
            wins_ai2 = 0
            draws = 0

            for game_num in range(games_per_pair):
                print(f"game {game_num + 1}/{games_per_pair} (AI1: {depth_ai1}, AI2: {depth_ai2})")
                ai1 = Player('1')
                ai2 = Player('2')

                if game_num % 2 == 0:
                    state = ConnectFourState((7, 6), current_player=ai1, other_player=ai2)
                else:
                    state = ConnectFourState((7, 6), current_player=ai2, other_player=ai1)

                while not state.is_finished():
                    current = state.get_current_player()
                    depth = depth_ai1 if current.char == '1' else depth_ai2
                    move, _ = get_best_move(state, depth, float('-inf'), float('inf'), current)
                    state = state.make_move(move)

                winner = state.get_winner()
                if winner == ai1:
                    wins_ai1 += 1
                elif winner == ai2:
                    wins_ai2 += 1
                else:
                    draws += 1

            result_text = (
                f"AI1(depth={depth_ai1}) vs AI2(depth={depth_ai2}) "
                f"=> AI1: {wins_ai1}, AI2: {wins_ai2}, Remisy: {draws}"
            )
            print( result_text)
            lines.append(result_text)
            results[(depth_ai1, depth_ai2)] = {
                'AI1 wins': wins_ai1,
                'AI2 wins': wins_ai2,
                'Draws': draws,
            }

    with open(output_file, 'w') as f:
        for line in lines:
            f.write(line + "\n")
    return results


play_against_ai(depth=5, human_start=True)
#compare_ai_strengths(max_depth=6, games_per_pair=10, output_file="wyniki1.txt")



