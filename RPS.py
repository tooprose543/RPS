choices = ['R', 'P', 'S']

# Given a move, returns the move that beats it
def get_winning_move(opponent_move):
    opponent_last_move_index = choices.index(opponent_move)
    guess_index = (opponent_last_move_index + 1) % 3 
    winning_move = choices[guess_index]
    return winning_move

state_counts = {}
def player(prev_play, opponent_history=[]):
    global state_counts

    # Reset when new game begins
    if prev_play == "":
        state_counts = {}
        opponent_history.clear()
    else:
        opponent_history.append(prev_play)
    
    moves_per_state = 3

    # Initial guess
    guess = "R"
    if len(opponent_history) > moves_per_state:
        pattern = join(opponent_history[-moves_per_state:])

        if join(opponent_history[-(moves_per_state + 1):]) in state_counts.keys():
            state_counts[join(opponent_history[-(moves_per_state + 1):])] += 1
        else:
            state_counts[join(opponent_history[-(moves_per_state + 1):])] = 1

        options = [pattern + x for x in choices]

        for i in options:
            if not i in state_counts.keys():
                state_counts[i] = 0
        
        # Get whatever the opponent plays most often
        most_frequent_next_state = max(options, key=lambda key: state_counts[key])

        # Beat the most common move from the opponent
        guess = get_winning_move(most_frequent_next_state[-1])

    return guess


def join(moves):
    return "".join(moves)
