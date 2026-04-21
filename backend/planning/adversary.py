from tests.unit.factories import create_random_maze, MazeCardFactory
import labyrinth.model.factories as factory
from labyrinth.model.bots import Bot
from labyrinth.model.reachable import Graph
from labyrinth.model.game import Board, BoardLocation, Game, Turns, Player, Adversary
from labyrinth.model.factories import create_maze
from labyrinth.model import interactors
import tests.unit.game_repository_mocks as game_repository_coach
from labyrinth.model.game import MazeCard


# An example string that can be used to create a maze
MAZE_STRING = """
###|#.#|#.#|###|#.#|#.#|###|
#..|#..|...|...|#..|..#|..#|
#.#|###|###|#.#|###|###|#.#|
---------------------------|
###|###|#.#|#.#|#.#|#.#|#.#|
...|...|#.#|#..|#.#|...|..#|
#.#|#.#|#.#|###|#.#|###|#.#|
---------------------------|
#.#|#.#|#.#|#.#|#.#|#.#|#.#|
#..|#..|..#|#..|..#|#.#|..#|
#.#|#.#|#.#|#.#|#.#|#.#|#.#|
---------------------------|
#.#|#.#|#.#|###|#.#|###|###|
..#|..#|#..|...|...|...|..#|
###|#.#|###|#.#|###|#.#|#.#|
---------------------------|
###|#.#|###|#.#|###|#.#|###|
#..|..#|#..|#.#|...|#..|...|
#.#|###|#.#|#.#|#.#|###|###|
---------------------------|
###|#.#|###|#.#|#.#|#.#|#.#|
..#|#..|...|...|#.#|#..|..#|
#.#|#.#|###|###|#.#|#.#|#.#|
---------------------------|
#.#|#.#|###|###|#.#|#.#|#.#|
#..|...|...|...|#.#|...|..#|
###|###|#.#|###|#.#|###|###|
---------------------------*

"""

def test_adversary_turns():
    '''
    Adversaries are a type of automatic opponent that are not always acting.
    Rather than immediately being given a turn, once "N" turns pass, they will
    shift a part of the board and then let the game continue.
    '''
    N = 5
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0))
    game = Game(identifier=7, board=board, turns=Turns())
    player = Player(identifier=1, game=0)
    game.add_player(player)
    turn_length = len(game.turns._turn_states)

    adversary = Adversary(identifier=2, game=0, attack_in_turns=N)
    game.add_player(adversary)

    print(len(game.turns._turn_states))
    assert len(game.turns._turn_states) is turn_length * N + 2


def test_adversary_move():
    '''
    Adversaries are a type of automatic opponent that are not always acting.
    Rather than immediately being given a turn, once "N" turns pass, they will
    shift a part of the board and then let the game continue.
    '''
    N = 2
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0))
    game = Game(identifier=7, board=board, turns=Turns())
    player = Player(identifier=1, game=0)
    game.add_player(player)
    turn_length = len(game.turns._turn_states)

    adversary = Adversary(identifier=2, game=0, attack_in_turns=N)
    game.add_player(adversary)

    print("Player location before move:")
    print(player.piece.maze_card)

    game_repository = game_repository_coach.when_game_repository_find_by_id_then_return(game)
    interactor = interactors.PlayerActionInteractor(game_repository=game_repository)
    game_repository.turns = Turns()

    interactor.perform_shift(game_id=7, player_id=1, shift_location=BoardLocation(0,1), shift_rotation=90)
    interactor.perform_move(game_id=7, player_id=1, move_location=BoardLocation(0,1))

    #Insert the straight path card at the top of the first column, which will allow the player to move east
    interactor.perform_shift(game_id=7, player_id=1, shift_location=BoardLocation(0,5), shift_rotation=90)
    # print the maze string representation of the board after this shift, which should show the straight path card at the top of the first column
    print("MAZE STRING")
    print(board.maze.pretty_print())

    interactor.perform_move(game_id=7, player_id=1, move_location=BoardLocation(1,3))

    interactor.perform_shift(game_id=7, player_id=2, shift_location=BoardLocation(1,0), shift_rotation=0)

    print("MAZE STRING")
    print(board.maze.pretty_print())

    print("Player location after move:")
    print(player.piece.maze_card)


test_adversary_turns()
test_adversary_move()