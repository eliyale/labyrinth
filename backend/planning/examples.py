from tests.unit.factories import create_random_maze, MazeCardFactory
import labyrinth.model.factories as factory
from labyrinth.model.bots import Bot
from labyrinth.model.reachable import Graph
from labyrinth.model.game import Board, BoardLocation, Game, Turns, Player
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

def test_board():
    '''
    Demonstrate construction of a board, and access to its properties, 
    including the maze and the maze cards.
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_random_maze_card())
    print("Board Properties:")
    print(dir(board))

    print("Maze Properties:")
    print(dir(board.maze))

    #list of (x,y) locations of the maze cards
    print(board.maze.maze_locations)

    # Print the first row, (7 cards) of the maze, and their properties
    for location in board.maze.maze_locations[:7]:
        print(f"Testing location: {location}")
        card = board.maze[location]
        print(f"Card at {location}: {card}")
        print(dir(card))
        print(card.CROSS)
        print(f"Card type: {type(card)}")

def test_graph_data_structure():
    '''
    Demonstrate construction of the graph data structure, 
    and access to its properties, including the neighbors 
    and reachable locations from a given location.
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_random_maze_card())
    graph = Graph(board.maze)
    print("Graph Properties:")
    print(dir(graph))

    #Print all neighbors of (0,1), there are two neighbors, (0,2) and (1,1)
    for neighbor in graph._neighbors(BoardLocation(0, 1)):
        print(f"Neighbor of (0,1): {neighbor}")

    #Print all reachable locations from (0,1), there are several connections
    for reachable in graph.reachable_locations(BoardLocation(0, 1)):
        print(f"Reachable from (0,1): {reachable}")

def test_bot():
    '''
    Bots are a subclass of player and link to the shared C++ libraries
    Using the bots is a bit more finnicky than just instantiating a player
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_random_maze_card())
    bot = Bot(board, identifier=1, game=0)
    print("Bot Properties:")
    print(dir(bot))

def test_player():
    '''
    Create a player and show the moves and shift methods, 
    which are the main ways to interact with the game.
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0))
    game = Game(identifier=7, board=board, turns=Turns())
    player = Player(identifier=1, game=0)
    game.add_player(player)
    print("Player Properties:")
    print(dir(player))

    print(dir(player.piece.maze_card))
    print("Player location before move:")
    print(game.board.maze.maze_card_location(player.piece.maze_card))

    game_repository = game_repository_coach.when_game_repository_find_by_id_then_return(game)
    interactor = interactors.PlayerActionInteractor(game_repository=game_repository)
    game_repository.turns = Turns()
    #Insert the straight path card at the top of the first column, which will allow the player to move east
    interactor.perform_shift(game_id=7, player_id=1, shift_location=BoardLocation(0,1), shift_rotation=90)
    # print the maze string representation of the board after this shift, which should show the straight path card at the top of the first column
    print("MAZE STRING")
    print(board.pretty_print())

    interactor.perform_move(game_id=7, player_id=1, move_location=BoardLocation(0,1))

    # Print the location of the player's piece after the move, which should be (0,1) since the player moved east from (0,0) to (0,1)
    print(f"Player piece location after move: {game.board.maze.maze_card_location(player.piece.maze_card)}")

def test_game():
    '''
    Show the properties of the game, which include the board and the turns.
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_random_maze_card())
    game = Game(board)
    print("Game Properties:")
    print(dir(game))

def test_goal():
    '''
    Show the properties of the goal, which include the location and the maze card.

    Note that moving a player will return True if the player has reached the goal, False otherwise.
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    objective = maze[BoardLocation(1, 1)]
    board = Board(maze, leftover_card=maze_card_factory.create_instance(MazeCard.STRAIGHT, 0), objective_maze_card=objective)
    game = Game(identifier=7, board=board, turns=Turns())
    print("Goal Properties:")
    print(dir(game.board.objective_maze_card))

    print("MAZE STRING before actions")
    print(board.pretty_print())

    #should be at (1,1)
    print("Goal Location before shift:")
    print(game.board.maze.maze_card_location(game.board.objective_maze_card))

    player = Player(identifier=1, game=0)
    game.add_player(player)

    print("Player location before move:")
    print(game.board.maze.maze_card_location(player.piece.maze_card))

    #check if a given state is a goal
    loc_obj = game.board.maze.maze_card_location(game.board.objective_maze_card)
    is_goal = loc_obj == BoardLocation(1, 1)
    print("Is goal?", loc_obj, "== (1, 1)?", is_goal)

    game_repository = game_repository_coach.when_game_repository_find_by_id_then_return(game)
    interactor = interactors.PlayerActionInteractor(game_repository=game_repository)
    # Seperate turns object here or should be the same as the one passed to the game??
    game_repository.turns = Turns()
    #Must shift befor moving, so test a shift that doesn't effect the player's location
    interactor.perform_shift(game_id=game.identifier, player_id=1, shift_location=BoardLocation(6,1), shift_rotation=90)
    #Insert the straight path card at the top of the first column, which will allow the player to move east
    interactor.perform_move(game_id=game.identifier, player_id=1, move_location=BoardLocation(0,1))

    print("MAZE STRING after actions")
    print(board.pretty_print())
    
    print("Player location after move:")
    print(game.board.maze.maze_card_location(player.piece.maze_card))

    #The location automatically randomly updates after a move
    loc_obj = game.board.maze.maze_card_location(game.board.objective_maze_card)
    print("Goal location after move:", loc_obj)

def test_leftover_card():
    '''
    Show the properties of the leftover card
    '''
    maze_card_factory = MazeCardFactory()
    maze = create_maze(MAZE_STRING, maze_card_factory)
    board = Board(maze, leftover_card=maze_card_factory.create_random_maze_card())
    game = Game(board)
    print("Game Properties:")
    print(dir(game))
    print("Leftover Card Properties:")
    print(dir(game.board.leftover_card))
    print("Leftover Card Location:")
    print(game.board.leftover_card.location)
    print("Leftover Card Maze Card:")
    print(game.board.leftover_card.maze_card)


test_board()
test_graph_data_structure()
test_bot()
test_player()
test_game()
test_goal()


