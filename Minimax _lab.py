# Evaluaci-n-1
Laberinto del gato y el ratón 
import random
import copy
import sys
import Time 

# --- Constantes de Configuración del Juego ---
BOARD_SIZE = 8
MAX_TURNS = 50
SYMBOLS = {'cat': '🐱', 'mouse': '🐭', 'empty': '◻️'}
# Profundidad de búsqueda del algoritmo Minimax. ¡Cuidado! Valores > 5 pueden ser muy lentos.
MINIMAX_DEPTH = 4

class Maze:
    """
    Representa el estado del juego: el tablero, las posiciones del gato y el ratón.
    """
    def __init__(self, size=BOARD_SIZE):
        self.size = size
        self.board = [[SYMBOLS['empty'] for _ in range(size)] for _ in range(size)]
        
        # Posiciones iniciales aleatorias y distintas
        self.cat_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        self.mouse_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        
        while self.cat_pos == self.mouse_pos:
            self.mouse_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        
        # Colocar símbolos en el tablero
        self.board[self.cat_pos[0]][self.cat_pos[1]] = SYMBOLS['cat']
        self.board[self.mouse_pos[0]][self.mouse_pos[1]] = SYMBOLS['mouse']

    def display(self):
        """Muestra el estado actual del tablero en la consola."""
        print("\n  " + " ".join([str(i) for i in range(self.size)]))
        for i, row in enumerate(self.board):
            print(str(i) + " " + " ".join(row))

    def is_valid(self, pos):
        """Verifica si una posición (fila, col) está dentro de los límites del tablero."""
        row, col = pos
        return 0 <= row < self.size and 0 <= col < self.size

    def move_player(self, player_type, new_pos):
        """Mueve un jugador ('cat' o 'mouse') a una nueva posición."""
        if player_type == 'cat':
            current_pos = self.cat_pos
            symbol = SYMBOLS['cat']
            self.cat_pos = new_pos
        else: # mouse
            current_pos = self.mouse_pos
            symbol = SYMBOLS['mouse']
            self.mouse_pos = new_pos
        
        self.board[current_pos[0]][current_pos[1]] = SYMBOLS['empty']
        self.board[new_pos[0]][new_pos[1]] = symbol

    def get_distance(self, pos1, pos2):
        """Calcula la distancia de Manhattan entre dos puntos."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def generate_moves(pos):
    """Genera los 8 movimientos posibles desde una posición dada."""
    moves = []
    row, col = pos
    
    # Movimientos en 8 direcciones (incluyendo diagonales)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),         (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        moves.append((new_row, new_col))
        
    return moves

# --- Núcleo de la IA: Algoritmo Minimax con Poda Alfa-Beta ---

def is_game_over(game_state):
    """El juego termina si el gato y el ratón están en la misma casilla."""
    return game_state.cat_pos == game_state.mouse_pos

def evaluate_state(game_state):
    """
    Función de evaluación (heurística).
    - El ratón (maximizador) quiere que la distancia sea grande.
    - El gato (minimizador) quiere que la distancia sea pequeña.
    """
    if is_game_over(game_state):
        # Si el gato atrapa al ratón, es el peor escenario para el ratón.
        return -1000 
    # La puntuación es la distancia. Más lejos es mejor para el ratón.
    return game_state.get_distance(game_state.mouse_pos, game_state.cat_pos)

def minimax_alpha_beta(game_state, depth, alpha, beta, is_maximizing_player):
    """
    Algoritmo Minimax con poda alfa-beta para decidir el mejor movimiento.
    - is_maximizing_player: True si es el turno del ratón, False si es el del gato.
    """
    if depth == 0 or is_game_over(game_state):
        return evaluate_state(game_state)

    if is_maximizing_player: # Turno del Ratón
        max_eval = -sys.maxsize
        possible_moves = generate_moves(game_state.mouse_pos)
        for move in possible_moves:
            if game_state.is_valid(move):
                # Crea una copia del estado para simular el movimiento
                new_state = copy.deepcopy(game_state)
                new_state.move_player('mouse', move)
                
                evaluation = minimax_alpha_beta(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break # Poda Beta
        return max_eval
    else: # Turno del Gato
        min_eval = sys.maxsize
        possible_moves = generate_moves(game_state.cat_pos)
        for move in possible_moves:
            if game_state.is_valid(move):
                new_state = copy.deepcopy(game_state)
                new_state.move_player('cat', move)

                evaluation = minimax_alpha_beta(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, evaluation)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break # Poda Alfa
        return min_eval

def get_best_move_for_mouse(game_state):
    """Encuentra el mejor movimiento para el ratón (maximizador)."""
    best_move = None
    best_value = -sys.maxsize
    possible_moves = generate_moves(game_state.mouse_pos)
    
    for move in possible_moves:
        if game_state.is_valid(move):
            new_state = copy.deepcopy(game_state)
            new_state.move_player('mouse', move)
            # El siguiente turno es del gato (minimizador)
            move_value = minimax_alpha_beta(new_state, MINIMAX_DEPTH, -sys.maxsize, sys.maxsize, False)
            if move_value > best_value:
                best_value = move_value
                best_move = move
    return best_move

def get_best_move_for_cat(game_state):
    """Encuentra el mejor movimiento para el gato (minimizador)."""
    best_move = None
    best_value = sys.maxsize
    possible_moves = generate_moves(game_state.cat_pos)
    
    for move in possible_moves:
        if game_state.is_valid(move):
            new_state = copy.deepcopy(game_state)
            new_state.move_player('cat', move)
            # El siguiente turno es del ratón (maximizador)
            move_value = minimax_alpha_beta(new_state, MINIMAX_DEPTH, -sys.maxsize, sys.maxsize, True)
            if move_value < best_value:
                best_value = move_value
                best_move = move
    return best_move

# --- Bucle Principal del Juego ---

def main():
    """Función principal que ejecuta la simulación."""
    print("🧠 El Laberinto del Gato y el Ratón: Un Duelo de Inteligencias 🧠")
    
    game = Maze()
    game.display()
    
    print("\n¡Que comience la caza! 🐱🐭\n")
    input("Presiona Enter para iniciar la simulación...")
    
    turn = 0
    while not is_game_over(game) and turn < MAX_TURNS:
        turn += 1
        print(f"\n--- Turno {turn} ---")
        
        # --- Turno del Ratón ---
        print("El ratón está calculando su escape...")
        best_mouse_move = get_best_move_for_mouse(game)
        if best_mouse_move:
            game.move_player('mouse', best_mouse_move)
            print(f"El ratón se mueve a: {best_mouse_move}")
        else:
            print("El ratón no tiene movimientos. ¡El gato gana!")
            break
            
        game.display()

        if is_game_over(game):
            print("\n¡El gato ha atrapado al ratón! Fin del juego.")
            break
        
        time.sleep(1) # Pausa para poder seguir la simulación

        # --- Turno del Gato ---
        print("\nEl gato está planeando su emboscada...")
        best_cat_move = get_best_move_for_cat(game)
        if best_cat_move:
            game.move_player('cat', best_cat_move)
            print(f"El gato se mueve a: {best_cat_move}")
        else:
            print("El gato no tiene movimientos. ¡El ratón escapa!")
            break
            
        game.display()
        
        if is_game_over(game):
            print("\n¡El gato ha atrapado al ratón! Fin del juego.")
            break
            
        time.sleep(1)

    # --- Condiciones de finalización ---
    if turn >= MAX_TURNS and not is_game_over(game):
        print(f"\nEl juego ha alcanzado los {MAX_TURNS} turnos. ¡El ratón logró escapar!")
    
    print("\n--- Simulación Finalizada ---")

if __name__ == "__main__":
    main()
