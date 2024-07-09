from IPython.display import clear_output

import random

def display_board(board):
    clear_output()
    print('   ||     ||')
    print(board[1]+'  ||  '+board[2]+'  ||  '+board[3])
    print('===||=====||===')
    print(board[4]+'  ||  '+board[5]+'  ||  '+board[6])
    print('===||=====||===')
    print(board[7]+'  ||  '+board[8]+'  ||  '+board[9])
    print('   ||     ||')


def player_input():
    marker = ''
    my_list = ['X','O']

    while marker not in my_list:
        marker = input('Select your marker (X or O) :').upper()

        if marker not in my_list:
            clear_output()
            print('Wrong choice of marker')

       
        if marker == 'X':
            return ('X','O')
        else:
            return ('O','X')


def place_marker(board,marker,position):
    board[position] = marker
    return board

def win_check(board,marker):
    return ((board[1]==board[2]==board[3]==marker) or
    (board[4]==board[5]==board[6]==marker) or
    (board[7]==board[8]==board[9]==marker) or
    (board[1]==board[4]==board[7]==marker) or
    (board[2]==board[5]==board[8]==marker) or
    (board[3]==board[6]==board[9]==marker) or
    (board[1]==board[5]==board[9]==marker) or
    (board[7]==board[5]==board[3]==marker))


def choose_first():
    turn = random.randint(0,1)
    if turn == 1:
        return 'player 1'
    else:
        return 'player 2'


def space_check(board,position):
    return board[position] == " "

def full_check(board):
    for i in range(1,10):
        if space_check(board,i):
            return False
        
    return True


def player_choice(board):
    position = 0

    while position not in [1,2,3,4,5,6,7,8,9] or not space_check(board,position):
        position = int(input('Choose the position (1-9):'))
    return position


def replay():
    choice = input('Play again? select Yes or No :')
    return choice == 'Yes'



# While loop to keep playing the game
print('Welcome to Tic Tac Toe')

while True:
    #play the game
    # Board, whoose first, choose marker,
    the_board = [' ']*10
    
    player1_marker,player2_marker = player_input()
    
    turn = choose_first()
    print(turn + 'will go first')
    
    play_game = input('Ready to play/ y or n?').lower()
    
    if play_game == 'y':
        game_on = True
    else:
        game_on = False
    
    ## Play Game
    while game_on:
        
        if turn == 'Player 1':
            display_board(the_board)
            #Choose position
            position = player_choice(the_board)
            place_marker(the_board,player1_marker,position)
            

            if win_check(the_board,player1_marker):
                display_board(the_board)
                print('Player 1 has won')
                game_on = False
            
            else:
                if full_check(the_board):
                    display_board(the_board)
                    print('Tie Game')
                    game_on = False
                else:
                    turn = 'Player 2'

        else:
            display_board(the_board)
            #Choose position
            position = player_choice(the_board)
            place_marker(the_board,player2_marker,position)
            

            if win_check(the_board,player2_marker):
                display_board(the_board)
                print('Player 2 has won')
                game_on = False
            
            else:
                if full_check(the_board):
                    display_board(the_board)
                    print('Tie Game')
                    game_on = False
                else:
                    turn = 'Player 1'            

    

    ## Player 1 turn

    ## Player 2 turn

    if not replay():
        break

# Break out from while on the replay()