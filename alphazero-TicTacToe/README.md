# Tic-Tac-Toe by Monte Carlo Tree Search guided by Actor-Critic RL Agent (alphazero)
Play the game [here](https://alphazero-tictactoeapp.streamlit.app/)

Two RL Agents are trained to play
* 3x3 Tic-Tac-Toe
* 6x6 Tic-Tac-Toe with pie rule ( 4 consecutive occupations win)
  
Each AI player is based on Monte Carlo Tree search guided by a Deep RL agent. The agent is CNN based, takes board position as input. It has two heads, one produces Actions (probabilities for all available Actions)for the input board position, another produces value (it represents the winning chance)for the position. The outputs of the RL agent guide the Monte Carlo Tree search.

### Notebooks
* 1.TictacToe - game setup & GUI.ipynb : Game setup and general familiarization.
* 2.MCTS_TicTacToe.ipynb : To play Tic-Tac-Toe 3x3 by classic MCTS
* 3.alphazero-TicTacToe.ipynb : Training of an RL agent to play 3x3 Tic-Tac-Toe
* 4.alphazero-TicTacToe_6by6_advanced.ipynb : Training of an RL agent to play 6x6 Tic-Tac-Toe

### Policy files
* Policy_alphazero_tictactoe.pth : Pytorch policy file for 3x3 Tic-Tac-Toe
* policy_6-6-4-pie-4500_tictactoe.pth : Pytorch policy file for 6x6 Tic-Tac-Toe


