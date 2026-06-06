# VCT-CN-Data-analysis
This project performs data analysis on VCT CN players and agents, covering data processing, visualization, and statistical findings.    
- `project_code.py` contains the core code for data handling and visualization
- `ml_agents.py` contains the code for training a model to predict the win-rates of agents, and plots the accuracy of our model on a graph.
- `ml_players.py` contains the code for training a model to predict the win-rates of players, and plots the accuracy of our model on a graph.
- `main.py` calls for the methods defined in `project_code.py`, `ml_agents.py`, and `ml_players.py`.
- `Testing.py` is used for code testing


To run this program, ensure that the datasets are in the same directory as the python files, in a folder labeled `datasets`.

For example, if this project folder was named `CSE163 Final` and located in your `C:\` drive directory, the code would be located in `C:\CSE163 Final\` and the datasets would be in `C:\CSE163 Final\datasets`.

Additionally, ensure there exists a `graphs` folder within `CSE163 Final` as well, so the Python file may generate data visualizations within this directory.

Then, run main.py to process the datasets and generate all graphs.

You may also run webscrape.py to generate the file `player_rnd_wins.csv` into the `datasets` folder as well.

Dependencies: Python libraries `pandas`, `matplotlib`, `sklearn`, `vlrdevapi`. Testing additionally requires `numpy`.
