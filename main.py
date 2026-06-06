"""
Project EDA/Milestone
Andrew Altamirano, Haiwen Lu
Professor Adrian Salguero
CSE163
5/14/2026
Entry point that runs the full analysis pipeline: summary statistics,
visualizations, and the machine learning models for players and agents.
"""
import pandas as pd
import project_code
import ml_players
import ml_agents


def main():
    """
    Imports data from the CSV files containing Valorant match data, prints
    summary statistics, builds the visualizations, and trains and evaluates
    the player and agent machine learning models.
    """
    data = pd.read_csv("datasets/player_agent.csv")
    data['KAST%'] = data['KAST%'].str.replace('%', '').astype(float)

    teamcomps = pd.read_csv("datasets/Team comp wr.csv")

    adata = pd.read_csv('datasets/agent_picks.csv')
    adata['pick%'] = adata['pick%'].apply(lambda x: float(x[:-1]))
    adata['win%'] = adata['win%'].apply(lambda x: float(x[:-1]))

    rnd_wins = pd.read_csv('datasets/player_rnd_wins.csv')

    print(project_code.player_stats_seven_num_sum(data).round(2))
    print(project_code.agent_pick_seven_num_sum(adata).round(2))
    print(project_code.round_win_seven_num_sum(rnd_wins).round(2))

    player_data = project_code.calculate_player_acs()
    agent_data = project_code.calculate_agent_acs()
    comp_data = project_code.calculate_comp_popularity_winrate()
    project_code.plot_players_acs(player_data)
    project_code.plot_top5_player_acs(player_data)
    project_code.plot_acs_by_agent(data, agent_data)
    project_code.plot_kast_by_agent(data, agent_data)
    project_code.plot_team_comp_heatmap(teamcomps)
    project_code.plot_team_wins_vs_losses(comp_data)
    project_code.plot_agent_pick_vs_win(adata)
    project_code.plot_agent_winrate_by_map(adata)

    # Machine learning challenge
    player_ml = ml_players.merge_player_ml_data()
    player_model, player_scaler = ml_players.player_ml_algorithm(player_ml)
    ml_players.plot_ml_predictions(player_ml, player_model, player_scaler)
    print('Top 5 players predicted by our model:')
    print(ml_players.predict_top_players(
        player_ml, player_model, player_scaler))

    agent_ml = ml_agents.merge_agent_ml_data()
    agent_model, agent_scaler = ml_agents.agent_ml_algorithm(agent_ml)
    ml_agents.plot_ml_predictions(agent_ml, agent_model, agent_scaler)
    print('Top 5 agents predicted by our model:')
    print(ml_agents.predict_top_agents(
        agent_ml, agent_model, agent_scaler))


if __name__ == '__main__':
    main()
