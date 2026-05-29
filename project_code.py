"""
Project EDA/Milestone
Andrew Altamirano, Haiwen Lu
Professor Adrian Salguero
CSE163
5/14/2026
This class contains methods which process the data in our raw datasets
as well as constructing our data visualizations.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def acs_kast_seven_num_sum(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a seven-number summary for the ACS and KAST% variables from the
    player performance data DataFrame.
    """
    return data[['ACS', 'KAST%']].describe()


def pick_win_seven_num_sum(data: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a seven-number summary for the pickrate and winrate stats from
    the agent pick data DataFrame.
    """
    return data[['pick%', 'win%']].describe()


def calculate_player_acs() -> pd.DataFrame:
    """
    Reads the CSV containing player data by each player, and returns a
    DataFrame comprised of each player and their average combat score
    across all rounds played (ACS).
    """
    data = pd.read_csv("datasets/player_agent.csv")

    data['TotalCS'] = data['Rnd'] * data['ACS']

    by_player = data.groupby("Player")

    player_data = (
        by_player.apply(lambda x: (x['Rnd'] * x['ACS']).sum() / x['Rnd'].sum())
        .round(2)
        .reset_index()
    )

    return player_data


def calculate_agent_acs() -> pd.DataFrame:
    """
    Reads the CSV containing player data by agent played, and returns a
    DataFrame comprised of each agent and the average combat score achieved
    across all rounds played on that agent by any player (ACS).
    """
    data = pd.read_csv("datasets/player_agent.csv")

    data['TotalCS'] = data['Rnd'] * data['ACS']

    by_agent = data.groupby('Agent')

    agent_data = (by_agent.apply(
        lambda x: (x['TotalCS'].sum()/x['Rnd'].sum()))
        .round(2)
        .sort_values(ascending=True)
        .reset_index()
        )

    return agent_data


def calculate_comp_popularity_winrate() -> pd.DataFrame:
    """
    Reads the team composition head-to-head CSV file and returns a new
    DataFrame comprised of each team composition, their number of wins,
    the total number of games they were featured in, and their win-rate.
    """
    teamcomps = pd.read_csv("Team comp wr.csv")
    result = teamcomps[['Winner', 'Total Wins', 'Losses']].copy()
    result['Total Games'] = result['Total Wins'] + result['Losses']
    result['Win Rate'] = (result['Total Wins']/result['Total Games']).round(3)
    result = result.sort_values('Win Rate', ascending=False).reset_index()
    return result


def plot_players_acs(player_data: pd.DataFrame) -> None:
    """
    Creates a barplot displaying every player's ACS based on the player
    performance DataFrame passed in.
    """
    player_data = player_data.rename(columns={0: 'ACS'})
    players = player_data.sort_values('ACS', ascending=False)

    plt.figure(figsize=(12, 30))
    sns.barplot(data=players, x='ACS', y='Player')
    plt.title("Players' Weighted ACS")
    plt.xlabel('ACS (Average Combat Score)')
    plt.ylabel('Player')
    plt.tight_layout()
    plt.savefig('players_acs.png', bbox_inches='tight')


def plot_acs_by_agent(data: pd.DataFrame, agent_data: pd.DataFrame) -> None:
    """
    Creates a multiple box-and-whisker plot relating the ACS achieved by each
    player with the agents played.
    """
    agent_order = agent_data['Agent'].tolist()
    sns.set_theme(style="whitegrid", palette="muted")

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=data, x="Agent", y="ACS", hue="Agent", order=agent_order,
        whis=[0, 100], width=.6
    )
    sns.stripplot(data=data, x="Agent", y="ACS")
    plt.xticks(rotation=75)
    plt.title("ACS (Average Combat Score) vs. Agent Played")
    plt.savefig('acs_by_agent.png', bbox_inches='tight')


def plot_kast_by_agent(data: pd.DataFrame, agent_data: pd.DataFrame) -> None:
    """
    Creates a multiple box-and-whisker plot relating the KAST% achieved by each
    player with the agents played.
    """
    agent_order = agent_data['Agent'].tolist()
    data_byKAST = data.sort_values(by=['KAST%'], ascending=True).reset_index()

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=data_byKAST, x="Agent", y="KAST%", hue="Agent", order=agent_order,
        whis=[0, 100], width=.6
    )
    sns.stripplot(data=data, x="Agent", y="KAST%")
    plt.xticks(rotation=75)
    plt.title("KAST% vs. Agent Played")
    plt.savefig('kast_by_agent.png', bbox_inches='tight')


def plot_team_comp_heatmap(teamcomps: pd.DataFrame) -> None:
    """
    Creates a heatmap of the match record of each team composition archetype
    against each other team composition.
    """
    teamcomps = teamcomps.drop(columns=['Total Wins', 'Losses'])
    teamcomps_numeric = teamcomps.set_index('Winner')

    plt.figure(figsize=(12, 10))
    sns.heatmap(data=teamcomps_numeric, annot=True)
    plt.xlabel("Loser")
    plt.xticks(rotation=45)
    plt.title("Team Composition Wins vs Losses")
    plt.savefig('team_comp_heatmap.png', bbox_inches='tight')


def plot_team_wins_vs_losses(comp_data: pd.DataFrame) -> None:
    """
    Creates a side-by-side barplot of the wins versus losses achieved by each
    team composition.
    """
    comp_wr = comp_data.melt(
        id_vars='Winner',
        value_vars=['Total Wins', 'Losses'],
        var_name='Record',
        value_name='Stats'
    )

    g = sns.catplot(
        data=comp_wr,
        kind="bar",
        x="Winner",
        y="Stats",
        hue="Record",
        height=8,
        aspect=1.5
    )
    g.fig.suptitle("Team Wins vs Losses by Composition")
    g.fig.subplots_adjust(top=0.95)
    plt.savefig('team_wins_vs_losses.png', bbox_inches='tight')


def plot_agent_pick_vs_win(adata: pd.DataFrame) -> None:
    """
    Creates a scatterplot relating the winrates achieved against the agents
    picked by players.
    """
    overall = adata[adata['map'] == 'overall']
    overall = overall[overall['pick%'] >= 5]
    plt.figure(figsize=(14, 8))

    sns.scatterplot(data=overall, x='pick%', y='win%')

    for row in overall.values:
        plt.annotate(
            row[1], (row[2], row[3]),
            xytext=(8, 8), textcoords='offset points', fontsize=9,
        )

    plt.xlabel = ('pick rate')
    plt.ylabel = ('win rate')
    plt.title('Agent Pick% vs Win% (Overall)')
    plt.savefig('agent_pick_vs_win.png', bbox_inches='tight')


def plot_agent_winrate_by_map(adata: pd.DataFrame) -> None:
    """
    Creates a heatmap of winrates of each agent when played on each map.
    Empty cells mean that the agent was not picked on a map.
    """
    adata = adata[adata['pick%'] > 0]
    heatmap_matrix = adata.pivot(index='agent', columns='map', values='win%')

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        linewidths=0.5
    )
    plt.title("Agent Win% by Map")
    plt.savefig('agent_winrate_by_map.png', bbox_inches='tight')


def main():
    """
    Imports data from the CSV files containing Valorant match data and uses
    various functions to make calculations and visualize various aspects
    of the data.
    """
    data = pd.read_csv("datasets/player_agent.csv")
    data['KAST%'] = data['KAST%'].str.replace('%', '').astype(float)

    teamcomps = pd.read_csv("Team comp wr.csv")

    print(acs_kast_seven_num_sum(data).round(2))

    adata = pd.read_csv('datasets/agent_picks.csv')
    adata['pick%'] = adata['pick%'].apply(lambda x: float(x[:-1]))
    adata['win%'] = adata['win%'].apply(lambda x: float(x[:-1]))

    print(pick_win_seven_num_sum(adata).round(2))

    player_data = calculate_player_acs()
    agent_data = calculate_agent_acs()
    comp_data = calculate_comp_popularity_winrate()
    plot_players_acs(player_data)
    plot_acs_by_agent(data, agent_data)
    plot_kast_by_agent(data, agent_data)
    plot_team_comp_heatmap(teamcomps)
    plot_team_wins_vs_losses(comp_data)
    plot_agent_pick_vs_win(adata)
    plot_agent_winrate_by_map(adata)


if __name__ == '__main__':
    main()
