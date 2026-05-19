"""
Project EDA/Milestone
Andrew Altamirano, Haiwen Lu
Professor Adrian Salguero
CSE163
5/14/2026
This class contains various tests meant to verify that the calculations done on
our datasets output the correct values.
"""

import project_code
import numpy as np


def player_acs_test():
    player_data = project_code.calculate_player_acs()
    stew_acs = player_data.loc[player_data['Player'] == 'stew', 0]
    zmjjkk_acs = player_data.loc[player_data['Player'] == 'ZmjjKK', 0]
    wsLeo_acs = player_data.loc[player_data['Player'] == 'WsLeo', 0]
    siufat_acs = player_data.loc[player_data['Player'] == 'SiuFatBB', 0]
    assert np.isclose(stew_acs, 212.95)
    assert np.isclose(zmjjkk_acs, 238.91)
    assert np.isclose(wsLeo_acs, 185.55)
    assert np.isclose(siufat_acs, 154.16)


def agent_acs_test() -> None:
    agent_acs = project_code.calculate_agent_acs()
    brimstone = agent_acs.loc[agent_acs['Agent'] == 'brimstone', 0]
    tejo = agent_acs.loc[agent_acs['Agent'] == 'tejo', 0]
    raze = agent_acs.loc[agent_acs['Agent'] == 'raze', 0]
    killjoy = agent_acs.loc[agent_acs['Agent'] == 'killjoy', 0]
    assert np.isclose(brimstone, 212.41)
    assert np.isclose(tejo, 172.23)
    assert np.isclose(raze, 233.22)
    assert np.isclose(killjoy, 197.79)


def comp_popularity_winrate_test() -> None:
    comp_data = project_code.calculate_comp_popularity_winrate()
    dd_games = comp_data.loc[comp_data['Winner'] == 'Double Duelist', 
                             'Total Games']
    dd_wr = comp_data.loc[comp_data['Winner'] == 'Double Duelist', 'Win Rate']
    ds_games = comp_data.loc[comp_data['Winner'] == 'Double Smokes', 
                             'Total Games']
    ds_wr = comp_data.loc[comp_data['Winner'] == 'Double Smokes', 'Win Rate']
    ddi_wr = comp_data.loc[comp_data['Winner'] == 'DD+DI', 'Win Rate']
    assert np.isclose(dd_games, 32)
    assert np.isclose(dd_wr, 0.625)
    assert np.isclose(ds_games, 10)
    assert np.isclose(ds_wr, 0.600)
    assert np.isclose(ddi_wr, 0.000)


def main() -> None:
    player_acs_test()
    agent_acs_test()
    comp_popularity_winrate_test()


if __name__ == '__main__':
    main()
