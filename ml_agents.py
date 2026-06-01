"""
Project EDA/Milestone
Andrew Altamirano, Haiwen Lu
Professor Adrian Salguero
CSE163
5/14/2026
This module contains functions for training and evaluating a
machine learning model to predict agents win rates from
individual performance statistics.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def merge_agent_ml_data() -> pd.DataFrame:
    """
    Reads player_agent.csv and player_winrate.csv, merges them
    on the Player column, and returns a DataFrame with player
    stats and win rate labels ready for machine learning.
    """
    agent_data = pd.read_csv('datasets/player_agent.csv')
    agent_data['KAST%'] = (
        agent_data['KAST%'].str.replace('%', '').astype(float)
    )

    agent_data['TotalRnd'] = agent_data['Rnd']
    agg = agent_data.groupby('Agent').apply(
        lambda x: pd.Series({
            'ACS': (x['ACS'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'K/D': (x['K/D'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'KAST%': (x['KAST%'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'ADR': (x['ADR'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'KPR': (x['KPR'] * x['Rnd']).sum() / x['Rnd'].sum(),
        }),
        include_groups=False
    ).reset_index()

    winrate_data = pd.read_csv('datasets/agent_picks.csv')
    winrate_data = winrate_data[winrate_data['map'] == 'overall']
    merged = agg.merge(winrate_data, left_on='Agent', right_on='agent')
    return merged

def agent_ml_algorithm(merged_data: pd.DataFrame) -> None:
    features = merged_data[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    merged_data['win%'] = (
        merged_data['win%'].str.replace('%', '').astype(float)
    )
    labels = merged_data['win%']

    features_train, features_test, labels_train, labels_test = (
        train_test_split(features, labels, test_size=0.2,
                         random_state=1)
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    model = MLPRegressor(
        hidden_layer_sizes=(20, 10),
        max_iter=50000,
        random_state=0,
        activation='logistic',
        solver='lbfgs',
    )
    model.fit(features_train_scaled, labels_train)
    score = model.score(features_test_scaled, labels_test)
    print('SCORE: ', score)
    return model, scaler


def predict_best_agent(data: pd.DataFrame, model, scaler) -> str:
    features = data[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    predicted = model.predict(scaler.transform(features))
    return data.loc[predicted.argmax(), 'agent']


def main():
    merged = merge_agent_ml_data()
    model, scaler = agent_ml_algorithm(merged)
    #plot_ml_predictions(merged, model, scaler)
    print('The most effective agent predicted by our model should be: ', predict_best_agent(merged, model, scaler))


if __name__ == '__main__':
    main()