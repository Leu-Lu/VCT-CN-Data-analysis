"""
Project EDA/Milestone
Andrew Altamirano, Haiwen Lu
Professor Adrian Salguero
CSE163
5/14/2026
This module contains functions for training and evaluating a
machine learning model to predict player win rates from
individual performance statistics.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def merge_player_ml_data() -> pd.DataFrame:
    """
    Reads player_agent.csv and player_winrate.csv, merges them
    on the Player column, and returns a DataFrame with player
    stats and win rate labels ready for machine learning.
    """
    player_data = pd.read_csv('datasets/player_agent.csv')
    player_data['KAST%'] = (
        player_data['KAST%'].str.replace('%', '').astype(float)
    )

    player_data['TotalRnd'] = player_data['Rnd']
    agg = player_data.groupby('Player').apply(
        lambda x: pd.Series({
            'ACS': (x['ACS'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'K/D': (x['K/D'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'KAST%': (x['KAST%'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'ADR': (x['ADR'] * x['Rnd']).sum() / x['Rnd'].sum(),
            'KPR': (x['KPR'] * x['Rnd']).sum() / x['Rnd'].sum(),
        }),
        include_groups=False
    ).reset_index()

    winrate_data = pd.read_csv('datasets/player_winrate.csv')
    merged = agg.merge(winrate_data, left_on='Player', right_on='Player')
    return merged


def player_ml_algorithm(merged_data: pd.DataFrame) -> None:
    features = merged_data[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    labels = merged_data['Win_Rate']

    features_train, features_test, labels_train, labels_test = (
        train_test_split(features, labels, test_size=0.2,
                         random_state=1)
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    model = MLPRegressor(
        hidden_layer_sizes=(20, 10),
        max_iter=10000,
        random_state=0,
        activation='logistic',
        solver='lbfgs',
    )
    model.fit(features_train_scaled, labels_train)
    score = model.score(features_test_scaled, labels_test)
    print('SCORE: ', score)
    return model, scaler

'''
def plot_ml_predictions(data: pd.DataFrame, model, scaler) -> None:
    top30 = data.head(30).copy()
    top30_features = top30[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    top30_scaled = scaler.transform(top30_features)
    top30['Predicted'] = model.predict(top30_scaled)

    plt.figure(figsize=(8, 8))
    plt.scatter(top30['Win_Rate'], top30['Predicted'])

    min_val = top30['Win_Rate'].min()
    max_val = top30['Win_Rate'].max()
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', label='Perfect Prediction')

    plt.xlabel('Actual Win Rate')
    plt.ylabel('Predicted Win Rate')
    plt.title('Predicted vs Actual Win Rate - Top 30 Players')
    plt.legend()
    plt.savefig('ml_predictions.png', bbox_inches='tight')
    plt.clf()
'''


def predict_best_player(data: pd.DataFrame, model, scaler) -> str:
    """
    Predicts the win rate for every player using the trained model and
    returns the name of the player with the highest predicted win rate.
    """
    features = data[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    predicted = model.predict(scaler.transform(features))
    return data.loc[predicted.argmax(), 'Player']


def main():
    merged = merge_player_ml_data()
    model, scaler = player_ml_algorithm(merged)
    #plot_ml_predictions(merged, model, scaler)
    print('The most effective player by our model should be: ', predict_best_player(merged, model, scaler))


if __name__ == '__main__':
    main()