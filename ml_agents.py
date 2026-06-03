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
    Reads player_agent.csv, averages each agent's stats across all
    players (weighted by rounds), and merges in each agent's overall
    win rate from agent_picks.csv, returning a DataFrame ready for
    machine learning.
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
    """
    Trains a neural network to predict an agent's win rate from its
    average performance stats. Prints the test R^2 score and returns
    the fitted model and the scaler used on the features.
    """
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


def plot_ml_predictions(data: pd.DataFrame, model, scaler) -> None:
    """
    Recreates the same train/test split used to fit the model and
    plots predicted vs actual win rates, coloring training and test
    points differently so over-fitting (train points hugging the
    line, test points scattered) is visible.
    """
    features = data[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    labels = data['win%']
    feat_train, feat_test, lab_train, lab_test = (
        train_test_split(features, labels, test_size=0.2, random_state=1)
    )

    pred_train = model.predict(scaler.transform(feat_train))
    pred_test = model.predict(scaler.transform(feat_test))

    low = labels.min()
    high = labels.max()

    plt.figure(figsize=(8, 8))
    plt.scatter(lab_train, pred_train, color='tab:blue', label='Train')
    plt.scatter(lab_test, pred_test, color='tab:orange', label='Test')
    plt.plot([low, high], [low, high], 'k--', label='Perfect Prediction')
    plt.xlabel('Actual Win Rate')
    plt.ylabel('Predicted Win Rate')
    plt.title('Predicted vs Actual Win Rate - Agents')
    plt.legend()
    plt.savefig('graphs/ml_agent_predictions.png', bbox_inches='tight')
    plt.clf()


def predict_top_agents(data: pd.DataFrame, model, scaler) -> list:
    """
    Predicts the win rate for every agent using the trained model and
    returns the names of the five agents with the highest predicted
    win rates.
    """
    result = data.copy()
    features = result[['ACS', 'K/D', 'KAST%', 'ADR', 'KPR']]
    result['Predicted'] = model.predict(scaler.transform(features))
    result = result.sort_values('Predicted', ascending=False)
    return list(result['agent'].head(5))


def main():
    """
    Builds the agent dataset, trains the model, and prints the five
    agents with the highest predicted win rates.
    """
    merged = merge_agent_ml_data()
    model, scaler = agent_ml_algorithm(merged)
    plot_ml_predictions(merged, model, scaler)
    print('Top 5 agents predicted by our model:')
    print(predict_top_agents(merged, model, scaler))


if __name__ == '__main__':
    main()
