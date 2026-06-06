import vlrdevapi as vlr
import pandas as pd

stage_teams = vlr.events.teams(2685)
print(stage_teams)

for team in stage_teams:
    print(f"{team.name}: ID {team.id}")

print(len(stage_teams))

stage_matches = vlr.events.matches(event_id=2685)
for match in stage_matches:
    print(f"{match.teams[0].name} (ID: {match.teams[0].id}) vs "
          f"{match.teams[1].name} (ID: {match.teams[1].id})")

matches_series = {}
for game in stage_matches:
    match_series = vlr.series.matches(game.match_id)
    matches_series[game.match_id] = match_series[1:]

data = {}
for match_id in matches_series:
    for map in matches_series[match_id]:
        teams = map.teams
        for round in map.rounds:
        for round in map.rounds:
            for player in map.players:
                if player.name not in data:
                    data[player.name] = {
                        'round_win': 0,
                        'atk_win': 0,
                        'def_win': 0,
                        'rnd': 0
                    }
                for team in teams:
                    if round.winner_team_id == team.id:
                        if player.team_id == team.id:
                            if round.winner_side == 'Attacker':
                                data[player.name]['atk_win'] += 1
                            else:
                                data[player.name]['def_win'] += 1
                            data[player.name]['round_win'] += 1
                data[player.name]['rnd'] += 1

print(data)

df = pd.DataFrame.from_dict(data, orient='index')

df.to_csv('datasets/player_rnd_wins.csv', index=True)