import vlrdevapi as vlr
import pandas as pd

stage_teams = vlr.events.teams(2685)
print(stage_teams)

for team in stage_teams:
    print(f"{team.name}: ID {team.id}")

print(len(stage_teams))

stage_matches = vlr.events.matches(event_id=2685)
for match in stage_matches:
    print(f"{match.teams[0].name} (ID: {match.teams[0].id}) vs {match.teams[1].name} (ID: {match.teams[1].id})")

matches_series = {}
for game in stage_matches:
    match_series = vlr.series.matches(game.match_id)
    matches_series[game.match_id] = match_series[1:]

data = {}
for match_id in matches_series:
    for map in matches_series[match_id]:
        teams = map.teams
        #print(teams)
        for team in teams:
            for player in map.players:
                if player.team_id == team.id:
                    #print(f"{player.name} is in {team.name}")
                    if player.name in data:
                        data[player.name]['round_win'] += team.score
                        data[player.name]['atk_win'] += team.attacker_rounds
                        data[player.name]['def_win'] += team.defender_rounds
                        data[player.name]['rnd'] += 13
                    else:
                        data[player.name] = {
                            'round_win':team.score,
                            'atk_win':team.attacker_rounds,
                            'def_win':team.defender_rounds,
                            'rnd':13}

print(data)

df = pd.DataFrame.from_dict(data, orient='index')

df.to_csv('datasets/player_rnd_wins.csv', index=True)

players = {}
for match_id in matches_series:
    match_players = matches_series[match_id][0].players
    #print(match_players)
    for player in match_players:
        name = player.name
        if name not in players:
            stats = {
                "team":player.team_id,
                "kills":player.k,
                "deaths":player.d,
                "assists":player.a,
                "kast":player.kast,
                "adr":player.adr,
                "fk":player.fk,
                "fd":player.fd,
                "maps":1
            }
            for key in stats:
                if stats[key] is None:
                    stats[key] = 0
            players[name] = stats
        else:
            print(f"{name}: {players[name]}")
            players[name]['maps'] += 1
            players[name]["kills"] += player.k
            players[name]["deaths"] += player.d
            players[name]["assists"] += player.a
            players[name]["kast"] = (players[name]['kast'] * (players[name]['maps'] - 1) + player.kast)/players[name]['maps']
            players[name]["adr"] = (players[name]['adr'] * (players[name]['maps'] - 1) + player.adr)/players[name]['maps']
            players[name]["fk"] += player.fk
            players[name]["fd"] += player.fd

print(players)

for match_id in matches_series:
    team1_wins = 0
    team2_wins = 0
    for map in matches_series[match_id]:
        team1 = map.teams[0]
        if team1.is_winner:
            team1_wins += 1
        else:
            team2_wins += 1
    if team1_wins > team2_wins:
        print(f"{map.teams[0].name} won")
    else:
        print(f"{map.teams[1].name} won")
