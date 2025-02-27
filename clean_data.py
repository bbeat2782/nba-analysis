import json

# Mapping of NBA Team IDs to Team Names (30 current teams)
nbaTeamIdToName = {
    1610612737: "Atlanta Hawks", 1610612738: "Boston Celtics", 1610612751: "Brooklyn Nets",
    1610612766: "Charlotte Hornets", 1610612741: "Chicago Bulls", 1610612739: "Cleveland Cavaliers",
    1610612742: "Dallas Mavericks", 1610612743: "Denver Nuggets", 1610612765: "Detroit Pistons",
    1610612744: "Golden State Warriors", 1610612745: "Houston Rockets", 1610612754: "Indiana Pacers",
    1610612746: "LA Clippers", 1610612747: "Los Angeles Lakers", 1610612763: "Memphis Grizzlies",
    1610612748: "Miami Heat", 1610612749: "Milwaukee Bucks", 1610612750: "Minnesota Timberwolves",
    1610612740: "New Orleans Pelicans", 1610612752: "New York Knicks", 1610612760: "Oklahoma City Thunder",
    1610612753: "Orlando Magic", 1610612755: "Philadelphia 76ers", 1610612756: "Phoenix Suns",
    1610612757: "Portland Trail Blazers", 1610612758: "Sacramento Kings", 1610612759: "San Antonio Spurs",
    1610612761: "Toronto Raptors", 1610612762: "Utah Jazz", 1610612764: "Washington Wizards"
}

def clean_json_data(input_filename="scripts/graph_data.json", output_filename="scripts/cleaned_graph_data.json"):
    """Filters JSON data to include only players & edges from the 30 NBA teams."""
    
    # Load original JSON data
    with open(input_filename, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Filter nodes: Only keep players whose "group" is a valid integer and belongs to the 30 NBA teams
    filtered_nodes = []
    for node in data["nodes"]:
        try:
            team_id = int(node["group"])  # Convert group to integer
            if team_id in nbaTeamIdToName:
                filtered_nodes.append(node)  # Keep player if the team is in the NBA list
        except ValueError:
            # Ignore players with "No Team" or invalid group values
            continue  

    # Create a set of allowed player IDs (to filter edges)
    valid_player_ids = {node["id"] for node in filtered_nodes}

    # Filter edges: Only keep edges where both players are in the filtered nodes
    filtered_links = [link for link in data["links"] if link["source"] in valid_player_ids and link["target"] in valid_player_ids]

    # Save cleaned data
    cleaned_data = {"nodes": filtered_nodes, "links": filtered_links}
    with open(output_filename, "w", encoding="utf-8") as file:
        json.dump(cleaned_data, file, indent=4)

    print(f"Cleaned JSON saved as {output_filename}")
    print(f"Total Nodes Before: {len(data["nodes"])}, After: {len(filtered_nodes)}")
    print(f"Total Links Before: {len(data["links"])}, After: {len(filtered_links)}")

# Run the cleaning function
clean_json_data()
