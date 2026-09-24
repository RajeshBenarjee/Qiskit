import json, networkx as nx

with open('data/processed/events/north_andhra_2026/affected_subgraph.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
G = nx.node_link_graph(data)

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print("\nSample Edges:")
for u, v, d in list(G.edges(data=True))[:20]:
    print(f"  {u:25s} -> {v:25s} | Class: {d.get('road_class'):10s} | Length: {d.get('length_m'):8.1f}m | Cost: {d.get('risk_weighted_cost')}")
