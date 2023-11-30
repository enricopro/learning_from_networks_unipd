import os
import networkx as nx
import numpy as np
from tabulate import tabulate

def read_graph_from_file(file_path):
    print(f"Reading graph from file: {file_path}")
    G = nx.read_edgelist(file_path)
    return G

def calculate_metrics(graph):
    print("Calculating metrics...")
    avg_closeness_centrality = np.mean(list(nx.closeness_centrality(graph).values()))
    avg_betweenness_centrality = np.mean(list(nx.betweenness_centrality(graph).values()))
    global_clustering_coefficient = nx.average_clustering(graph)
    return avg_closeness_centrality, avg_betweenness_centrality, global_clustering_coefficient

def calculate_z_scores(graph, random_graphs):
    print("Calculating z-scores...")
    observed_closeness = np.mean(list(nx.closeness_centrality(graph).values()))
    closeness_centrality_z = (observed_closeness - np.mean(random_graphs['closeness'])) / np.std(random_graphs['closeness'])

    observed_betweenness = np.mean(list(nx.betweenness_centrality(graph).values()))
    betweenness_centrality_z = (observed_betweenness - np.mean(random_graphs['betweenness'])) / np.std(random_graphs['betweenness'])

    observed_clustering = nx.average_clustering(graph)
    clustering_coefficient_z = (observed_clustering - np.mean(random_graphs['clustering'])) / np.std(random_graphs['clustering'])

    return closeness_centrality_z, betweenness_centrality_z, clustering_coefficient_z

def generate_random_graphs(graph, num_graphs=10):
    print("Generating random graphs...")
    random_graphs = {'closeness': [], 'betweenness': [], 'clustering': []}
    
    for i in range(num_graphs):
        random_graph = nx.erdos_renyi_graph(len(graph.nodes), p=0.1)
        random_graphs['closeness'].append(np.mean(list(nx.closeness_centrality(random_graph).values())))
        random_graphs['betweenness'].append(np.mean(list(nx.betweenness_centrality(random_graph).values())))
        random_graphs['clustering'].append(nx.average_clustering(random_graph))
        print(f"Generated random graph {i + 1}/{num_graphs}")

    return random_graphs

def save_results_to_file(results):
    print("\nResults Table:")
    print(tabulate(results, headers="keys", tablefmt="grid"))  # Change "grid" to "pipe" or "plain" for different styles

    with open("results.txt", "w") as file:
        file.write(tabulate(results, headers="keys", tablefmt="pipe"))

def main():
    folder_path = "."  # Change this to your folder path
    results = []

    for file_name in os.listdir(folder_path):
        if file_name == "results.txt" or not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"\nAnalyzing file: {file_path}")

        graph = read_graph_from_file(file_path)
        avg_closeness, avg_betweenness, global_clustering = calculate_metrics(graph)
        random_graphs = generate_random_graphs(graph)
        closeness_z, betweenness_z, clustering_z = calculate_z_scores(graph, random_graphs)

        results.append({
            'File Name': file_name,
            'Avg Closeness': avg_closeness,
            'Avg Betweenness': avg_betweenness,
            'Global Clustering': global_clustering,
            'Z-scores - Closeness': closeness_z,
            'Z-scores - Betweenness': betweenness_z,
            'Z-scores - Clustering': clustering_z
        })

    save_results_to_file(results)

if __name__ == "__main__":
    main()
