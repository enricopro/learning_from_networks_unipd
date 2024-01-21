import os
import networkx as nx
import numpy as np
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

RESULTS_FILE = "results.txt"

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
    print(f"Generating {num_graphs} random graphs...")
    random_graphs = {'closeness': [], 'betweenness': [], 'clustering': []}
    
    for i in range(num_graphs):
        random_graph = nx.gnm_random_graph(len(graph.nodes), len(graph.edges))
        random_graphs['closeness'].append(list(nx.closeness_centrality(random_graph).values()))
        random_graphs['betweenness'].append(list(nx.betweenness_centrality(random_graph).values()))
        random_graphs['clustering'].append(nx.average_clustering(random_graph))
        print(f"Generated random graph {i + 1}/{num_graphs}")

    return random_graphs

def save_results_to_file(results, output_file_path):
    print("\nResults Table:")
    print(tabulate(results, headers="keys", tablefmt="grid"))  # Change "grid" to "pipe" or "plain" for different styles

    with open(output_file_path, "a") as file:
        file.write(tabulate(results, headers="keys", tablefmt="pipe"))
        file.write("\n\n")

def analyze_file(file_path, output_file_path):
    graph = read_graph_from_file(file_path)
    avg_closeness, avg_betweenness, global_clustering = calculate_metrics(graph)

    # Generate 10 random graphs for each graph
    random_graphs = generate_random_graphs(graph, num_graphs=10)

    closeness_z, betweenness_z, clustering_z = calculate_z_scores(graph, random_graphs)

    result = {
        'File Name': os.path.basename(file_path),
        'Avg Closeness': avg_closeness,
        'Avg Betweenness': avg_betweenness,
        'Global Clustering': global_clustering,
        'Z-scores - Closeness': closeness_z,
        'Z-scores - Betweenness': betweenness_z,
        'Z-scores - Clustering': clustering_z
    }

    save_results_to_file([result], output_file_path)

def main(folder_path, output_file):

    if os.path.exists(output_file):
        os.remove(output_file)  # Remove existing results file

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_file, os.path.join(folder_path, file_name), output_file) for file_name in os.listdir(folder_path)]

        for future in as_completed(futures):
            try:
                future.result()  # This will save the result to the file
            except Exception as e:
                print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a folder of graphs")
    parser.add_argument("-f", "--folder", help="Path to folder containing graphs")
    parser.add_argument("-o", "--output", help="Path to output file")
    args = parser.parse_args()

    main(args.folder, args.output)
