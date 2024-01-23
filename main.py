"""
    This simple script allows you to process a folder of graphs and calculate the following metrics:
    - Average Closeness Centrality
    - Average Betweenness Centrality
    - Global Clustering Coefficient
    - Z-scores for each metric
    - Top 5 proteins for closeness and betweenness

    The scripts uses the following libraries:
    - networkx
    - numpy
"""
import os
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def read_graph_from_file(file_path):
    """
        This simple function will read a graph from a file and return it.

        Args:
            file_path (str): Path to graph file
        
        Returns:
            networkx.Graph: Graph read from file
    """
    print(f"Reading graph from file: {file_path}")
    G = nx.read_edgelist(file_path)
    return G

def calculate_metrics(graph):
    """
        This function will calculate the following metrics:
        - Closeness Centrality
        - Betweenness Centrality
        - Clustering Coefficient

        Args:
            graph (networkx.Graph): Original graph
        
        Returns:
            tuple: Tuple containing the metrics in the order above
    """
    print("Calculating metrics...")
    closeness_centralities = nx.closeness_centrality(graph)
    betweenness_centralities = nx.betweenness_centrality(graph)
    clustering_coefficients = nx.clustering(graph)

    return closeness_centralities, betweenness_centralities, clustering_coefficients

def calculate_z_scores(graph, random_graphs):
    """
        This function will calculate the z-scores for the following metrics:
        - Closeness Centrality
        - Betweenness Centrality
        - Clustering Coefficient

        Args:
            graph (networkx.Graph): Original graph
            random_graphs (dict): Dictionary containing the random graphs and their metrics

        Returns:
            tuple: Tuple containing the z-scores for each metric in the order above
    """
    print("Calculating z-scores...")
    observed_closeness = np.mean(list(nx.closeness_centrality(graph).values()))
    closeness_centrality_z = (observed_closeness - np.mean(random_graphs['closeness'])) / np.std(random_graphs['closeness'])

    observed_betweenness = np.mean(list(nx.betweenness_centrality(graph).values()))
    betweenness_centrality_z = (observed_betweenness - np.mean(random_graphs['betweenness'])) / np.std(random_graphs['betweenness'])

    observed_clustering = nx.average_clustering(graph)
    clustering_coefficient_z = (observed_clustering - np.mean(random_graphs['clustering'])) / np.std(random_graphs['clustering'])

    return closeness_centrality_z, betweenness_centrality_z, clustering_coefficient_z

def generate_random_graphs(graph, num_graphs=10):
    """
        This function will generate a number of random graphs with the same number of nodes and edges as the original graph
        utilyzing the G(n,m) model. Then it will calculate the following metrics for each random graph:
        - Closeness Centrality
        - Betweenness Centrality
        - Clustering Coefficient

        Args:
            graph (networkx.Graph): Original graph
            num_graphs (int): Number of random graphs to generate

        Returns:
            dict: Dictionary containing the random graphs and their metrics
    """
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
    """
        This function will save the results to a text file with the same name as the graph file.
        It is meant to be used after the metrics have been calculated by the analyze_file function.

        Args:
            results (dict): Dictionary containing the results from the analyze_file function
            output_file_path (str): Path to output file
        
        Returns:
            None
    """
    print(f"Saving results to file: {output_file_path}")
    with open(output_file_path, 'w') as f:
        f.write("File name: " + results['File Name'] + "\n")
        f.write("Avg Closeness: " + str(results['Avg Closeness']) + "\n")
        f.write("Avg Betweenness: " + str(results['Avg Betweenness']) + "\n")
        f.write("Global Clustering: " + str(results['Global Clustering']) + "\n")
        f.write("Z-scores - Closeness: " + str(results['Z-scores - Closeness']) + "\n")
        f.write("Z-scores - Betweenness: " + str(results['Z-scores - Betweenness']) + "\n")
        f.write("Z-scores - Clustering: " + str(results['Z-scores - Clustering']) + "\n")
        f.write("Top 5 Closeness: \n")
        for protein in results['Top 5 Closeness']:
            f.write("\t" + str(protein) + "\n")
        f.write("Top 5 Betweenness: \n")
        for protein in results['Top 5 Betweenness']:
            f.write("\t" + str(protein) + "\n")


def analyze_file(file_path, output_file_path):
    """
        This function will process a single file and calculate the following metrics:
        - Average Closeness Centrality
        - Average Betweenness Centrality
        - Global Clustering Coefficient
        - Z-scores for each metric
        - Top 5 proteins for closeness and betweenness

        The results will be saved in a text file with the same name as the graph file.

        Args:
            file_path (str): Path to graph file
            output_file_path (str): Path to output file
        
        Returns:
            None
    """
    graph = read_graph_from_file(file_path)
    closeness_centralities, betweenness_centralities, clustering_coefficients = calculate_metrics(graph)

    # Calculate average closeness and betweenness
    avg_closeness = np.mean(list(closeness_centralities.values()))
    avg_betweenness = np.mean(list(betweenness_centralities.values()))
    global_clustering = np.mean(list(clustering_coefficients.values()))

    # Generate 10 random graphs for each graph
    random_graphs = generate_random_graphs(graph, num_graphs=10)

    closeness_z, betweenness_z, clustering_z = calculate_z_scores(graph, random_graphs)

    # Extracting top 5 proteins per metric
    top_5_closeness = sorted(closeness_centralities.items(), key=lambda item: item[1], reverse=True)[:5]
    top_5_betweenness = sorted(betweenness_centralities.items(), key=lambda item: item[1], reverse=True)[:5]

    result = {
        'File Name': os.path.basename(file_path),
        'Avg Closeness': avg_closeness,
        'Avg Betweenness': avg_betweenness,
        'Global Clustering': global_clustering,
        'Z-scores - Closeness': closeness_z,
        'Z-scores - Betweenness': betweenness_z,
        'Z-scores - Clustering': clustering_z,
        'Top 5 Closeness': top_5_closeness,
        'Top 5 Betweenness': top_5_betweenness,
    }

    save_results_to_file(result, output_file_path)

def main(folder_path, output_folder_path):
    """
        This function will process all the files in the folder and save the results to the output folder.
        The results will be saved in a text file with the same name as the graph file, one file per graph.

        Args:
            folder_path (str): Path to folder containing graphs
            output_folder_path (str): Path to output folder
        
        Returns:
            None
    """
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(analyze_file, os.path.join(folder_path, file_name), os.path.join(output_folder_path,file_name)) for file_name in os.listdir(folder_path)]

        for future in as_completed(futures):
            try:
                future.result()  # This will save the result to the file
            except Exception as e:
                print(f"Error processing file: {e}")

if __name__ == "__main__":
    """
        Execute as follows:
        python3 main.py -f <path to folder containing graphs> -o <path to output folder>
    """
    parser = argparse.ArgumentParser(description="Process a folder of graphs")
    parser.add_argument("-f", "--folder", help="Path to folder containing graphs")
    parser.add_argument("-o", "--output", help="Path to output folder")
    args = parser.parse_args()

    main(args.folder, args.output)
