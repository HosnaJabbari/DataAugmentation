import numpy as np
import pandas as pd

def pairwise_mean_augmentation(dataset):
    """
    Augments the dataset by creating synthetic samples based on pairwise means.

    Parameters:
    dataset (pd.DataFrame): Original dataset.

    Returns:
    pd.DataFrame: Augmented dataset including original and synthetic samples.
    """
    augmented_data = dataset.copy()
    num_samples = len(dataset)
    
    # Generate synthetic samples by averaging pairs
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            mean_sample = (dataset.iloc[i] + dataset.iloc[j]) / 2
            augmented_data = augmented_data.append(mean_sample, ignore_index=True)
    
    return augmented_data
def noise_addition_augmentation(dataset, mean=0, stddev=0.01):
    """
    Augments the dataset by adding Gaussian noise to each sample.

    Parameters:
    dataset (pd.DataFrame): Original dataset.
    mean (float): Mean of the Gaussian noise.
    stddev (float): Standard deviation of the Gaussian noise.

    Returns:
    pd.DataFrame: Augmented dataset including original and synthetic samples.
    """
    augmented_data = dataset.copy()
    
    # Apply noise to each sample in the dataset
    for i in range(len(dataset)):
        original_sample = dataset.iloc[i]
        noise = np.random.normal(mean, stddev, original_sample.shape)
        
        # Create synthetic samples by adding and subtracting noise
        sample_plus_noise = original_sample + noise
        sample_minus_noise = original_sample - noise
        
        augmented_data = augmented_data.append([sample_plus_noise, sample_minus_noise], ignore_index=True)
    
    return augmented_data
import networkx as nx
from scipy.stats import kendalltau

def metabolite_correlation_network(dataset, threshold=0.8):
    """
    Constructs a metabolic correlation network based on the Kendall correlation method.

    Parameters:
    dataset (pd.DataFrame): Dataset of metabolites.
    threshold (float): Threshold for correlation significance.

    Returns:
    networkx.Graph: Graph representing the correlation network.
    """
    num_metabolites = dataset.shape[1]
    correlation_matrix = np.zeros((num_metabolites, num_metabolites))
    graph = nx.Graph()
    
    # Calculate Kendall correlation for each metabolite pair
    for i in range(num_metabolites):
        for j in range(i + 1, num_metabolites):
            correlation, _ = kendalltau(dataset.iloc[:, i], dataset.iloc[:, j])
            if correlation >= threshold:
                graph.add_edge(i, j, weight=correlation)
    
    return graph
def strongly_correlated_network(dataset):
    """
    Constructs a strongly correlated network by connecting each metabolite
    to its most strongly correlated partner.

    Parameters:
    dataset (pd.DataFrame): Dataset of metabolites.

    Returns:
    networkx.Graph: Graph representing the strongly correlated network.
    """
    num_metabolites = dataset.shape[1]
    graph = nx.Graph()
    
    for i in range(num_metabolites):
        max_correlation = -1
        max_metabolite = None
        
        for j in range(num_metabolites):
            if i != j:
                correlation, _ = kendalltau(dataset.iloc[:, i], dataset.iloc[:, j])
                if correlation > max_correlation:
                    max_correlation = correlation
                    max_metabolite = j
        
        if max_metabolite is not None:
            graph.add_edge(i, max_metabolite, weight=max_correlation)
    
    return graph
