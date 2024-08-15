import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def visualize_attention(tokens, attention_scores):
    """
    Visualize the attention scores for each token.

    Args:
    tokens (list of str): List of tokens.
    attention_scores (numpy array): Attention scores for each token.
    """
    # Normalize attention scores
    attention_scores = attention_scores / attention_scores.sum(axis=1, keepdims=True)

    # Create a heatmap
    plt.figure(figsize=(15, 5))
    plt.imshow(attention_scores, cmap='viridis', aspect='auto')

    # Set the labels
    plt.xticks(ticks=np.arange(len(tokens)), labels=tokens, rotation=90)
    plt.yticks(ticks=np.arange(attention_scores.shape[0]),
               labels=[f'Head {i + 1}' for i in range(attention_scores.shape[0])])

    plt.colorbar(label='Attention Score')
    plt.title('Attention Scores for Each Token')
    plt.xlabel('Tokens')
    plt.ylabel('Attention Heads')
    plt.show()