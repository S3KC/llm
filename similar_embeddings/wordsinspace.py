import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

words = ["King", "Queen", "Man", "Woman", "Prince", "Princess", "Father", "Mother"]
embeddings = np.array([
    [1.0, 1.2, 0.8],   # King
    [0.9, 1.1, 0.7],   # Queen
    [1.1, -0.8, 0.5],  # Man
    [1.0, -0.7, 0.4],  # Woman
    [1.2, 0.9, 0.6],   # Prince
    [0.8, 0.8, 0.7],   # Princess
    [0.5, -1.2, 0.8],  # Father
    [0.6, -1.0, 0.9]   # Mother
])

# Reduce the dimensionality to 2D using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Plotting the words in 2D space
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], color='blue')

# Annotate each point with the corresponding word
for i, word in enumerate(words):
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
                 textcoords="offset points", xytext=(5, 5), ha='center', fontsize=12, color='red')

# Add title and labels
plt.title("Words in Space - Visualizing Word Embeddings", fontsize=16)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

