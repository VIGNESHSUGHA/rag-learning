from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import  matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter

import re
'''
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
'''

# Step 1 load document
with open("documents.txt", "r", encoding= 'utf-8') as f:
    text = f.read()
     # re.sub(pattern, replacement, text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)


# Step 2 Chunk the document and store as output.txt file
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                               chunk_overlap=20)

chunkings= text_splitter.split_text(text)


with open("output.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunkings):
        f.write(f"\n{'='*50}\n")
        f.write(f"Chunk {i+1}\n")
        f.write(chunk)
        f.write("\n")


#Step 3: Generate Embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                       device = "cuda")

chunk_embeddings= model.encode(chunkings)

df = pd.DataFrame(chunk_embeddings)
df.to_csv("embeddings.csv")

print(chunk_embeddings.shape) # (7, 384)

# Step 4: Ask Question
query = "How is loan eligibility calculated?"

query_embedding = model.encode([query])

query_df = pd.DataFrame(query_embedding)
query_df.to_csv("query_embeddings.csv")

# Step 5: Similarity Search
scores = cosine_similarity(
    query_embedding,
    chunk_embeddings
)

print(scores)  # [[0.15369216 0.02966083 0.2727212  0.77440864 0.40040958 0.53378963 0.04922779]]

# Find Best Chunk

best_index = np.argmax(scores)

print(chunkings[best_index])


#  Reduce to 2 Dimensions and visualize
from sklearn.decomposition import PCA

all_vectors = np.vstack([
    chunk_embeddings,
    query_embedding
])

pca = PCA(n_components=2)

points = pca.fit_transform(all_vectors)

# Save Coordinates
viz_df = pd.DataFrame({
    "label": [
        "Chunk1",
        "Chunk2",
        "Chunk3",
        "Chunk4",
        "Chunk5",
        "Chunk6",
        "Chunk7",
        "Query"
    ],
    "x": points[:,0],
    "y": points[:,1]
})

viz_df.to_csv("embedding_coordinates.csv", index=False)

print(viz_df)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))

for i in range(len(chunkings)):
    plt.scatter(points[i,0], points[i,1])
    plt.annotate(
        f"Chunk {i+1}",
        (points[i,0], points[i,1])
    )

plt.scatter(
    points[-1,0],
    points[-1,1],
    marker="X",
    s=200
)

plt.annotate(
    "QUERY",
    (points[-1,0], points[-1,1])
)

plt.title("Chunk Embeddings vs Query Embedding")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.savefig(
    "embedding_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

print("Plot saved successfully")