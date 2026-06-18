from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import  matplotlib.pyplot as plt

import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                       device = "cuda")


texts = [
    "Refund policy allows refunds within 30 days."
] * 10000

embeddings = model.encode(
    texts,
    batch_size=128,
    show_progress_bar=True
)