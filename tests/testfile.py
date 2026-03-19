from app.recognition import cosine_similarity
import numpy as np

a = np.random.rand(256)
b = np.random.rand(256)

print(cosine_similarity(a,b))