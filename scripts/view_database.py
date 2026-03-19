import pickle
import faiss
import os

users_path = "database/users.pkl"
index_path = "database/face_index.faiss"

if not os.path.exists(users_path):
    print("No users database found")
else:
    with open(users_path, "rb") as f:
        users = pickle.load(f)

    print("\nRegistered Users:")
    print("------------------")

    for i, user in enumerate(users):
        print(f"{i+1}. {user}")

    print("\nTotal Users:", len(users))


if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    print("Total Embeddings:", index.ntotal)
else:
    print("No FAISS index found")