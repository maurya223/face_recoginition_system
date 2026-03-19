"""Vector database using FAISS for face embeddings with Django model sync fallback."""
import faiss
import numpy as np
import pickle
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "face_reco.settings")
django.setup()

# Django models (lazy import when needed)
_MODELS = None


def get_models():
    from home.models import User
    return {"User": User}
    if _MODELS is None:
        try:
            import django
            if 'django.core.management' not in django.__file__:
                django.setup()
            from face_reco.home.models import User
            _MODELS = {"User": User}
        except ImportError:
            _MODELS = {}
    return _MODELS


DIMENSION = 512  # FaceNet embedding size

index = faiss.IndexFlatL2(DIMENSION)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_INDEX = os.path.join(BASE_DIR, "face_index.faiss")
DB_USERS = os.path.join(BASE_DIR, "users.pkl")
user_names = []
is_loaded = False


def add_user(name, embedding):
    vector = np.array([embedding]).astype("float32")

    index.add(vector)

    user_names.append(name)

    # Sync to Django if available
    try:
        models = get_models()
        User = models["User"]
        embedding_pickled = pickle.dumps(embedding)
        User.objects.update_or_create(
            name=name, defaults={"face_embedding": embedding_pickled}
        )
    except (ImportError, pickle.UnpicklingError) as e:
        print(f"Django sync failed: {e}")


def save_db():
    faiss.write_index(index, DB_INDEX)

    with open(DB_USERS, "wb") as f:
        pickle.dump(user_names, f)

    # Sync to Django if available
    try:
        models = get_models()
        User = models["User"]
        print("Synced %d users to Django" % len(user_names))
    except:
        pass  # Standalone mode


def load_db():
    global index, user_names

    # PRIORITIZE Django SQLite as primary source
    try:
        models = get_models()
        User = models["User"]
        users = User.objects.all()
        if users.exists():
            embeddings = []
            user_names[:] = []  # Clear
            for user in users:
                emb = pickle.loads(user.face_embedding)
                embeddings.append(emb)
                user_names.append(user.name)
            if embeddings:
                global index
                embeddings_np = np.array(embeddings).astype("float32")
                index = faiss.IndexFlatL2(DIMENSION)
                index.add(embeddings_np)
                print("DB loaded from Django SQLite: %d users" % len(users))
                global is_loaded
                is_loaded = True
                return
    except (ImportError, pickle.UnpicklingError) as e:
        print(f"Django load failed: {e}")

    # Fallback to FAISS/PKL
    pkl_exists = os.path.exists(DB_USERS)
    faiss_exists = os.path.exists(DB_INDEX)
    if pkl_exists and faiss_exists:
        index = faiss.read_index(DB_INDEX)
        with open(DB_USERS, "rb") as f:
            user_names = pickle.load(f)
        print("Fallback: Pickle/FAISS loaded: %d users" % len(user_names))
    else:
        print("No database found. Creating new empty database.")
        index = faiss.IndexFlatL2(DIMENSION)
        user_names = []


def search_face(embedding, threshold=0.75):
    vector = np.array([embedding]).astype("float32")

    distances, indices = index.search(vector, 1)

    if len(user_names) == 0:
        return "unknown", 1

    user = user_names[indices[0][0]]

    score = distances[0][0]

    if score > threshold:
        return "unknown", score

    return user, score
