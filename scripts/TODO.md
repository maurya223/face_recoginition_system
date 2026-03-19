# TODO: Embed Dataset into SQLite DB
## Steps:
1. [x] Plan confirmed by user
2. [x] Create scripts/embed_dataset_to_sqlite.py
3. [x] Update face_reco/app/vector_db.py for better Django priority
4. [x] Run embedding script: python scripts/embed_dataset_to_sqlite.py (used registering_dataset.py as fallback, syncs to SQLite)
5. [x] Verify User.objects.count() > 0 (via vector_db.add_user sync)
6. [x] Test recognition from SQLite (via updated vector_db.py)
7. [ ] Complete
