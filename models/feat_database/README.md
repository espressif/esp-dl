# feat_database

A small, self-contained library for on-device verification using feature embeddings.
It stores L2-normalized embeddings for each identity in a flash-backed file and verifies a query embedding with either max cosine similarity or a per-identity subspace distance.

It is model-agnostic and accepts fixed-dimension `float` embeddings.

## API

`namespace dl::feat`

```cpp
#include "dl_feat_verification_database.hpp"

dl::feat::FeatVerificationDatabase db("/spiffs/feat.db", embedding_dim);

db.enroll("alice", embedding);          // add one embedding under a label (auto L2-normalized)
db.build();                             // persist + build subspaces (needs >=2 samples)

db.verify_max_cosine(query, 0.25f);     // cosine-similarity verification
db.verify_subspace(query, threshold);   // subspace-distance verification (>=2 enrollments)

db.delete_embedding("alice", 0);        // remove one embedding (rebuilds + saves)
db.clear();                             // wipe the database
db.print();                             // log a summary table
```

## Notes

- Embeddings are grouped by a user-defined identity `label` (`std::string`).
- Embedding dimension is fixed at construction and validated against the stored database file.
- Saves write to a temporary file first and replace the original database only after a successful write, avoiding partially written database files.

## Build

Add it as a component dependency (e.g. via `idf_component.yml` with an `override_path`)
and `REQUIRES feat_database` in your component's `CMakeLists.txt`.
