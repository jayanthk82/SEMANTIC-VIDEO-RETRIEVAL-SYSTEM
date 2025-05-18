import chromadb
def chromadb_setup(setup_path, collection_name, vector, samples_count, summary):
    client = chromadb.PersistentClient(path=setup_path)
    collection = client.get_or_create_collection(name=collection_name)

    if not vector:
        return
    print('inserting into chromaDB:')
    for i in range(samples_count):
        collection.add(
            ids=[str(i)],
            embeddings=[vector[i]],
            documents=[summary[i]]
        )
    return collection
