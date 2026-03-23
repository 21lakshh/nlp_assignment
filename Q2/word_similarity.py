import gensim.downloader as api

def main():
    print("Loading pre-trained GloVe embeddings (glove-wiki-gigaword-50)...")
    # Load a relatively small pre-trained GloVe model
    # (50-dimensional embeddings based on Wikipedia 2014 + Gigaword 5)
    model = api.load("glove-wiki-gigaword-50")
    
    word_pairs = [
        ("king", "queen"),
        ("doctor", "nurse"),
        ("car", "tree")
    ]
    
    print("\nCalculating cosine similarity for the specific word pairs:\n")
    print(f"{'Word Pair':<20} | {'Cosine Similarity':<15}")
    print("-" * 40)
    
    for word1, word2 in word_pairs:
        # Check if both words exist in the model's vocabulary
        if word1 in model.key_to_index and word2 in model.key_to_index:
            # model.similarity computes the cosine similarity between two words
            similarity = model.similarity(word1, word2)
            print(f"{word1 + ' - ' + word2:<20} | {similarity:.4f}")
        else:
            print(f"{word1 + ' - ' + word2:<20} | One or both words not in vocabulary")

if __name__ == "__main__":
    main()
