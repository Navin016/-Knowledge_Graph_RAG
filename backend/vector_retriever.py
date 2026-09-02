from backend.embedding_model import get_rag_embedding_model

from backend.neo4j_client import Neo4jClient


MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_INDEX_NAME = "chunk_text_vector"


class VectorRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.client = Neo4jClient()

       
        self.model = get_rag_embedding_model()
        

    def search(self, query: str):
        """
        Embed the user query and perform vector search in Neo4j.
        """

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        # -----------------------------------------------------
        # Create query embedding
        # -----------------------------------------------------

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # -----------------------------------------------------
        # Neo4j vector search
        # -----------------------------------------------------

        cypher = """
       MATCH (node:Chunk)
SEARCH node IN (
    VECTOR INDEX chunk_text_vector
    FOR $query_embedding
    LIMIT $top_k
)
SCORE AS score

RETURN
    node.chunk_id AS chunk_id,
    node.source_file AS source_file,
    node.text AS text,
    score

ORDER BY score DESC
        """

        with self.client.driver.session(
            database=self.client.database
        ) as session:

            result = session.run(
                cypher,
                index_name=VECTOR_INDEX_NAME,
                top_k=self.top_k,
                query_embedding=query_embedding
            )

            return [record.data() for record in result]

    def close(self):
        self.client.close()


def main():
    retriever = VectorRetriever(top_k=5)

    try:
        query = (
            "How does fluorescence imaging reduce "
            "motion artifacts?"
        )

        print()
        print("=" * 80)
        print("VECTOR SEARCH")
        print("=" * 80)

        print(f"Query: {query}")

        results = retriever.search(query)

        print()
        print(f"Retrieved {len(results)} chunks")
        print()

        for i, result in enumerate(results, start=1):

            print("-" * 80)
            print(f"RESULT {i}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Score:    {result['score']:.4f}")
            print(f"Source:   {result['source_file']}")
            print()
            print(result["text"][:1000])

    finally:
        retriever.close()


if __name__ == "__main__":
    main()