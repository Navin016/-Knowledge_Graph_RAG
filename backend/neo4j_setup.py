from backend.neo4j_client import Neo4jClient


def setup_schema():
    client = Neo4jClient()

    try:
        with client.driver.session(database=client.database) as session:

            # ---------------------------------------------------------
            # Entity uniqueness
            # ---------------------------------------------------------
            session.run("""
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity)
                REQUIRE e.name IS UNIQUE
            """)

            # ---------------------------------------------------------
            # Chunk uniqueness
            # Use chunk_key so multiple PDFs can safely coexist.
            # ---------------------------------------------------------
            session.run("""
                CREATE CONSTRAINT chunk_key_unique IF NOT EXISTS
                FOR (c:Chunk)
                REQUIRE c.chunk_key IS UNIQUE
            """)

            # ---------------------------------------------------------
            # Vector index
            #
            # all-MiniLM-L6-v2 produces 384-dimensional embeddings.
            # ---------------------------------------------------------
            session.run("""
                CREATE VECTOR INDEX chunk_text_vector IF NOT EXISTS
                FOR (c:Chunk) ON (c.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }
                }
            """)

        print("Neo4j schema setup completed successfully.")

    finally:
        client.close()


if __name__ == "__main__":
    setup_schema()