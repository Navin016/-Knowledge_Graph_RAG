from backend.embedding_model import get_rag_embedding_model
from backend.neo4j_client import Neo4jClient


ENTITY_VECTOR_INDEX = "entity_name_vector"


def create_entity_vector_index(
    client: Neo4jClient,
):
    """
    Create a vector index for Entity.embedding.

    all-MiniLM-L6-v2 produces 384-dimensional embeddings.
    """

    with client.driver.session(
        database=client.database
    ) as session:

        session.run(
            """
            CREATE VECTOR INDEX entity_name_vector IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: 384,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        )

    print("Entity vector index created/verified.")


def embed_entities(
    client: Neo4jClient,
    model,
):
    """
    Load every Entity name and create its embedding.

    Existing entities are updated; no new entities are created.
    """

    with client.driver.session(
        database=client.database
    ) as session:

        result = session.run(
            """
            MATCH (e:Entity)
            RETURN e.name AS name
            ORDER BY e.name
            """
        )

        names = [
            record["name"]
            for record in result
        ]

    if not names:
        print("No Entity nodes found.")
        return

    print()
    print(f"Entities found: {len(names)}")
    print("Generating entity embeddings...")

    embeddings = model.encode(
        names,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("Entity embeddings generated.")

    with client.driver.session(
        database=client.database
    ) as session:

        for name, embedding in zip(
            names,
            embeddings,
        ):
            session.run(
                """
                MATCH (e:Entity {name: $name})
                SET e.embedding = $embedding
                """,
                name=name,
                embedding=embedding.tolist(),
            )

    print(
        f"Updated embeddings for {len(names)} entities."
    )


def verify_entity_index(
    client: Neo4jClient,
):
    """
    Verify the vector index state.
    """

    with client.driver.session(
        database=client.database
    ) as session:

        result = session.run(
            """
            SHOW VECTOR INDEXES
            YIELD name, state, populationPercent
            WHERE name = $index_name
            RETURN name, state, populationPercent
            """,
            index_name=ENTITY_VECTOR_INDEX,
        )

        record = result.single()

        if record is None:
            print("Entity vector index not found.")
            return

        print()
        print("Entity vector index:")
        print(f"Name:             {record['name']}")
        print(f"State:             {record['state']}")
        print(f"Population:       {record['populationPercent']}%")


def main():

    client = Neo4jClient()

    try:

        client.driver.verify_connectivity()

        print("Neo4j connection verified.")

        # -----------------------------------------------------
        # Create entity vector index
        # -----------------------------------------------------

        create_entity_vector_index(
            client
        )

        # -----------------------------------------------------
        # Get shared RAG embedding model
        # -----------------------------------------------------

        model = get_rag_embedding_model()

        # -----------------------------------------------------
        # Embed existing entities
        # -----------------------------------------------------

        embed_entities(
            client,
            model
        )

        # -----------------------------------------------------
        # Verify
        # -----------------------------------------------------

        verify_entity_index(
            client
        )

        print()
        print(
            "Entity vector setup complete."
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()