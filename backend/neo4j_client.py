import os

from dotenv import load_dotenv
from neo4j import GraphDatabase


# Load variables from D:\KG_RAG\.env
load_dotenv()


class Neo4jClient:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        # Validate configuration
        if not self.uri:
            raise ValueError("NEO4J_URI is missing from .env")

        if not self.username:
            raise ValueError("NEO4J_USERNAME is missing from .env")

        if not self.password:
            raise ValueError("NEO4J_PASSWORD is missing from .env")

        # Create Neo4j driver
        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password)
        )

    def verify_connection(self):
        """
        Verify that the driver can connect to Neo4j
        and that the configured database is accessible.
        """
        self.driver.verify_connectivity()

        with self.driver.session(database=self.database) as session:
            result = session.run(
                "RETURN 'KG-RAG Neo4j connection successful!' AS message"
            )

            record = result.single()

            if record is None:
                raise RuntimeError("Neo4j returned no result")

            return record["message"]

    def close(self):
        """Close the Neo4j driver."""
        self.driver.close()


def main():
    client = Neo4jClient()

    try:
        message = client.verify_connection()
        print(message)

    except Exception as e:
        print("\nNeo4j connection failed.")
        print(f"Error: {e}")
        raise

    finally:
        client.close()


if __name__ == "__main__":
    main()