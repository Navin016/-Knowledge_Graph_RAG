from backend.extraction import extract_triples


text = """
Neo4j is a graph database that stores information as
nodes and relationships. Neo4j uses Cypher as its query
language. LangChain provides GraphCypherQAChain, which
converts natural language questions into Cypher queries.
"""


triples = extract_triples(text)


print("\nExtracted triples:")
print("=" * 60)


for triple in triples:

    print(
        f"{triple.subject} "
        f"--[{triple.relation}]--> "
        f"{triple.object}"
    )