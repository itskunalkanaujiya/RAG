from langchain_community.retrievers import WikipediaRetriever

retriever=WikipediaRetriever(
  top_k_results=2,
  lang='en'
)

query = "the geopolitical history of india and pakistan from the perspective of a chinese"

docs=retriever.invoke(query)

for i,doc in enumerate(docs):
  print(f"\n --- RESULT {i+1}---")
  print(f" Page Content -- {doc.page_content}")