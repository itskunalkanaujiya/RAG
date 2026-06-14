from langchain_text_splitters import RecursiveCharacterTextSplitter


splitter=RecursiveCharacterTextSplitter(
  chunk_size=10,
  chunk_overlap=0
)

result=splitter.split_text(
  """
My name is Kunal.
I live in Guragon.

I study in Europe.
My hobby is playing cricket.
"""
)

print(result)