from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# INDEXING -- 

video_id="J5_-l7WIO_w"
try:
   transcript_list=YouTubeTranscriptApi().fetch(video_id=video_id, languages=["hi"])
   transcript=" ".join(chunk.text for chunk in transcript_list)
#    print(transcript)


except TranscriptsDisabled:
    print("No captions available for this video.")

# Splitting----

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=splitter.create_documents([transcript])
# print(chunks)

# Storing Vectors---
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=FAISS.from_documents(chunks,embedding)

# print(vector_store.index_to_docstore_id)


# Retriever ---

retriever=vector_store.as_retriever(search_type="similarity",kwargs={"k":4})





# Augmentation ---

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",  # ✅ Try this
    task="text-generation"
   
)

model = ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question="is LLM discussed in this video? if yes then what was discussed"

retrieved_doc=retriever.invoke(question)

context="/n/n".join(doc.page_content for doc in retrieved_doc)

final_prompt=prompt.invoke(
    {
        "context":context,
        "question":question
    }
)

result=model.invoke(final_prompt)
# print(result.content)


# Buildinding  Chain--
def chunks_making(retrieved_doc):
    context="/n/n".join(doc.page_content for doc in retrieved_doc)
    return context

parallel_chain=RunnableParallel(
    {
    "context":retriever | RunnableLambda(chunks_making),
    "question": RunnablePassthrough()
    }
)

parser=StrOutputParser()
chain=parallel_chain | prompt | model | parser
result=chain.invoke("can you summarize about the project discussed in this video?")
print(result)
