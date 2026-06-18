from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# INDEXING -- 

video_id="J5_-l7WIO_w"
try:
   transcript_list=YouTubeTranscriptApi().fetch(video_id=video_id, languages=["hi"])
   transcript=" ".join(chunk.text for chunk in transcript_list)
   print(transcript)


except TranscriptsDisabled:
    print("No captions available for this video.")

# Splitting----

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks=splitter.create_documents([transcript])
print(chunks)

# Storing Vectors---
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store=FAISS.from_documents(chunks,embedding)

print(vector_store.index_to_docstore_id)