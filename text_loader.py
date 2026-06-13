from langchain_community.document_loaders import TextLoader

text_loader=TextLoader('cricket.txt',encoding='utf-8')

results=text_loader.load()

# print(result[0].page_content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",  # ✅ Try this
    task="text-generation"
   
)

model = ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
  
    template="give me small meaning of {text}",
    input_variables=['text']
)
parser=StrOutputParser()

chain =   prompt | model | parser

result=chain.invoke({
  'text':results[0].page_content
})
print(result)