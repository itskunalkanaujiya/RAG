from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('dl-curriculum.pdf')

result=loader.load()

print(len(result))