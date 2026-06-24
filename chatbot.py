from langchain_core.tools import retriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_community.vectorstores import FAISS
from openai import vector_stores
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv() 

ChatOpenAI(
    os.getenv('OPENAI_MODEL')
)

# ======> step-1 vector store generation 18 <======
#transcript generation from the youtube video

video_id= "mitxLXzYjQRE"
try:
    api=YouTubeTranscriptApi() 
    fetched_transcript=api.fetch(video_id, Languages=["en"])
    transcript=" ".join([snippet.text for snippet in fetched_transcript.snippets])
except TranscriptsDisabled:
    print("No captions available for this video")

# ========> RAG pipeline <========
#text-splitter
splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks=splitter.create_documents([transcript])
print("chunks count", len(chunks))

#convert to vector and store in vector-store

embeddings=OpenAIEmbeddings(model="text-embedding-3-small") 
vector_store=FAISS.from_documents(chunks, embeddings)
print("vector store created")
#print(vector_store.index_to_docstore_id)


# =======> step-2 retrieval <========
retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print("retriever created")
#print(retriever)
#retriever.invoke("What is rag")

# =======>  step-3 augmentation <======
llm=ChatOpenAI(model="gpt-40", temperature=0.2)
prompt=PromptTemplate(
    template="""
    You are a helpful assistant that can answer questions about the youtube video.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)
#question="is the topic of production rag discussed in this video? if yes then what was discussed"
#retrieved_docs-retriever.invoke(question)
# context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
#final prompt prompt.invoke(("context": context_text, "question": question})
# step-4 generation
#ans=llm.invoke(final prompt)
#print(ans)

#chain
def format_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain=RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
#parallel_chain. invoke('what is rag')

#parser
parser=StrOutputParser()
main_chain=parallel_chain | prompt | llm | parser
result=main_chain.invoke('what is rag')
print(result)