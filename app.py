


from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
print("🚀 Current working directory:", os.getcwd())
print("📂 Files & folders here:", os.listdir())
print("✅ Does 'templates/index.html' exist?:", os.path.exists('templates/index.html'))


app = Flask(__name__, template_folder='templates')


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "medicalbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

chatModel = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')




@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response["answer"]
    except Exception as e:
        print("Error:", e)
        answer = "⚠ Sorry, the bot can't answer now (maybe out of quota)."
    return str(answer)




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

