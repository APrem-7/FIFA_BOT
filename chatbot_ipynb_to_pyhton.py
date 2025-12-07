# %% [markdown]
# THIS IS THE NOTEBEOOK WHERE WE ARE GOING TO FIRST INITIALIZE A MODEL AND THEN THEN EMBEDD THE DATA INTO THE RAG PIPELINE
# 

# %%
import pandas as pd
from langchain_core.documents import Document

# Load the FIFA matches data
df = pd.read_csv("data/Fifa_world_cup_matches.csv")  # adjust path if needed

# Quick peek
print(df.head())
print(df.columns)


# %%


# %% [markdown]
# converting each row of the data into text to embedded into the vector Store

# %%
df.columns


# %%
df.columns

# %%
def row_to_text(row):
    # Adjust column names to your actual CSV headers
    return (
       
        f"FIFA World Cup 2022 Match {row['team1']} vs {row['team2']}. "
        f"Score: {row['number of goals team1']} - {row['number of goals team2']}. "
        f"Date: {row['date']}. "
        f"Category:{row['category']} ,"
        f"Possesion of {row['team1']} is {row['possession team1']} and {row['team2']} is {row['possession team2']} "
        f"{row['team1']} had {row['on target attempts team1']} On Target Attempts and {row['team2']} had  {row['on target attempts team2']} attempts"
         f"{row['team1']} had {row['assists team1']} assists and {row['team2']} had {row['assists team2']} assists"
    )

docs= []
for _, row in df.iterrows():
    text = row_to_text(row)
    metadata = {
        "team1": row["team1"],
        "team2": row["team2"],
        "category":row['category']
    }
    docs.append(Document(page_content=text, metadata=metadata))

len(docs), docs[0]


# %%
print(docs)

# %% [markdown]
# INTIATING THE EMBEDDING MODEL

# %%
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
)



# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_core.documents import Document  

def load_and_chunk_all_text_files(data_dir="data", chunk_size=500, chunk_overlap=100):
    """
    Load all .txt files from data directory and chunk them recursively
    
    Args:
        data_dir: Path to data directory
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Document objects with chunked content
    """
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    docs = []
    data_path = Path(data_dir)
    
    # Recursively find all .txt files
    for txt_file in data_path.rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split the text into chunks
            chunks = text_splitter.split_text(content)
            
            # Create documents from chunks
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": str(txt_file),
                        "file_name": txt_file.stem,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                )
                docs.append(doc)
                
            print(f"✅ Loaded {txt_file.name}: {len(chunks)} chunks")
        
        except Exception as e:
            print(f"❌ Error loading {txt_file.name}: {e}")
    
    print(f"\n📊 Total documents created: {len(docs)}")
    return docs

# Usage in your notebook:
# text_docs = load_and_chunk_all_text_files("data")
# Then add these to your vector store along with CSV docs

# %%
from langchain_community.vectorstores import FAISS
csv_docs = docs  # your existing CSV documents

# Load and chunk all txt files
text_docs = load_and_chunk_all_text_files("data")

# Combine both
all_docs = csv_docs + text_docs

# Create vector store with all documents
vector_store = FAISS.from_documents(all_docs, embedding=embeddings)

# %% [markdown]
# TEXT SPLITTER FOR THE RECAPS DATA
# 

# %% [markdown]
# Vector Storing

# %% [markdown]
# Now that the vector embeddings are completed we will now be doing a test for similarity search so that we can see what is actually going on here now using similiratiy_score_threshhold
# 
# similarity = “just give me closest stuff”
# 
# similarity_score_threshold = “closest stuff but drop trash”
# 
# mmr = “closest but also diverse (good for longer answers)”

# %%
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 10,                 # consider up to 10
        "score_threshold": 0.9   # filter out low similarity
    }
)



# %%
retriever_s = vector_store.as_retriever(
    search_type="similarity",               # or "similarity_score_threshold", "mmr"
    search_kwargs={"k": 5}                  # top 5 matches
)


# %%
prompt = "Tell me about the match between Portugal and Morocco"  # ⬅️ Add this

results = retriever_s.invoke(prompt)   # ⬅️ this replaces get_relevant_documents

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
    print("-" * 40)

# %% [markdown]
# Now Setting Up The Actualy LLM and then seeing what can be done

# %%
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool,ToolRuntime

# %%
llm = ChatOllama(
    model="llama3.2:3b",  # or any local ollama model name you’ve pulled
    temperature=0.2,
)

checkpointer = InMemorySaver()

# %%

from langchain_core.tools import create_retriever_tool

retriever_tool = create_retriever_tool(retriever_s,name='retriever_data',description='Usign this tool we will retrive the data from the vector store')



# %%
@tool
def retrieve_data(query: str) -> str:
    """Retrieve World Cup match data from the vector store"""
    results = retriever_s.invoke(query)
    return "\n\n".join([doc.page_content for doc in results])

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=[retrieve_data],
    checkpointer=checkpointer,
    system_prompt="""You are a FIFA Football Retrieval & Analysis Assistant.

Your job is to answer questions only using the information retrieved from the provided knowledge base or documents, and explain things in a simple, structured, and easy-to-follow way, like a smart football analyst teaching a fan.

1. Retrieval Rules

Always perform a retrieval step before answering.

If no relevant information is found, reply exactly with:

No data available in the current database.

Never invent facts.

No guessing.

No assumptions beyond what is explicitly in the retrieved content.

2. Answering Style

When you answer:

Start with a 1–2 line summary of the key point or outcome.

Example: “Brazil dominated chances, but Croatia’s mentality and penalties decided the match.”

Then give a short, clear explanation focusing on:

What happened (facts).

Why it mattered (impact, context).

How it happened (tactics, key players, key moments), but only if this is supported by the retrieved data.

Use simple, direct language and avoid long paragraphs. Prefer:

Short paragraphs

Bullet points

Clear section titles (e.g., “Possession”, “Key Players”, “Penalty Shootout”).

If the question is about stats, show them in a:

Neat bullet list, or

Table, if there are many numbers to compare.

Do not just list stats. Always add one or two lines of interpretation based on the retrieved data:

For example, explain what a big shots-on-target difference suggests about the game.

Keep answers concise and factual, but give enough context so a normal fan can understand the “story” behind the numbers.

3. Conflicting or Limited Data

If multiple retrieved documents conflict, explicitly mention the conflict and do not pick a side.

Example: “One source says X, another says Y. The data is inconsistent, so I can’t be sure which is correct.”

If the data is incomplete, be honest about the limits:

Example: “The database only includes league games for this season, so cup matches are not reflected here.”

4. Match Recaps

When the user asks for a match recap and the data is available, include (if present in the retrieved content):

Final score

Goal scorers and minutes

Major events, such as:

Red/yellow cards

Penalties (given, scored, missed)

Important substitutions or injuries

Any key stats that help explain the game (e.g., shots, possession, xG)

A short 2–3 sentence explanation of the overall story of the match (e.g., dominance vs. efficiency, comeback, defensive masterclass).

5. Context: Teams, Players, Competitions

If the result involves teams or players and the information is available in the retrieved data, always mention:

Season

Competition

Scoreline

This helps keep answers precise and avoids confusion between different seasons or tournaments.

6. Clarifications

If the user asks something vague, like:

“How was Madrid’s season?”

“How did Messi play that year?”

then ask for the specific season/year and competition before giving an answer.

If the user asks a question that is unrelated to football, politely refuse:

Example: “I’m only able to answer football-related questions using this database.”

7. Tone

Friendly but direct, like a football commentator who knows their stats and can explain them simply.

No overhyping, no drama for its own sake.

Focus on clarity, accuracy, and helping the user understand the why and how, not just the raw numbers."""
)

# # %%
# config={"configurable": {"thread_id": "9"}}

# result=agent.invoke({
#     "messages":[{"role":"user","content":"Hey I am Ananda Prem How are you doing"}]
# },config)

# resulted=agent.invoke({
#     "messages":[{"role":"user","content":"Tell Me about Portugal Vs Morroc's match in the FIFA 2022 world cup"}]
# },config)

# result_2=agent.invoke({
#     "messages":[{"role":"user","content":"What would you say about the key moment where their fate in the worldcup was decided"}]
# },config)



# # %%
# print("Result 1:")
# print(resulted["messages"][-1].content)
# print("\nResult 2:")
# print(result_2["messages"][-1].content)

# print("\nResult 3:")

# print(result["messages"][-1].content)

# # %%


# # %%


# # %%
# print("Result 3 (Full):")
# print(result_3["messages"][-1].content)


# # %%
# result["messages"][-1].pretty_print()
# result_2["messages"][-1].pretty_print()
# result_3["messages"][-1].pretty_print()


# # %%
# config={"configurable": {"thread_id": "1"}}

# agent.invoke({"messages": "hi, my name is Nathan"}, config)
# resp=agent.invoke({"messages": "Tell me about Portugals Match Against Morroco"}, config)
# resp=agent.invoke({"messages": "What Would You tell About the possession did portugal deserve to win based off the possesion they maintained throught the match"}, config)
# resp["messages"][-1].pretty_print()
# resp=agent.invoke({"messages":"how many attempts did portugal have in this match compared to morroco?"},config)
# resp["messages"][-1].pretty_print()


# # %%
# config={"configurable": {"thread_id": "1"}}
# resp=agent.invoke({"messages":"What About Total Attempts On and Off Target"},config)
# resp["messages"][-1].pretty_print()

# %%


# %% [markdown]
# Creating a Function to Get Input from a User instead Of hardCoding it

# %% [markdown]
# its now inside main_func.py from where we are going to call it

# %%
from main_func import ask_fifa_bot


# # %%
# # Test with new thread and different question
# print(ask_fifa_bot("Tell me about Brazil vs South Korea match", "new_user_123"))
# print("\n" + "="*60 + "\n")
# print(ask_fifa_bot("What were the possession stats for Japan vs Croatia?", "new_user_456"))

# # %%
# print(ask_fifa_bot("Tell me about Brazil vs South Korea match", "new_user_023"))

# # %% [markdown]
# # HElllo


