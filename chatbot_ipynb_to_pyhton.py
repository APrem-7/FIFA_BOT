# %% [markdown]
# THIS IS THE NOTEBEOOK WHERE WE ARE GOING TO FIRST INITIALIZE A MODEL AND THEN THEN EMBEDD THE DATA INTO THE RAG PIPELINE
# 

# %%
import pandas as pd
from langchain_core.documents import Document

# Load the FIFA matches data
df = pd.read_csv("data/Fifa_world_cup_matches.csv")  # adjust path if needed

# Quick peek
# print(df.head())
# print(df.columns)


# %%


# %% [markdown]
# converting each row of the data into text to embedded into the vector Store

# %%
# df.columns


# %%
# df.columns

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

docs = []
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
# print(docs)

# %% [markdown]
# INTIATING THE EMBEDDING MODEL

# %%
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large:latest",
)



# %% [markdown]
# TEXT SPLITTER FOR THE RECAPS DATA
# 

# %% [markdown]
# Vector Storing

# %%
from langchain_community.vectorstores import FAISS
vector_store = FAISS.from_documents(docs, embedding=embeddings)


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


# %%
retriever_s = vector_store.as_retriever(
    search_type="similarity",               # or "similarity_score_threshold", "mmr"
    search_kwargs={"k": 5}                  # top 5 matches
)


# %%


# %%


# %%
# prompt = "Tell me about the match between Portugal and Morocco"  # ⬅️ Add this

# results = retriever_s.invoke(prompt)   # ⬅️ this replaces get_relevant_documents

# for doc in results:
#     print(doc.page_content)
#     print(doc.metadata)
#     print("-" * 40)

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
    system_prompt="""You are a FIFA Football Retrieval Assistant.
Your job is to answer questions only using the information retrieved from the provided knowledge base or documents.

Rules:

Always perform a retrieval step before answering.
If nothing relevant is found, say:

“No data available in the current database.”

Never invent facts.
No guessing, no assumptions beyond retrieved content.

Keep answers concise, factual, and match football terminology.

If the user asks for stats, show them in a neat bullet list or table.

If multiple retrieved documents conflict, mention the conflict and do NOT pick a side.

Do not summarize entire documents; answer only the question asked.

Style:

Friendly but direct, like a football commentator who knows their stats.

If a result involves teams or players, always mention season, competition, and score if available.

When giving match recaps, include:

final score

goal scorers

major events (cards, penalties, substitutions)

Clarifications:

If the user asks something vague (“How was Madrids season?”), request a specific season/year before proceeding.

If the query is unrelated to football, politely refuse."""
)

# %%
# config={"configurable": {"thread_id": "9"}}

# result=agent.invoke({
#     "messages":[{"role":"user","content":"Hey I am Ananda Prem How are you doing"}]
# },config)

# resulted=agent.invoke({
#     "messages":[{"role":"user","content":"How Did Urugay Perform in this World Cup"}]
# },config)

# result_2=agent.invoke({
#     "messages":[{"role":"user","content":"What would you say about the key moment where their fate in the worldcup was decided"}]
# },config)



# %%
# print("Result 1:")
# print(resulted["messages"][-1].content)

# print("\nResult 2:")
# print(result_2["messages"][-1].content)

# print("\nResult 3:")
# print(result["messages"][-1].content)

# %%


# %%
# print("Result 3 (Full):")
# print(result_3["messages"][-1].content)


# %%
# result["messages"][-1].pretty_print()
# result_2["messages"][-1].pretty_print()
# result_3["messages"][-1].pretty_print()


# %%
# config={"configurable": {"thread_id": "1"}}

# agent.invoke({"messages": "hi, my name is Nathan"}, config)
# resp=agent.invoke({"messages": "Tell me about Portugals Match Against Morroco"}, config)
# resp=agent.invoke({"messages": "What Would You tell About the possession did portugal deserve to win based off the possesion they maintained throught the match"}, config)
# resp["messages"][-1].pretty_print()
# resp=agent.invoke({"messages":"What is my name again ?"},config)
# resp["messages"][-1].pretty_print()


# %%
# config={"configurable": {"thread_id": "1"}}
# resp=agent.invoke({"messages":"Tell me about Portugals Match Against Morroco"},config)
# resp["messages"][-1].pretty_print()

# %% [markdown]
# Creating a Function to Get Input from a User instead Of hardCoding it

# %% [markdown]
# its now inside main_func.py from where we are going to call it

# %%
from main_func import ask_fifa_bot


# %%
# Test with new thread and different question
# print(ask_fifa_bot("Tell me about Brazil vs South Korea match", "new_user_123"))
# print("\n" + "="*60 + "\n")
# print(ask_fifa_bot("What were the possession stats for Japan vs Croatia?", "new_user_456"))

# # %%
# print(ask_fifa_bot("Tell me about Brazil vs South Korea match", "new_user_023"))

# %% [markdown]
# HElllo


