from chatbot_ipynb_to_pyhton import agent
# Main chat function
def ask_fifa_bot(message: str, thread_id: str = "default") -> str:
    
    config = {"configurable": {"thread_id": thread_id}}
    
    result = agent.invoke({"messages": message}, config)
    
    return result["messages"][-1].content


ask_fifa_bot("Tell me about Brazil vs South Korea match", "new_user_023")