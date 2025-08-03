import streamlit as st
from basic_chatbot import chatbot
from langchain_core.messages import HumanMessage

st.title("Chatbot")

CONFIG={"configurable":{"thread_id":"123"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"]=[]

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
 # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    response=chatbot.invoke({"messages":[HumanMessage(content=user_input)]},config=CONFIG)
    ai_response=response["messages"][-1].content
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_response})
    with st.chat_message('assistant'):
        st.markdown(ai_response)


