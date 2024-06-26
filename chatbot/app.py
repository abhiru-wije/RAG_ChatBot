from rag_functionality import rag_function
import streamlit as st

st.title("AI Chat Assistant")
st.subheader("Chat with an AI to get your questions answered.")

# set initial message
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, how can I help you"}
    ]


if "messages" in st.session_state.keys():
    # display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


# get user input
user_prompt = st.chat_input()


if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = rag_function(user_prompt)
            st.write(ai_response)
            
    new_ai_message = {"role": "user", "content": ai_response}
    st.session_state.messages.append(new_ai_message)