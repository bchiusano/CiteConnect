import streamlit as st
from io import StringIO

st.set_page_config(layout = "wide")
st.title("CiteConnect")

# setting the session state variables
if "letter_text" not in st.session_state:
    st.session_state.letter_text = ""
                   

def load_file():
    
    file = st.session_state['txt_file'] # uploaded file when button senses change
    if file:
        data = file.read().decode("utf-8")
        st.session_state.letter_text = data



editor, info = st.columns(spec=[0.5,0.5],
                          gap="medium", 
                          vertical_alignment="top", 
                          border=True)

with info: 
    st.header("RAG")

    prompt = st.chat_input("Say something" )
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")


with editor:

    
    st.header("Editor")
    
    # TODO: create your own component (upload button) because this is too big
    upload = st.file_uploader(
        label="Upload Letter", 
        accept_multiple_files=False, 
        type=".txt",
        key="txt_file",
        label_visibility="hidden",
        on_change=load_file)


    # Visual text area
    letter_text = st.text_area(
        label = "Editor", 
        placeholder="Upload a legal advice letter", 
        key=st.session_state.letter_text,
        value = st.session_state.get("letter_text", ""),
        label_visibility="hidden",
        height=400)
    

    download_col, preview_col = st.columns(2, vertical_alignment="bottom")

    with download_col:
        ## Download button
        st.download_button(
            label="Download text",
            data=letter_text, 
            file_name="letter.txt", # in the future change
            on_click="ignore",
            type="primary",
            icon=":material/download:",
        )


    with preview_col: 
        ## Preview button 
        # TODO: change icon to preview off if the file is not present
        preview = st.button(
            label= "Preview PDF",
            type="primary",
            icon=":material/preview:"
        )