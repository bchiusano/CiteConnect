import streamlit as st
from io import StringIO

st.set_page_config(layout = "wide")
st.title("CiteConnect")
                   

## Upload data
def get_letter_text():
    
    
    st.session_state['letter_area'] = "hello"



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

    # Visual text area
    # on change?
    letter_text = st.text_area(label = "Editor", placeholder="Upload a legal advice letter", key="letter_area")
    
    download_col, preview_col, upload_col = st.columns(3, vertical_alignment="bottom")


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

    with upload_col:
        ## Upload button
        upload = st.button(
            label= "Upload Letter",
            type="primary",
            #on_click = get_letter_text(),
            icon=":material/upload_file:"
        )

    #upload = st.file_uploader(
    #    "Import Legal Advice Letter", 
    #    accept_multiple_files = False, 
    #    type=".txt",
    #    key= "uploaded_letter",
    #    on_change= get_letter_text()
    #)

    



