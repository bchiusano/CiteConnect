import streamlit as st
from io import StringIO

st.set_page_config(layout="wide")
st.title("CiteConnect")

# setting the session state variables
if "letter_text" not in st.session_state:
    st.session_state.letter_text = ""

if "generate" not in st.session_state:
    st.session_state.generate = False


def load_file():
    file = st.session_state['txt_file']  # uploaded file when button senses change
    if file:
        data = file.read().decode("utf-8")
        st.session_state.letter_text = data


def generate_clicked():
    st.session_state.generate = True


editor, info = st.columns(spec=[0.5, 0.5],
                          gap="medium",
                          vertical_alignment="top",
                          border=True)

with info:
    st.header("RAG")

    prompt = st.chat_input("Say something")

    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")

    # TODO: if the file is empty you can't click on the button

    generate_button = st.button(
        label="Generate!",
        type="primary",
        on_click= generate_clicked
    )

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
        label="Editor",
        placeholder="Upload a legal advice letter",
        key=st.session_state.letter_text,
        value=st.session_state.get("letter_text", ""),
        label_visibility="hidden",
        height=400)

    download_col, preview_col = st.columns(2, vertical_alignment="bottom")

    with download_col:
        st.download_button(
            label="Download text",
            data=letter_text,
            file_name="letter.txt",  # in the future change
            on_click="ignore",
            type="primary",
            icon=":material/download:",
        )

    with preview_col:
        preview = st.button(
            label="Preview PDF",
            type="primary",
            icon=":material/preview:"
        )


if st.session_state.generate:
    # this part is kept in the state
    # generating citations for the text
    st.write("Generating citations for: ", st.session_state.letter_text)
