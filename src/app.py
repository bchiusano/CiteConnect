import streamlit as st
from run_rag import LegalRAGSystem
import pandas as pd

st.set_page_config(layout="wide")
st.title("CiteConnect")


# Initialize RAG class
@st.cache_resource
def get_rag_instance():
    return LegalRAGSystem()


@st.cache_resource
def get_ecli_data():
    df = pd.read_excel("../data/DATA ecli_nummers juni 2025 v1 (version 1).xlsx")
    return df.set_index('ecli_nummer')


# Get the cached instance
rag = get_rag_instance()
ecli_df = get_ecli_data()

# Initialize session state variables
if "letter_text" not in st.session_state:
    st.session_state.letter_text = ""

if "show_parameters" not in st.session_state:
    st.session_state.show_parameters = False

if "parameters_confirmed" not in st.session_state:
    st.session_state.parameters_confirmed = False

# Store the actual parameters
if "search_params" not in st.session_state:
    st.session_state.search_params = {
        "num_citations": 9,
        "court_decision": "Irrelevant",
        "min_accuracy": 75,
        "additional_requests": ""
    }

if "ecli_list" not in st.session_state:
    st.session_state.ecli_list = []

if 'selected_ecli' not in st.session_state:
    st.session_state.selected_ecli = {}


def load_file():
    """Load uploaded file into session state"""
    file = st.session_state['txt_file']
    if file:
        data = file.read().decode("utf-8")
        st.session_state.letter_text = data


def trigger_parameter_selection():
    """Called when user clicks 'Search Citations' button"""
    st.session_state.show_parameters = True
    # st.session_state.parameters_confirmed = False


def confirm_parameters():
    """Called when user confirms they want to set parameters"""
    st.session_state.parameters_confirmed = True


def decline_parameters():
    """Called when user declines to set parameters"""
    st.session_state.show_parameters = False
    # st.session_state.parameters_confirmed = False
    # st.rerun() # is this necessary?


def save_parameters():
    """Called when user clicks Generate button"""
    st.session_state.show_parameters = False
    st.session_state.parameters_confirmed = False

    # TODO: add domain and other functions
    # results = your_rag_function(
    #     prompt=prompt,
    #     letter_text=st.session_state.letter_text,
    #     num_citations=st.session_state.search_params["num_citations"],
    #     court_decision=st.session_state.search_params["court_decision"],
    #     min_accuracy=st.session_state.search_params["min_accuracy"],
    #     additional_requests=st.session_state.search_params["additional_requests"]
    # )

    st.session_state.ecli_list = rag.get_top_10_for_letter(st.session_state.letter_text)

    # to test ui
    # st.session_state.ecli_list = ['NL:RBAMS:2009:BH5047', 'NL:RBOVE:2018:517', 'NL:RBAMS:2017:1136', 'NL:RVS:2018:3471',
    #                              'NL:CRVB:2018:934',
    #                              'NL:RBAMS:2015:1603', 'NL:RVS:2018:3271', 'NL:CBB:2021:931', 'NL:CRVB:2021:612',
    #                              'NL:RVS:2017:3205']

    st.rerun()


def accept_ecli_selection():
    selected = [ref for ref, choice in st.session_state.selected_ecli.items() if choice == True]

    if selected:
        # if there is at least one ecli selected
        st.session_state.letter_text += f"\n{', '.join(selected)}"


# Create columns
editor, info = st.columns(spec=[0.5, 0.5],
                          gap="medium",
                          vertical_alignment="top",
                          border=True)


def fetch_description(number):
    # assumes that the ecli numbers are unique
    return ecli_df.loc[f"ECLI:{number}", 'ecli_tekst']


with info:
    st.header("RAG")

    # TODO: for button background colour (currently not working)
    st.markdown("""
            <style>
            /* Green for Select buttons */
            button[key^="yes_"] {
                background-color: #4CAF50 !important;
                color: white !important;
            </style>
            """, unsafe_allow_html=True)

    # Always show chat interface first (unless in parameter form)

    # if not st.session_state.parameters_confirmed:
    # Normal chat interface

    # prompt = st.chat_input("Say something")
    # if prompt:
    #     st.write(f"User has sent the following prompt: {prompt}")

    # Display parameters being used
    #     st.info(f"Using parameters: {st.session_state.search_params}")

    # Check if we should show parameter selection
    if st.session_state.show_parameters and not st.session_state.parameters_confirmed:
        # Ask if user wants to add parameters
        st.subheader("Add search parameters?")
        st.write("Would you like to specify parameters for the ECLI citation search?")

        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes", on_click=confirm_parameters, type="primary", use_container_width=True)
        with col2:
            st.button("No", on_click=decline_parameters, use_container_width=True)

    if st.session_state.parameters_confirmed:
        # Show parameter form
        st.subheader("Select parameters for ECLI citation search")
        st.caption("Be aware of possible bias that may come from added parameters.")

        # Number of citations
        st.session_state.search_params["num_citations"] = st.number_input(
            "Number of total citations*",
            min_value=1,
            max_value=100,
            value=st.session_state.search_params["num_citations"],
            step=1
        )

        # Court Decision
        st.session_state.search_params["court_decision"] = st.selectbox(
            "Court Decision (Guilty/Innocent/Irrelevant)*",
            options=["Irrelevant", "Guilty", "Innocent"],
            index=["Irrelevant", "Guilty", "Innocent"].index(
                st.session_state.search_params["court_decision"]
            )
        )

        # Minimum Accuracy
        st.session_state.search_params["min_accuracy"] = st.number_input(
            "Minimum Accuracy (0%-100%)*",
            min_value=0,
            max_value=100,
            value=st.session_state.search_params["min_accuracy"],
            step=1
        )

        # Additional requests
        st.session_state.search_params["additional_requests"] = st.text_area(
            "Any additional requests?",
            value=st.session_state.search_params["additional_requests"],
            placeholder="Quote only the numbered paragraphs",
            height=100
        )

        # Generate button
        if st.button("Generate", type="primary", use_container_width=True):
            save_parameters()

    # Display results
    elif st.session_state.ecli_list:

        ecli_list = st.session_state.ecli_list

        descriptions = {}

        for ecli in ecli_list:

            # fetching the descriptions of the ecli from data
            descriptions[ecli] = fetch_description(ecli)

            # select functionality for session state
            if ecli not in st.session_state.selected_ecli:
                st.session_state.selected_ecli[ecli] = False

            ecli_name, buttons = st.columns([8, 1])  # 8:1 ratio

            with ecli_name:
                # user can expand to see the description of the ecli
                with st.expander(
                        f"{'✅' if st.session_state.selected_ecli[ecli] == True else '❌'} {ecli}"):
                    if descriptions[ecli]:
                        st.write(descriptions[ecli])

            with buttons:
                if st.button("✓", key=f"yes_{ecli}"):
                    st.session_state.selected_ecli[ecli] = True
                    st.rerun()

        change_params, regenerate, accept = st.columns(3)
        with change_params:
            st.button("Update Parameters", type='primary')  # TODO
        with regenerate:
            st.button("Regenerate", type='primary')     # TODO
        with accept:
            st.button("Accept", type='primary', on_click=accept_ecli_selection)


with editor:
    st.header("Editor")

    # File uploader
    upload = st.file_uploader(
        label="Upload Letter",
        accept_multiple_files=False,
        type=".txt",
        key="txt_file",
        label_visibility="hidden",
        on_change=load_file)

    # Visual text area - always enabled
    letter_text = st.text_area(
        label="Editor",
        placeholder="Upload a legal advice letter or type directly to begin...",
        value=st.session_state.get("letter_text", ""),
        label_visibility="hidden",
        height=400
    )

    # Update session state if text is manually edited
    if letter_text != st.session_state.letter_text:
        st.session_state.letter_text = letter_text

    # Check if editor has content (from file OR manual typing)
    has_content = st.session_state.letter_text.strip() != ""

    # Buttons row
    download_col, preview_col, search_col = st.columns(3, vertical_alignment="bottom")

    with download_col:
        st.download_button(
            label="Download text",
            data=letter_text if has_content else "",
            file_name="letter.txt",
            type="primary",
            icon=":material/download:",
            disabled=not has_content
        )

    with preview_col:
        preview = st.button(
            label="Preview PDF",
            type="primary",
            icon=":material/preview:",
            disabled=not has_content
        )

        if preview and has_content:
            st.info("Preview functionality to be implemented")

    with search_col:
        # New button to trigger citation search
        st.button(
            label="Add Citations",
            type="primary",
            icon=":material/search:",
            on_click=trigger_parameter_selection,
            disabled=not has_content
        )
