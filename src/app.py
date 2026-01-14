import streamlit as st

st.set_page_config(layout="wide")
st.title("CiteConnect")

# Initialize session state variables
if "letter_text" not in st.session_state:
    st.session_state.letter_text = ""

# if "generate" not in st.session_state:
#    st.session_state.generate = False

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


def load_file():
    """Load uploaded file into session state"""
    file = st.session_state['txt_file']
    if file:
        data = file.read().decode("utf-8")
        st.session_state.letter_text = data


def trigger_parameter_selection():
    """Called when user clicks 'Search Citations' button"""
    st.session_state.show_parameters = True
    st.session_state.parameters_confirmed = False


def confirm_parameters():
    """Called when user confirms they want to set parameters"""
    st.session_state.parameters_confirmed = True


def decline_parameters():
    """Called when user declines to set parameters"""
    st.session_state.show_parameters = False
    st.session_state.parameters_confirmed = False
    # Use default parameters and proceed with search
    # Here you would call your RAG backend with default parameters
    st.rerun()


#def generate_clicked():
#    st.session_state.generate = True


def save_parameters():
    """Called when user clicks Generate button"""
    st.session_state.show_parameters = False
    st.session_state.parameters_confirmed = False
    # Parameters are already saved in session_state.search_params
    # Here you would call your RAG backend with the saved parameters
    st.rerun()


# Create columns
editor, info = st.columns(spec=[0.5, 0.5],
                          gap="medium",
                          vertical_alignment="top")

# editor, info = st.columns(spec=[0.5, 0.5],
#                          gap="medium",
#                          vertical_alignment="top",
#                          border=True)

with info:
    st.header("RAG")

    # Always show chat interface first (unless in parameter form)
    if not st.session_state.parameters_confirmed:
        # Normal chat interface
        prompt = st.chat_input("Say something")
        if prompt:
            st.write(f"User has sent the following prompt: {prompt}")

            # Here you would call your RAG backend with the parameters:
            # Example:
            # results = your_rag_function(
            #     prompt=prompt,
            #     letter_text=st.session_state.letter_text,
            #     num_citations=st.session_state.search_params["num_citations"],
            #     court_decision=st.session_state.search_params["court_decision"],
            #     min_accuracy=st.session_state.search_params["min_accuracy"],
            #     additional_requests=st.session_state.search_params["additional_requests"]
            # )

            # Display parameters being used
            st.info(f"Using parameters: {st.session_state.search_params}")

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

    elif st.session_state.parameters_confirmed:
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

        # generate_button = st.button(
        #    label="Generate!",
        #    type="primary",
        #    on_click=generate_clicked
        # )

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

# if st.session_state.generate:
# this part is kept in the state
# generating citations for the text
# st.write("Generating citations for: ", st.session_state.letter_text)