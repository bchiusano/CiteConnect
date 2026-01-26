import streamlit as st
from test_rag_infloat_multilingual import LegalRAGSystem
import pandas as pd
import re

st.set_page_config(layout="wide")
st.title("CiteConnect")


# Initialize RAG class
@st.cache_resource
def get_rag_instance():
    return LegalRAGSystem()


@st.cache_resource
def get_train_ids():
    return rag.prepare_train_ids_for_citation_db()


@st.cache_resource
def get_ecli_data():
    df = pd.read_excel("../data/DATA ecli_nummers juni 2025 v1 (version 1).xlsx")
    return df.set_index('ecli_nummer')


@st.cache_resource
def load_citations():
    rag.init_citation_db(train_ids, False)


# Get the cached instance
rag = get_rag_instance()
train_ids = get_train_ids()
load_citations()
ecli_df = get_ecli_data()

# Initialize session state variables
if "letter_text" not in st.session_state:
    st.session_state.letter_text = ""

if "show_parameters" not in st.session_state:
    st.session_state.show_parameters = True

# if "parameters_confirmed" not in st.session_state:
#    st.session_state.parameters_confirmed = False

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

if "all_sorted_list" not in st.session_state:
    st.session_state.all_sorted_list = []

if "ecli_index_display" not in st.session_state:
    st.session_state.ecli_index_display = 0

if 'selected_ecli' not in st.session_state:
    st.session_state.selected_ecli = {}

# NEW: Chat/Search interface state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "search_results" not in st.session_state:
    st.session_state.search_results = {}

if "chat_selected_ecli" not in st.session_state:
    st.session_state.chat_selected_ecli = {}


# Check if editor has content (from file OR manual typing)
has_content = st.session_state.letter_text.strip() != ""


def load_file():
    """Load uploaded file into session state"""
    file = st.session_state['txt_file']
    if file:
        data = file.read().decode("utf-8")
        st.session_state.letter_text = data


def trigger_parameter_selection():
    """Called when user clicks 'Search Citations' button"""
    st.session_state.show_parameters = True


def save_parameters():
    """Called when user clicks Generate button"""
    st.session_state.show_parameters = False
    # st.session_state.parameters_confirmed = False

    # resetting this to 0 if the parameters where changed and saved
    st.session_state.ecli_index_display = 0

    with st.spinner("Finding the most relevant ECLI citations for your letter..."):
        st.session_state.all_sorted_list = rag.get_top_10_for_letter(st.session_state.letter_text, domain="bicycle", train_ids=train_ids)

    print("FULL List: ", st.session_state.all_sorted_list)
    show_sorted_citations()

    st.rerun()


def show_sorted_citations():

    start = st.session_state.ecli_index_display
    end = start + st.session_state.search_params["num_citations"]

    st.session_state.ecli_list = st.session_state.all_sorted_list[start:end]

    print(st.session_state.ecli_list)
    print("HOW MANY ECLI's: ", len(st.session_state.all_sorted_list))


def accept_ecli_selection():
    """Add selected ECLIs from citation results to editor"""
    selected = [ref for ref, choice in st.session_state.selected_ecli.items() if choice == True]

    if selected:
        st.session_state.letter_text += f"\n{', '.join(selected)}"


def accept_chat_ecli_selection():
    """Add selected ECLIs from chat search to editor"""
    selected = [ref for ref, choice in st.session_state.chat_selected_ecli.items() if choice == True]
    
    if selected:
        st.session_state.letter_text += f"\n{', '.join(selected)}"
        # Clear selections after adding
        st.session_state.chat_selected_ecli = {}


def search_database(query):
    """
    Perform RAG search and return top 5 results in conversational format
    """
    # Use the RAG system to get top results, but limit to 5
    all_results = rag.get_top_10_for_letter(query, domain="bicycle", train_ids=train_ids)
    top_eclis = all_results[:5]  # Limit to 5
    
    # Store results for this query
    st.session_state.search_results[query] = top_eclis
    
    return top_eclis


def generate_summary_paragraph(ecli_list):
    """
    Generate a summary paragraph describing the overall themes of the retrieved cases
    """
    if not ecli_list:
        return ""
    
    # Collect overviews from all cases
    case_overviews = []
    for ecli in ecli_list:
        try:
            overview = get_case_overview(ecli, max_sentences=2)
            case_overviews.append(overview)
        except:
            continue
    
    if not case_overviews:
        return ""
    
    # Analyze common themes (simple keyword extraction)
    all_text = " ".join(case_overviews).lower()
    
    # Common legal terms to identify themes
    themes = []
    if "fiets" in all_text or "bicycle" in all_text:
        themes.append("bicycle-related matters")
    if "verwijder" in all_text or "wegslepen" in all_text or "removal" in all_text:
        themes.append("vehicle removal/towing")
    if "kennisgeving" in all_text or "notification" in all_text or "notificatie" in all_text:
        themes.append("notification procedures")
    if "bezwaar" in all_text or "objection" in all_text:
        themes.append("objection proceedings")
    if "gemeente" in all_text or "municipality" in all_text:
        themes.append("municipal authority decisions")
    if "kosten" in all_text or "cost" in all_text:
        themes.append("cost recovery")
    if "eigenaar" in all_text or "owner" in all_text:
        themes.append("ownership rights")
    
    # Build summary paragraph
    if themes:
        theme_text = ", ".join(themes[:-1]) + f" and {themes[-1]}" if len(themes) > 1 else themes[0]
        summary = f"**Summary of Retrieved Cases:**\n\nThe {len(ecli_list)} cases found primarily address {theme_text}. "
        summary += f"These decisions provide legal precedent and guidance on similar situations, "
        summary += f"with rulings from various Dutch courts including administrative and civil courts."
    else:
        summary = f"**Summary of Retrieved Cases:**\n\nFound {len(ecli_list)} relevant legal decisions. "
        summary += "These cases provide judicial perspectives on matters related to your query."
    
    return summary


def fetch_description(number):
    """Fetch ECLI description from dataframe"""
    try:
        return ecli_df.loc[f"ECLI:{number}", 'ecli_tekst']
    except:
        return "Description not available"


def get_snippet(text, max_length=300):
    """Get a snippet from the beginning of text"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def get_case_overview(number, max_sentences=3):
    """Get a brief overview of the case (first few sentences)"""
    try:
        full_text = ecli_df.loc[f"ECLI:{number}", 'ecli_tekst']
        
        # Split into sentences (basic split on periods followed by space/newline)
        sentences = re.split(r'\.[\s\n]+', str(full_text))
        
        # Take first N sentences and rejoin
        overview = '. '.join(sentences[:max_sentences])
        
        # Add period if not already there
        if overview and not overview.endswith('.'):
            overview += '.'
            
        # If still too long, truncate to reasonable length
        if len(overview) > 500:
            overview = overview[:500] + "..."
            
        return overview if overview else "Overview not available"
    except:
        return "Overview not available"


# Create columns
editor, info = st.columns(spec=[0.5, 0.5],
                          gap="medium",
                          vertical_alignment="top",
                          border=True)


with info:
    st.header("Search ECLI Citations")

    if st.session_state.show_parameters:
        # Show parameter form
        st.subheader("Select parameters for search:")
        st.caption("Be aware of possible bias that may come from added parameters.")

        # Number of citations
        st.session_state.search_params["num_citations"] = st.number_input(
            "Number of total citations*",
            min_value=1,
            max_value=50,
            value=st.session_state.search_params["num_citations"],
            step=1
        )

        # Minimum Accuracy
        st.session_state.search_params["min_accuracy"] = st.number_input(
            "Minimum Accuracy (0%-100%)*",
            min_value=0,
            max_value=100,
            value=st.session_state.search_params["min_accuracy"],
            step=1
        )

        # Generate button
        if st.button("Generate", type="primary", use_container_width=True, disabled=not has_content):
            save_parameters()

    # Display citation results
    elif st.session_state.ecli_list:
        ecli_list = st.session_state.ecli_list
        descriptions = {}

        for ecli, relevance_score in ecli_list:
            # fetching the descriptions of the ecli from data
            descriptions[ecli] = fetch_description(ecli)

            # select functionality for session state
            if ecli not in st.session_state.selected_ecli:
                st.session_state.selected_ecli[ecli] = False

            ecli_name, buttons = st.columns([8, 1])

            with ecli_name:
                # user can expand to see the description of the ecli
                with st.expander(
                        f"{'✅' if st.session_state.selected_ecli[ecli] == True else '❌'} {ecli} (**Score**: {relevance_score*100:.3f}%)"):
                    if descriptions[ecli]:
                        st.write(descriptions[ecli])

            with buttons:
                if st.button("✓", key=f"yes_{ecli}"):
                    st.session_state.selected_ecli[ecli] = True
                    st.rerun()

        change_params, regenerate, accept = st.columns(3)
        with change_params:
            if st.button("Update Parameters", type='primary'):
                # Clear results and go back to parameter selection
                st.session_state.ecli_list = []
                st.session_state.selected_ecli = {}
                st.session_state.show_parameters = True
                # st.session_state.parameters_confirmed = True
                st.rerun()

        with regenerate:
            if st.button("Regenerate", type='primary'):
                # Re-run the search with same parameters
                # Shows the other ecli's found in the sorted list
                # Increments the index of the start of the list to display
                st.session_state.ecli_index_display += st.session_state.search_params["num_citations"]

                # bug check if regenerate was pressed too many times and all ecli's have been viewed
                next_end_index = st.session_state.ecli_index_display + st.session_state.search_params["num_citations"]
                if next_end_index >= len(st.session_state.all_sorted_list):
                    # resetting this to 0
                    st.session_state.ecli_index_display = 0

                show_sorted_citations()
                st.rerun()

        with accept:
            if st.button("Accept", type='primary', on_click=accept_ecli_selection):
                # Clear citation mode
                st.session_state.show_parameters = True
                st.session_state.ecli_list = []
                st.session_state.selected_ecli = {}
                st.rerun()


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


    # Buttons row
    download_col, preview_col = st.columns(2, vertical_alignment="bottom")

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
