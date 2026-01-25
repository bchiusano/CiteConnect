import gradio as gr
from test_rag_infloat_multilingual import LegalRAGSystem
import pandas as pd
import re

# Initialize RAG system and data
rag = LegalRAGSystem()
ecli_df = pd.read_excel("../data/DATA ecli_nummers juni 2025 v1 (version 1).xlsx").set_index('ecli_nummer')


# Global state (will be managed through Gradio's state components)
class AppState:
    def __init__(self):
        self.all_sorted_list = []
        self.ecli_index_display = 0
        self.search_params = {
            "num_citations": 9,
            "min_accuracy": 75
        }


def fetch_description(number):
    """Fetch ECLI description from dataframe"""
    try:
        return ecli_df.loc[f"ECLI:{number}", 'ecli_tekst']
    except:
        return "Description not available"


def get_case_overview(number, max_sentences=3):
    """Get a brief overview of the case (first few sentences)"""
    try:
        full_text = ecli_df.loc[f"ECLI:{number}", 'ecli_tekst']
        sentences = re.split(r'\.[\s\n]+', str(full_text))
        overview = '. '.join(sentences[:max_sentences])
        if overview and not overview.endswith('.'):
            overview += '.'
        if len(overview) > 500:
            overview = overview[:500] + "..."
        return overview if overview else "Overview not available"
    except:
        return "Overview not available"


def generate_citations(letter_text, num_citations, min_accuracy, state):
    """Generate ECLI citations based on the letter text"""
    if not letter_text.strip():
        return "Please enter or upload letter text first.", "", state

    # Get all sorted results
    state.all_sorted_list = rag.get_top_10_for_letter(letter_text)
    state.ecli_index_display = 0
    state.search_params["num_citations"] = num_citations
    state.search_params["min_accuracy"] = min_accuracy

    # Display results
    results_html = format_citation_results(state)

    return results_html, gr.update(visible=True), state


def format_citation_results(state):
    """Format citation results as HTML"""
    start = state.ecli_index_display
    end = start + state.search_params["num_citations"]
    ecli_list = state.all_sorted_list[start:end]

    if not ecli_list:
        return "<p>No citations found.</p>"

    html = "<div style='padding: 10px;'>"
    html += f"<h3>Found {len(state.all_sorted_list)} citations (showing {len(ecli_list)})</h3>"

    for ecli, relevance_score in ecli_list:
        description = fetch_description(ecli)
        score_pct = relevance_score * 100

        html += f"""
        <details style='margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>
            <summary style='cursor: pointer; font-weight: bold;'>
                {ecli} (Score: {score_pct:.3f}%)
            </summary>
            <div style='padding: 10px; margin-top: 10px; background-color: #f9f9f9;'>
                <p>{description}</p>
                <button onclick='navigator.clipboard.writeText("{ecli}")' 
                        style='padding: 5px 10px; background-color: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 3px;'>
                    Copy ECLI
                </button>
            </div>
        </details>
        """

    html += "</div>"
    return html


def regenerate_citations(state):
    """Show next batch of citations"""
    state.ecli_index_display += state.search_params["num_citations"]

    # Reset to beginning if we've shown all
    next_end_index = state.ecli_index_display + state.search_params["num_citations"]
    if next_end_index >= len(state.all_sorted_list):
        state.ecli_index_display = 0

    results_html = format_citation_results(state)
    return results_html, state


def add_selected_ecli_to_text(letter_text, ecli_to_add):
    """Add selected ECLI to the letter text"""
    if ecli_to_add.strip():
        return letter_text + f"\n{ecli_to_add.strip()}"
    return letter_text


def load_file(file):
    """Load uploaded file"""
    if file is not None:
        content = file.decode('utf-8') if isinstance(file, bytes) else file.read().decode('utf-8')
        return content
    return ""


# Create Gradio interface
with gr.Blocks(title="CiteConnect", theme=gr.themes.Soft()) as app:
    gr.Markdown("# CiteConnect")
    gr.Markdown("### Legal Citation Search System")

    # State management
    state = gr.State(AppState())

    with gr.Row():
        # Left column: Editor
        with gr.Column(scale=1):
            gr.Markdown("## Editor")

            #file_upload = gr.File(
            #    label="Upload Letter (.txt)",
            #    file_types=[".txt"],
            #    type="binary"
            #)

            file_upload = gr.UploadButton(
                label="Upload Letter (.txt)",
                file_types=[".txt"],
                type="binary",
                file_count="single"
            )

            letter_text = gr.Textbox(
                label="Letter Text",
                placeholder="Upload a legal advice letter or type directly to begin...",
                lines=15,
                max_lines=20
            )

            with gr.Row():
                download_btn = gr.DownloadButton(
                    label="Download text",
                    icon="📥"
                )
                preview_btn = gr.Button(
                    "Preview PDF",
                    icon="👁️"
                )

            preview_output = gr.Textbox(
                label="Preview Status",
                visible=False
            )

        # Right column: Search and Results
        with gr.Column(scale=1):
            gr.Markdown("## Search ECLI Citations")

            with gr.Group():
                gr.Markdown("### Search Parameters")
                gr.Markdown("*Be aware of possible bias that may come from added parameters.*")

                num_citations = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=9,
                    step=1,
                    label="Number of total citations"
                )

                min_accuracy = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=75,
                    step=1,
                    label="Minimum Accuracy (%)"
                )

                generate_btn = gr.Button(
                    "Generate Citations",
                    variant="primary",
                    size="lg"
                )

            results_display = gr.HTML(
                label="Citation Results",
                visible=False
            )

            with gr.Row(visible=False) as action_buttons:
                regenerate_btn = gr.Button("Regenerate", variant="primary")

            with gr.Row():
                ecli_input = gr.Textbox(
                    label="ECLI to add to letter",
                    placeholder="Paste ECLI here to add to letter..."
                )
                add_ecli_btn = gr.Button("Add to Letter", variant="secondary")

    # Event handlers
    file_upload.upload(
        fn=load_file,
        inputs=[file_upload],
        outputs=[letter_text]
    )

    generate_btn.click(
        fn=generate_citations,
        inputs=[letter_text, num_citations, min_accuracy, state],
        outputs=[results_display, action_buttons, state]
    )

    regenerate_btn.click(
        fn=regenerate_citations,
        inputs=[state],
        outputs=[results_display, state]
    )

    add_ecli_btn.click(
        fn=add_selected_ecli_to_text,
        inputs=[letter_text, ecli_input],
        outputs=[letter_text]
    )

    preview_btn.click(
        fn=lambda: "Preview functionality to be implemented",
        inputs=[],
        outputs=[preview_output]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[preview_output]
    )

    # Download button functionality
    letter_text.change(
        fn=lambda text: gr.DownloadButton(
            label="Download text",
            value=text,
            visible=bool(text.strip())
        ),
        inputs=[letter_text],
        outputs=[download_btn]
    )

if __name__ == "__main__":
    app.launch()