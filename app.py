import gradio as gr
from test_rag_infloat_multilingual import LegalRAGSystem
import pandas as pd
from pathlib import Path
import re
import numpy as np

# Initialize RAG system and data
# rag = LegalRAGSystem()
ecli_df = pd.read_excel("data/DATA ecli_nummers juni 2025 v1 (version 1).xlsx").set_index('ecli_nummer')


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


def generate_citations(letter_text, num_citations, min_accuracy, state):
    """Generate ECLI citations based on the letter text"""
    if not letter_text.strip():
        return "Please enter or upload letter text first.", "", state

    # Get all sorted results
    # state.all_sorted_list = [('NL:CRVB:2021:612', np.float32(0.8029959)), ('NL:RVS:2013:53', np.float32(0.8029759))]
    state.all_sorted_list = rag.get_top_10_for_letter(letter_text, domain="bicycle", train_ids=train_ids)
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
    print("ECLI LIST FOUND: ", ecli_list)

    if not ecli_list:
        return "<p>No citations found.</p>"

    html = "<div style='padding: 10px;'>"
    html += f"<h3>Found {len(state.all_sorted_list)} citations (showing {len(ecli_list)})</h3>"

    print("Entered generate citations")
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
    if file is None:
        return ""

    with open(file, 'r', encoding='utf-8') as f:
        return f.read()


# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# CiteConnect")
    gr.Markdown("### Legal Citation Search System")

    # State management
    state = gr.State(AppState())

    with gr.Column():
        # Upload, Download and Preview buttons
        with gr.Row():
            file_upload = gr.UploadButton(
                label="Upload Letter (.txt)",
                file_types=[".txt"],
                type="filepath",
                file_count="single"
            )

            download_btn = gr.DownloadButton(
                label="Download text",
            )
            preview_pdf_btn = gr.Button(
                "Preview PDF",
                variant="secondary"
            )
            download_pdf_btn = gr.DownloadButton(
                label="Download PDF",
                visible=False
            )

        # PDF Preview - displayed above the editor
        pdf_output = gr.File(
            label="PDF Preview",
            visible=False,
            interactive=False,
            show_label=True
        )

        # Editor
        letter_text = gr.Textbox(
            label="Legal Advice Letter",
            placeholder="Upload a legal advice letter or type directly to begin...",
            max_lines=2000
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Search ECLI Citations")
                with gr.Group():
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

                with gr.Row(visible=False) as action_buttons:
                    regenerate_btn = gr.Button("Regenerate", variant="primary")

                with gr.Group():
                    ecli_input = gr.Textbox(
                        label="ECLI to add to letter",
                        placeholder="Paste ECLI here to add to letter..."
                    )
                    add_ecli_btn = gr.Button("Add to Letter", variant="secondary")

            with gr.Column():
                gr.Markdown("## Display retrieved ECLI Citations")
                # Displays the list of ecli citations found
                results_display = gr.HTML(
                    label="Citation Results",
                    visible=True
                )

    # Event handlers
    file_upload.upload(
        fn=load_file,
        inputs=file_upload,
        outputs=letter_text
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


    # Download button functionality
    def prepare_download(text):
        """Save text to a temporary file for download"""
        if not text.strip():
            return None
        
        # Create a temporary file
        temp_file = Path("temp_letter.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return str(temp_file)
    
    download_btn.click(
        fn=prepare_download,
        inputs=[letter_text],
        outputs=[download_btn]
    )
    
    # Preview PDF functionality
    def generate_pdf_preview(text):
        """Generate PDF for preview"""
        if not text.strip():
            return None, gr.update(visible=False), None, gr.update(visible=False)
        
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.enums import TA_LEFT
            
            # Create PDF with proper Dutch formatting
            pdf_path = Path("juridisch_advies.pdf")
            doc = SimpleDocTemplate(
                str(pdf_path), 
                pagesize=A4,
                leftMargin=2.5*cm,
                rightMargin=2.5*cm,
                topMargin=2.5*cm,
                bottomMargin=2.5*cm
            )
            
            styles = getSampleStyleSheet()
            
            # Create a custom style for better formatting
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                alignment=TA_LEFT,
                spaceAfter=12
            )
            
            story = []
            
            # Split text into paragraphs and add to PDF
            paragraphs = text.split('\n\n')
            for para_text in paragraphs:
                if para_text.strip():
                    # Replace single newlines with <br/> for line breaks within paragraphs
                    formatted_text = para_text.replace('\n', '<br/>')
                    p = Paragraph(formatted_text, normal_style)
                    story.append(p)
                    story.append(Spacer(1, 0.3*cm))
            
            doc.build(story)
            return str(pdf_path), gr.update(visible=True), str(pdf_path), gr.update(visible=True)
            
        except ImportError as e:
            print(f"ReportLab not installed: {e}")
            return None, gr.update(visible=False), None, gr.update(visible=False)
        except Exception as e:
            print(f"Error generating PDF: {e}")
            return None, gr.update(visible=False), None, gr.update(visible=False)
    
    # Function to close/hide the PDF preview
    def close_pdf_preview():
        """Hide the PDF preview and download button"""
        return gr.update(visible=False, value=None), gr.update(visible=False)
    
    preview_pdf_btn.click(
        fn=generate_pdf_preview,
        inputs=[letter_text],
        outputs=[pdf_output, pdf_output, download_pdf_btn, download_pdf_btn]
    )
    
    # When user clicks X on the file preview, hide both preview and download button
    pdf_output.clear(
        fn=close_pdf_preview,
        inputs=[],
        outputs=[pdf_output, download_pdf_btn]
    )


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())