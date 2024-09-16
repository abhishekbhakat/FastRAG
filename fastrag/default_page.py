from fastui import AnyComponent, components as c
from fastui.events import GoToEvent, PageEvent

from fastrag.models import UploadForm, URLForm


def get_default_page() -> list[AnyComponent]:
    return [
        c.PageTitle(text="FastRAG"),
        c.Navbar(
            title="",
            title_event=GoToEvent(url="/"),
            start_links=[
                c.Link(components=[c.Text(text="Home")], on_click=GoToEvent(url="/"), active="startswith:/", mode="navbar"),
                c.Link(components=[c.Text(text="Data")], on_click=GoToEvent(url="/data"), active="startswith:/data", mode="navbar"),
            ],
        ),
        c.Page(components=get_page_components()),
        c.Footer(extra_text="RAG Chatbot powered by FastUI", links=[]),
    ]


def get_page_components() -> list[AnyComponent]:
    return [
        c.Div(
            components=[
                c.Heading(text="FastRAG", level=1),
                c.Paragraph(text="Upload documents, paste URLs, and chat with your data using LLM and vector database."),
            ]
        ),
        c.Div(
            components=[
                c.Heading(text="Upload Documents", level=2),
                c.ModelForm(
                    model=UploadForm,
                    submit_url="/api/upload_documents",
                    loading=[c.Spinner(text="Uploading ...")],
                    submit_trigger=PageEvent(name="upload_documents"),
                ),
            ],
            class_name="border-top mt-3 pt-1",
        ),
        c.Div(
            components=[
                c.Heading(text="Add URLs", level=2),
                c.ModelForm(model=URLForm, submit_url="/api/add_url", submit_trigger=PageEvent(name="add_url")),
            ],
            class_name="border-top mt-3 pt-1",
        ),
        c.Toast(title="Document Uploaded", body=[c.Paragraph(text="Successfully processed the document.")], open_trigger=PageEvent(name="document-upload-success"), position="bottom-center"),
        c.Toast(title="Document Upload Failed", body=[c.Paragraph(text="Failed to process document.")], open_trigger=PageEvent(name="docuent-upload-failed"), position="bottom-center"),
        c.Toast(title="Add URL Success", body=[c.Paragraph(text="Successfully added URL.")], open_trigger=PageEvent(name="add_url_success"), position="bottom-center"),
        c.Toast(title="Add URL Failed", body=[c.Paragraph(text="Failed to add URL.")], open_trigger=PageEvent(name="add_url_failed"), position="bottom-center"),
    ]
