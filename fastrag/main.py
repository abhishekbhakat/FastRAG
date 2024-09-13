import asyncio
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastui import AnyComponent, FastUI, components as c, prebuilt_html
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import GoToEvent, PageEvent
from fastui.forms import fastui_form

from fastrag.config import logger
from fastrag.models import ChatForm, MessageHistoryModel, UploadForm, URLForm
from fastrag.routes import ingest, query
from fastrag.services.bootstrap import bootstrap_app

logger.info("Initializing FastAPI application")
app = FastAPI()
app.message_history = []

logger.info("Including routers")
app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application")
    try:
        bootstrap_app()
        logger.info("Application bootstrapped successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        # You might want to raise the exception here if you want to prevent the app from starting
        # raise e


def get_page_components() -> list[AnyComponent]:
    logger.debug("Getting page components")
    return [
        c.Heading(text="FastRAG", level=1),
        c.Paragraph(text="Upload documents, paste URLs, and chat with your data using LLM and vector database."),
        c.Heading(text="Upload Documents", level=2),
        c.ModelForm(
            model=UploadForm,
            submit_url="/api/upload_documents",
            loading=[c.Spinner(text="Uploading ...")],
            submit_trigger=PageEvent(name="upload_documents"),
        ),
        c.Heading(text="Add URLs", level=2),
        c.ModelForm(model=URLForm, submit_url="/api/add_url", method="POST"),
        c.Heading(text="Chat", level=2),
        c.Table(
            data=app.message_history,
            data_model=MessageHistoryModel,
            columns=[DisplayLookup(field="message", mode=DisplayMode.markdown, table_width_percent=100)],
            no_data_message="No messages yet.",
        ),
        c.ModelForm(model=ChatForm, submit_url="/api/chat", method="POST"),
        c.Div(
            components=[
                c.ServerLoad(
                    path="/api/sse/chat",
                    sse=True,
                    load_trigger=PageEvent(name="load"),
                    components=[],
                ),
            ],
            class_name="my-2 p-2 border rounded",
        ),
        c.Link(
            components=[c.Text(text="Reset Chat")],
            on_click=GoToEvent(url="/?reset=true"),
        ),
        c.Toast(title="Document Uploaded", body=[c.Paragraph(text="Successfully processed the document.")], open_trigger=PageEvent(name="document-upload-success"), position="bottom-center"),
        c.Toast(title="Document Upload Failed", body=[c.Paragraph(text="Failed to process document.")], open_trigger=PageEvent(name="docuent-upload-failed"), position="bottom-center"),
    ]


@app.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def api_index():
    logger.debug("Handling API index request")
    return [
        c.PageTitle(text="RAG Chatbot"),
        c.Page(
            components=get_page_components(),
        ),
        c.Footer(extra_text="RAG Chatbot powered by FastUI", links=[]),
    ]


@app.post("/api/upload_documents", response_model=FastUI, response_model_exclude_none=True)
async def upload_documents(file: UploadFile = File(...)) -> list[AnyComponent]:
    # result = await ingest.ingest_document(file)
    result = {"status": "success"}
    if result["status"] == "success":
        return [
            c.FireEvent(event=PageEvent(name="document-upload-success")),
            c.ModelForm(
                model=UploadForm,
                submit_url="/api/upload_documents",
                loading=[c.Spinner(text="Uploading ...")],
                submit_trigger=PageEvent(name="upload_documents"),
            ),
        ]
    else:
        return [
            c.FireEvent(event=PageEvent(name="document-upload-failed")),
            c.ModelForm(
                model=UploadForm,
                submit_url="/api/upload_documents",
                loading=[c.Spinner(text="Uploading ...")],
                submit_trigger=PageEvent(name="upload_documents"),
            ),
        ]


@app.post("/api/add_url")
async def add_url(url_form: URLForm):
    return [
        c.Page(components=[c.Heading(text="URL Added"), c.Paragraph(text=f"Successfully added URL: {url_form.url}"), c.Link(components=[c.Text(text="Back to Home")], on_click=GoToEvent(url="/"))])
    ]


@app.post("/api/chat", response_model=FastUI, response_model_exclude_none=True)
async def chat(chat_form: Annotated[ChatForm, fastui_form(ChatForm)]):
    response = await query.query(chat_form.message)
    app.message_history.append(MessageHistoryModel(message=f"User: {chat_form.message}"))
    app.message_history.append(MessageHistoryModel(message=f"Chatbot: {response['response']}"))
    return [c.Markdown(text=response["response"]), c.FireEvent(event=GoToEvent(url="/"))]


async def chat_response_generator(message: str) -> AsyncIterable[str]:
    response = await query.query(message)
    app.message_history.append(MessageHistoryModel(message=f"User: {message}"))
    app.message_history.append(MessageHistoryModel(message=f"Chatbot: {response['response']}"))
    m = FastUI(root=[c.Markdown(text=response["response"])])
    msg = f"data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
    yield msg
    while True:
        yield msg
        await asyncio.sleep(10)


@app.get("/{path:path}")
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title="FastRAG"))
