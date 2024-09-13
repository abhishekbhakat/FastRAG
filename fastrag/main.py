import asyncio
from collections.abc import AsyncIterable
from typing import Annotated

from fastapi import File, UploadFile
from fastapi.responses import HTMLResponse
from fastui import AnyComponent, FastUI, components as c, prebuilt_html
from fastui.events import GoToEvent, PageEvent
from fastui.forms import fastui_form

from fastrag.app_config import app
from fastrag.config import logger
from fastrag.default_page import get_default_page
from fastrag.models import ChatForm, MessageHistoryModel, UploadForm, URLForm
from fastrag.routes import ingest, query
from fastrag.services.bootstrap import bootstrap_app

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


@app.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def api_index():
    logger.debug("Handling API index request")
    return get_default_page()


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
