import asyncio
from collections.abc import AsyncIterable

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastui import FastUI, components as c, prebuilt_html
from fastui.components.display import DisplayLookup, DisplayMode
from fastui.events import GoToEvent, PageEvent
from starlette.responses import StreamingResponse

from fastrag.models import ChatForm, MessageHistoryModel, UploadForm, URLForm
from fastrag.routes import ingest, query
from fastrag.services.bootstrap import bootstrap_app
from fastrag.config import logger

app = FastAPI()
app.message_history = []

app.include_router(ingest.router, prefix="/api")
app.include_router(query.router, prefix="/api")


@app.on_event("startup")
async def startup_event():
    try:
        bootstrap_app()
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # You might want to raise the exception here if you want to prevent the app from starting
        # raise e


@app.get("/api/", response_model=FastUI, response_model_exclude_none=True)
def api_index():
    return [
        c.PageTitle(text="RAG Chatbot"),
        c.Page(
            components=[
                c.Heading(text="RAG Chatbot"),
                c.Paragraph(text="Upload documents, paste URLs, and chat with your data using LLM and vector database."),
                c.Heading(text="Upload Documents", level=2),
                c.ModelForm(
                    model=UploadForm,
                    submit_url="/api/upload_documents",
                    method="POST",
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
            ]
        ),
        c.Footer(extra_text="RAG Chatbot powered by FastUI", links=[]),
    ]


@app.post("/api/upload_documents")
async def upload_documents(file: UploadFile = File(...)):
    result = await ingest.ingest_document(file)
    if result["status"] == "success":
        return [
            c.Toast(
                title="Document Uploaded",
                body=[c.Paragraph(text=f"Successfully processed: {file.filename}")],
                open_trigger=PageEvent(name="show-toast"),
                position="bottom-end",
            ),
            c.FireEvent(event=PageEvent(name="show-toast")),
        ]
    else:
        return [
            c.Toast(
                title="Document Upload Failed",
                body=[c.Paragraph(text=f"Failed to process: {file.filename}")],
                open_trigger=PageEvent(name="show-toast"),
                position="bottom-end",
            ),
            c.FireEvent(event=PageEvent(name="show-toast")),
        ]


@app.post("/api/add_url")
async def add_url(url_form: URLForm):
    return [
        c.Page(components=[c.Heading(text="URL Added"), c.Paragraph(text=f"Successfully added URL: {url_form.url}"), c.Link(components=[c.Text(text="Back to Home")], on_click=GoToEvent(url="/"))])
    ]


@app.post("/api/chat")
async def chat(chat_form: ChatForm):
    return StreamingResponse(chat_response_generator(chat_form.message), media_type="text/event-stream")


async def chat_response_generator(message: str) -> AsyncIterable[str]:
    response = await query.query(message)
    app.message_history.append(MessageHistoryModel(message=f"User: {message}"))
    app.message_history.append(MessageHistoryModel(message=f"Chatbot: {response['response']}"))
    m = FastUI(root=[c.Markdown(text=response['response'])])
    msg = f"data: {m.model_dump_json(by_alias=True, exclude_none=True)}\n\n"
    yield msg
    while True:
        yield msg
        await asyncio.sleep(10)


@app.get("/{path:path}")
async def html_landing() -> HTMLResponse:
    return HTMLResponse(prebuilt_html(title="SuperMemPy"))
