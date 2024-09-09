# FastRAG: FastAPI and FastUI Web Application for RAG

FastRAG is a powerful and user-friendly web application that combines FastAPI and FastUI to provide a seamless experience for Retrieval-Augmented Generation (RAG). It offers an easy way to import, process, and query various data sources using a vector store backed by PostgreSQL.

## Features

- **Easy Data Import**: Directly import data from GitHub repositories, URLs, and various file types through the web UI.
- **Vector Store Integration**: Utilizes PostgreSQL as a vector store for efficient similarity searches.
- **FastAPI Backend**: Robust and high-performance API built with FastAPI.
- **FastUI Frontend**: Intuitive and responsive user interface powered by FastUI.
- **Docker Support**: Simple deployment using Docker for consistency across environments.
- **RAG Capabilities**: Perform advanced queries on your imported data using state-of-the-art RAG techniques.

## Prerequisites

- Docker and Docker Compose
- Git (for cloning the repository)

## Quick Start

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fastrag.git
   cd fastrag
   ```

2. Build and run the Docker containers:
   ```
   docker-compose up --build
   ```

3. Access the application:
   Open your web browser and navigate to `http://localhost:3000`

## Usage

1. **Importing Data**:
   - Use the web UI to import data from GitHub repositories, URLs, or by uploading files directly.
   - Supported file types include PDF, TXT, MD, and more.

2. **Querying**:
   - Once data is imported, use the query interface to ask questions or retrieve information.
   - The RAG system will process your query and return relevant results from the imported data.

3. **Managing Data**:
   - View, update, or delete imported data sources through the web interface.

## Configuration

- Environment variables can be set in the `.env` file or passed to Docker.
- Key configurations:
  - `DATABASE_URL`: PostgreSQL connection string
  - `OPENAI_API_KEY`: Your OpenAI API key for RAG functionality

## Development

To set up a development environment:

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI server:
   ```
   uvicorn fastrag.main:app --reload --host 0.0.0.0 --port 3000
   ```

3. For frontend development, refer to the FastUI documentation for setup instructions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- FastAPI
- FastUI
- PostgreSQL
- OpenAI (for RAG capabilities)

For more information or support, please open an issue on the GitHub repository.
