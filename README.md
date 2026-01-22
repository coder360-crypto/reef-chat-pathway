# REEFChat

![ReefChat Demo](<./gifs/01%20Reef%20(Transparent%20Background).gif>)

This repository contains a Multi-Agent Dynamic RAG system implemented in 5 folders as shown below for the Inter IIT Tech Meet 13.0

```
REEFChat/
├── agents/
│   ├── main.py                      # agent orchestration
│   ├── services/
│   │   ├── carp_services.py
│   │   ├── squid_services.py
│   │   ├── multihop_service.py
│   │   └── moray_services.py
│   └── utils/
│       ├── metadata_generator.py
│       │   ...
│       ├── code_architect_utils/    # carp_utils
│       ├── squid_utils/
│       ├── moray_utils/
│       ├── equity_generation_utils/
│       │   ...                      # other use case utils
│       └── tools/
│           └── ...                  # all tool implementations
│
├── embedder/
│   └── modal_service.py             # modal-based embedding service

├── indexer/
│   ├── services/
│   │   ├── indexer.py               # main indexing service
│   │   └── credentials.json         # gdrive credentials
│   └── utils/
│       ├── connector.py             # data source connectors
│       └── index_utils.py           # indexing utilities
│
└── interface/
│   ├── main.py                      # fastapi endpoint
│   └── services/
│       └── conversation_handler.py  # chat memory
│
└── guardrails/
    └── guard.py
```

**Key Components:**

- **embedder**: Handles document embedding using STELLA-1.5B model via Modal
- **agents**: Provides the three pipelines:
  - CARP (**C**ode **AR**chitect for **P**lanning)
  - SQUID (**S**elf-critical **Q**uery **U**nderstanding via **I**ntelligent **D**elegation)
  - MORAY (**M**ulti-agent **O**rchestrated **R**etrieval and D**A**G s**Y**ntesis)
- **indexer**: Handles document indexing and Google Drive connectivity
- **interface**: Provides the FastAPI web interface
- **guardrails**: Provides robust gaurdrailing to the entire system

## Environment Setup

Set the environment variables in the `.env` file of each microservice. Refer to the `.env.dev` file in each microservice directory for examples.

- For the `GDRIVE_LINK`, enter only the Object ID (End of the folder link). Refer to Pathway's developer docs for further details.
- For `PORT`, `EMBEDDING_API_URL` and `RETRIEVER_API_URL` fields, refer to the Docker compose files for a valid configuration. Change only if absolutely necessary.

Create a `credentials.json` file using Google Cloud. This is required for gaining access to the GDRive Data connector (refer to Pathway's developer docs).

- Place the file in the `indexer/services` directory.

## Running with Docker

1. Ensure docker daemon is running:

   ```bash
   docker info
   ```

2. Build the docker images:

   ```bash
   docker build -t <service-name> .
   ```

   For embedder service, create a Modal Token ID and Token Secret on https://modal.com/ and then run:

   ```bash
   docker build --build-arg TOKEN_ID=your_token_id --build-arg TOKEN_SECRET=your_token_secret -t your_image_name .
   ```

   > **Caution:** When running docker build, it may take upto 15 minutes to download the STELLA-1.5B model for embeddings.

3. Start the services:
   ```bash
   docker compose up
   ```

## Running Locally with Poetry

### Prerequisites

- Install poetry using pipx: [Poetry Installation Guide](https://python-poetry.org/docs/#installation)

### Troubleshooting

> **Caution:** macOS may not work as expected with some services.

If you encounter issues with DBus and keyring:

```bash
sudo apt-get install python3-dbus python3-secretstorage python3-keyring
```

If you encounter issues with system site packages:

1. Locate your virtual environment's `pyvenv.cfg` file
2. Set `include-system-site-packages = true`

> **Caution:** In case of errors with versions, please use the following command in each folder.

```bash
#install python3.11
poetry env use python3.11
```

### Embedder Service

1. Install dependencies:
   ```bash
   cd embedder
   poetry install
   ```
2. Create your modal token and authorize your modal account
   ```bash
   poetry run modal token new
   ```
3. Start the service (Note the change in the command):

   ```bash
   poetry run modal run modal_service.py
   ```

   > **Caution:** Please ensure the STELLA-1.5B model is downloaded onto your Modal account before running this file. There are instructions to do the same in the file `modal_service.py`

### Indexer Service

> Note: Ensure embedder service is running first

1. Install dependencies:
   ```bash
   cd indexer
   poetry install
   ```
2. Start the service (Note the change in the command):
   ```bash
   poetry run python services/indexer.py
   ```

### Agents Service

> Note: Ensure embedder and indexer services are running (either on cloud or locally)

1. Install dependencies:
   ```bash
   cd agents
   poetry install
   ```
2. Start the service:
   ```bash
   poetry run python main.py
   ```

### Interface

1. Install dependencies:
   ```bash
   cd pipeline
   poetry install
   ```
2. Start the service:
   ```bash
   poetry run python main.py
   ```

### Endpoint for Backend

If every service is run without errors, a fastapi endpoint will be available for backend interaction.
