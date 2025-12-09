VCO SD-WAN LLM Assistant

A simple chatbot that can answer questions about your VeloCloud / Arista SD-WAN environment by calling the VCO API through a Python backend.
The chatbot uses AWS Bedrock, PydanticAI, FastAPI, FastMCP, and Streamlit.

This project is designed as an educational example showing how to build an LLM-powered network assistant with real API tool-calling.

🚀 Features

Ask natural-language questions about SD-WAN health

List edges, check edge health, basic enterprise status

Uses AWS Bedrock (Claude) for LLM reasoning

Clean Python backend with VCO client + typed models

Optional Streamlit chat UI

Optional MCP server for tool access

🧱 Tech Stack

Python 3.11+

Pydantic & PydanticAI

FastAPI (backend API)

FastMCP (tool server)

Streamlit (chat frontend)

boto3 (AWS Bedrock Runtime)

📁 Project Structure
backend/
  ai/agent.py        # LLM agent + Bedrock connection + tools
  vco/vco_client.py  # Wrapper around VCO API
  api/app.py         # FastAPI REST endpoints
  mcp/server.py      # FastMCP tool server
frontend/
  streamlit_app.py   # Simple chat UI
README.md
requirements.txt or uv.toml
.env.example

🔧 Setup
1. Install dependencies

Using uv:

uv sync


or pip:

pip install -r requirements.txt

2. Configure environment variables

Copy .env.example → .env and fill in:

VCO_BASE_URL=
VCO_API_TOKEN=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=
BEDROCK_MODEL_ID=


(Optional LASSO proxy settings if needed.)

▶️ Run the project
Run the backend API
uvicorn backend.api.app:app --reload

Run the Streamlit UI
streamlit run frontend/streamlit_app.py

Run the MCP server (optional)
python -m backend.mcp.server

💬 Example Usage

After starting Streamlit or FastAPI, you can ask things like:

“List my edge devices”

“Show health for edge Dallas-01 for the last hour”

“Give me a quick SD-WAN summary”

The LLM will automatically call tools that talk to the VCO API.
