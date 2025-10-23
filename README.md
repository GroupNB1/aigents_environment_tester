# 🦜 LangChain LangGraph Ollama Demo API

This project provides a comprehensive FastAPI backend that demonstrates the integration of **LangChain**, **LangGraph**, and **Ollama** with multiple LLM providers. It showcases modern LLM application development patterns including multi-provider support, workflow orchestration, and local model deployment.

## ✨ Features

- **Multi-Provider Support**: OpenAI, Anthropic Claude, Google Gemini, and Ollama
- **LangChain Integration**: Unified interface for different LLM providers
- **LangGraph Workflows**: Multi-step research workflows with state management
- **Local LLM Support**: Run models locally with Ollama
- **Interactive Demo**: Streamlit web interface for testing all features
- **Comprehensive Testing**: Full test suite with mocking support
- **Docker Support**: Complete containerized deployment with Docker Compose
- **DevContainer Ready**: VS Code development container with all tools pre-configured

## 🔧 Supported Providers

- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude-3.5-sonnet, Claude-3-sonnet, Claude-3-haiku
- **Google Gemini**: Gemini-1.5-pro, Gemini-1.5-flash, Gemini-pro
- **Ollama**: Any locally hosted model (Llama2, Mistral, CodeLlama, etc.)
- **Legacy Support**: Grok and custom REST endpoints

## Project Structure

```
aigents_environment_tester/
├── main.py                     # Main FastAPI application with LangChain/LangGraph
├── demo_app.py                 # Streamlit demo interface
├── test_main.py               # Comprehensive test suite
├── requirements.txt            # Project dependencies
├── .env.example               # Template for environment variables
├── Dockerfile                 # Multi-service Docker configuration
├── docker-compose.yml         # Full stack with Ollama, API, and Streamlit
├── .devcontainer/             # VS Code dev container configuration
│   ├── devcontainer.json      # Dev container settings
│   └── setup.sh              # Post-create setup script
├── search_the_internet_and_summarize.ipynb  # Jupyter notebook example
├── app/                       # Legacy app structure (for reference)
└── README.md                  # This documentation
```

## Setup Instructions

### Prerequisites Checklist
Before starting, ensure you have:
- [ ] Python 3.11+ installed (`python --version`)
- [ ] Git installed 
- [ ] At least 4GB free RAM (for local models)
- [ ] Internet connection (for package installation and model downloads)
- [ ] Optional: API keys for external providers (OpenAI, Anthropic, Google)

### 1. Clone the repository
```bash
git clone <repository-url>
cd aigents_environment_tester
```

### 2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

**Note**: If you encounter import errors, upgrade pip first:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up environment variables
Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:
```bash
# OpenAI (required for OpenAI provider)
OPENAI_API_KEY=sk-your_openai_api_key_here

# Anthropic (required for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google (required for Gemini models)
GOOGLE_API_KEY=your_google_api_key_here

# Ollama (local models - configured automatically)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:7b-chat

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=aigents_demo
```

### 5. Install and configure Ollama

#### Install Ollama server
```bash
# Install Ollama (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# For Windows, download from: https://ollama.ai/download
```

#### Start Ollama server
```bash
# Check if Ollama is already running
curl -s http://localhost:11434/api/tags

# If not running, start it
ollama serve
```

**Troubleshooting Ollama Installation:**
- If you get "ollama: command not found", restart your terminal after installation
- If you get "address already in use", Ollama is already running (which is good!)
- Check if Ollama is responding: `curl http://localhost:11434/api/tags`

#### Pull models (required for local LLM support)
```bash
# Essential models for the demo
ollama pull llama2:7b-chat        # General chat model
ollama pull mistral:7b-instruct   # Alternative chat model
ollama pull codellama:7b-instruct # Code generation model

# Verify models are installed
ollama list
```

**Model Selection Tips:**
- Start with `llama2:7b-chat` (smaller, faster)
- Use `mistral:7b-instruct` for better performance
- Try `codellama:7b-instruct` for coding tasks
- Avoid 13B/70B models unless you have 16GB+ RAM

### 6. Verify your setup
```bash
# Run the verification script
python verify_setup.py

# This will check:
# ✅ All required files exist
# ✅ Python packages are installed
# ✅ Ollama server is running
# ✅ Models are available
# ✅ Environment variables are set
```

**Expected output:**
```
🔍 LangChain LangGraph Ollama Demo - System Verification
============================================================
📁 File Structure:
✅ Main API file: main.py
✅ Streamlit demo: demo_app.py
...
🎉 Everything looks good! Your setup is ready.
```

### 7. Run the application

#### Option A: Direct Python execution
```bash
# Start the API server
python main.py

# In another terminal, start the Streamlit demo
streamlit run demo_app.py
```

#### Option B: Using Docker Compose (Recommended)
```bash
# Start all services (API, Ollama, Streamlit)
docker-compose up --build

# This will:
# - Start Ollama server on port 11434
# - Start FastAPI server on port 8000  
# - Start Streamlit demo on port 8501
# - Automatically pull Ollama models
```

### 7. Access the applications
- **Streamlit Demo**: `http://localhost:8501` (Interactive UI)
- **FastAPI Server**: `http://localhost:8000` (API endpoints)
- **API Documentation**: `http://localhost:8000/docs` (Swagger UI)
- **Ollama Server**: `http://localhost:11434` (Model server)
- **Health Check**: `http://localhost:8000/health`

## 🚀 Quick Start with Streamlit Demo

The fastest way to explore all features is through the Streamlit demo:

1. Start the services: `docker-compose up --build`
2. Open your browser to `http://localhost:8501`
3. Try the different tabs:
   - **Generate**: Single text generation with any provider
   - **Chat**: Multi-turn conversations
   - **Workflow**: LangGraph research workflows
   - **Models**: Browse available models

## 📖 API Usage

### 1. Generate Text Endpoint

**POST** `/generate`

#### Request Body:
```json
{
  "provider": "ollama",
  "prompt": "Explain quantum computing in simple terms",
  "model": "llama2:7b-chat",
  "max_tokens": 512,
  "temperature": 0.7,
  "system_message": "You are a helpful physics tutor."
}
```

#### Parameters:
- `provider` (required): One of `openai`, `anthropic`, `gemini`, `ollama`
- `prompt` (required): The text prompt to send to the model
- `model` (optional): Specific model name (uses provider defaults if not specified)
- `max_tokens` (optional): Maximum tokens to generate (default: 512)
- `temperature` (optional): Sampling temperature (default: 0.0)
- `system_message` (optional): System prompt to guide the model's behavior

#### Response:
```json
{
  "provider": "ollama",
  "model": "llama2:7b-chat", 
  "text": "Quantum computing is a revolutionary computing paradigm...",
  "meta": {
    "provider": "ollama",
    "langchain": true,
    "timestamp": "2025-10-23T10:30:00"
  }
}
```

### 2. Chat Endpoint

**POST** `/chat`

Multi-turn conversation support with context.

#### Request Body:
```json
{
  "provider": "ollama",
  "messages": [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is..."},
    {"role": "user", "content": "Can you give me an example?"}
  ],
  "model": "llama2:7b-chat",
  "temperature": 0.7
}
```

### 3. LangGraph Workflow Endpoint

**POST** `/workflow`

Run multi-step research workflows powered by LangGraph.

#### Request Body:
```json
{
  "task_type": "research",
  "input_text": "Latest developments in artificial intelligence",
  "provider": "ollama",
  "model": "llama2:7b-chat"
}
```

#### Response:
```json
{
  "workflow_type": "research",
  "input": "Latest developments in artificial intelligence",
  "provider": "ollama",
  "steps_completed": ["research", "analysis", "synthesis"],
  "result": "Comprehensive research report...",
  "meta": {
    "research_findings": "Initial research results...",
    "analysis": "Detailed analysis..."
  }
}
```

### 4. Models Endpoint

**GET** `/models/{provider}`

List available models for each provider.

```bash
curl http://localhost:8000/models/ollama
curl http://localhost:8000/models/openai
```

### Example Usage with curl:

```bash
# OpenAI example
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "prompt": "Write a haiku about programming",
    "temperature": 0.8
  }'

# Custom provider example
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "custom",
    "prompt": "Hello world",
    "extra": {
      "url": "https://api.example.com/generate",
      "headers": {"Authorization": "Bearer your-token"},
      "body_template": {
        "input": "{prompt}",
        "parameters": {"temperature": "{temperature}"}
      }
    }
  }'
```

## Provider-Specific Notes

### OpenAI
- Requires `OPENAI_API_KEY` environment variable
- Default model: `gpt-4o-mini`
- Supports all OpenAI chat completion parameters via `extra`

### Anthropic  
- Requires `ANTHROPIC_API_KEY` environment variable
- Default model: `claude-2.1`
- Uses legacy completion format with Human/Assistant prompts

### Google Gemini
- Requires `GOOGLE_API_KEY` environment variable
- Requires `google-generativeai` package
- Default model: `models/text-bison-001`

### Grok
- Requires `GROK_API_KEY` environment variable
- Optionally configure `GROK_API_URL` (defaults to example URL)
- Generic HTTP implementation - adjust for actual Grok API format

### Custom Provider
- Allows integration with any REST API
- Configure via `extra` parameters:
  - `url`: API endpoint
  - `method`: HTTP method (default: POST)
  - `headers`: Custom headers
  - `body_template`: JSON template with parameter substitution

## 🧪 Testing

The project includes comprehensive tests covering all providers and endpoints.

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest test_main.py -v

# Run specific test classes
pytest test_main.py::TestHealthAndUtility -v
pytest test_main.py::TestGenerateEndpoint -v

# Run manual tests (no dependencies required)
python test_main.py
```

### Test Coverage
- ✅ Health and utility endpoints
- ✅ Provider configuration validation
- ✅ Generate endpoint with all providers
- ✅ Chat endpoint functionality
- ✅ LangGraph workflow execution
- ✅ Models listing for all providers
- ✅ Error handling and validation
- ✅ Environment configuration

## 🛠️ Development

### Development Container
This project includes a complete VS Code development container:

```bash
# Open in VS Code
code .

# VS Code will prompt to "Reopen in Container"
# This provides:
# - Python 3.11 environment
# - All dependencies pre-installed
# - Ollama server running
# - Extensions for Python, Jupyter, etc.
```

### Manual Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Support
```bash
# Full stack with all services
docker-compose up --build

# Run specific services only
docker-compose up ollama api          # Just API + Ollama
docker-compose up ollama api streamlit # All services

# Check service logs
docker-compose logs ollama
docker-compose logs api
docker-compose logs streamlit

# Pull additional Ollama models
docker-compose exec ollama ollama pull mistral:7b-instruct
docker-compose exec ollama ollama list
```

## 📚 Usage Examples

### Example 1: Quick Text Generation
```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "provider": "ollama",
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "model": "codellama:7b-instruct",
    "temperature": 0.2
})

print(response.json()["text"])
```

### Example 2: Multi-turn Conversation
```python
import requests

# Start conversation
messages = [{"role": "user", "content": "What is FastAPI?"}]

response = requests.post("http://localhost:8000/chat", json={
    "provider": "ollama",
    "messages": messages,
    "model": "llama2:7b-chat"
})

# Add assistant response and continue
messages.append({
    "role": "assistant", 
    "content": response.json()["response"]
})
messages.append({
    "role": "user", 
    "content": "How does it compare to Flask?"
})

response = requests.post("http://localhost:8000/chat", json={
    "provider": "ollama",
    "messages": messages
})
```

### Example 3: Research Workflow
```python
import requests

response = requests.post("http://localhost:8000/workflow", json={
    "task_type": "research",
    "input_text": "Impact of large language models on software development",
    "provider": "ollama",
    "model": "llama2:7b-chat"
})

# Get comprehensive research report
research_report = response.json()["result"]
print(research_report)
```

## 🔧 Troubleshooting

### Installation Issues

**1. Python Package Import Errors**
```bash
# If you get "ImportError: No module named 'langchain'"
pip install --upgrade pip
pip install -r requirements.txt

# If specific packages fail, install individually:
pip install langchain langchain-openai langchain-ollama langgraph
pip install fastapi uvicorn streamlit
```

**2. Ollama Command Not Found**
```bash
# After installing Ollama, restart your terminal
# Or manually add to PATH:
export PATH=$PATH:/usr/local/bin

# Verify installation:
which ollama
ollama --version
```

**3. Ollama Server Issues**
```bash
# Check if already running:
ps aux | grep ollama
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve &

# Or run in foreground for debugging:
ollama serve
```

**4. Permission Issues (Linux)**
```bash
# Add user to ollama group:
sudo usermod -a -G ollama $USER

# Restart terminal or run:
newgrp ollama
```

### Runtime Issues

**1. Ollama Models Not Loading**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull models manually
ollama pull llama2:7b-chat
ollama pull mistral:7b-instruct

# List available models
ollama list
```

**2. API Key Issues**
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY

# Verify .env file exists
ls -la .env
cat .env
```

**3. Port Conflicts**
```bash
# Check what's running on ports
lsof -i :8000  # FastAPI
lsof -i :8501  # Streamlit  
lsof -i :11434 # Ollama

# Use different ports if needed
uvicorn main:app --port 8001
streamlit run demo_app.py --server.port 8502
```

**4. Docker Issues**
```bash
# Check container logs
docker-compose logs api
docker-compose logs ollama

# Restart services
docker-compose restart
docker-compose down && docker-compose up --build

# Clean up
docker-compose down -v  # Remove volumes
docker system prune    # Clean Docker cache
```

**5. Memory Issues with Large Models**
```bash
# Check system resources
docker stats

# Use smaller models
ollama pull llama2:7b-chat      # Instead of 13B/70B models
ollama pull mistral:7b-instruct # Lightweight alternative
```

## 🚀 Quick Start Summary

**TL;DR - Get running in 5 minutes:**

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd aigents_environment_tester
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Install Ollama**:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama serve &
   ollama pull llama2:7b-chat
   ```

3. **Verify and run**:
   ```bash
   python verify_setup.py  # Check everything works
   python main.py &         # Start API server
   streamlit run demo_app.py # Start demo interface
   ```

4. **Access**:
   - Demo UI: http://localhost:8501
   - API docs: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

**Using Docker (even easier)**:
```bash
git clone <repository-url>
cd aigents_environment_tester
cp .env.example .env  # Edit with your API keys if needed
docker-compose up --build
# Visit http://localhost:8501
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

cd /workspaces/aigents_environment_tester && python test_main.py