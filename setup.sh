#!/bin/bash

# Post-create setup script for the devcontainer

echo "🚀 Setting up LangChain LangGraph Ollama Demo environment..."

# Update system packages
apt-get update && apt-get upgrade -y

# Install Ollama
echo "📦 Installing Ollama..."
curl -fsSL https://ollama.ai/install.sh | sh

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
fi

# Start Ollama service in background
echo "🦙 Starting Ollama service..."
ollama serve &

# Wait a bit for Ollama to start
sleep 5

# Pull some useful models (this can be customized)
echo "📥 Pulling Ollama models..."
ollama pull llama2:7b-chat
ollama pull codellama:7b-instruct
ollama pull mistral:7b-instruct

# Set up git config if not already set
if [ -z "$(git config --global user.email)" ]; then
    echo "🔧 Setting up Git configuration..."
    git config --global user.email "developer@example.com"
    git config --global user.name "Dev Container User"
fi

echo "✅ Setup complete! You can now:"
echo "  - Run FastAPI: python main.py"
echo "  - Run Streamlit demo: streamlit run demo_app.py"
echo "  - Access Ollama at: http://localhost:11434"
echo "  - Check available models: ollama list"

# Make the script executable for future runs
chmod +x .devcontainer/setup.sh