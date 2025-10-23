"""
Streamlit Demo App for LangChain LangGraph Ollama API

This app provides a user-friendly interface to interact with the multi-provider LLM API.
Run with: streamlit run demo_app.py
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def get_available_models(provider):
    """Get available models for a provider."""
    try:
        response = requests.get(f"{API_BASE_URL}/models/{provider}")
        if response.status_code == 200:
            return response.json().get("models", [])
        return []
    except:
        return []

def call_generate_api(provider, prompt, model=None, temperature=0.7, max_tokens=512, system_message=None):
    """Call the generate API endpoint."""
    payload = {
        "provider": provider,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if model:
        payload["model"] = model
    if system_message:
        payload["system_message"] = system_message
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=payload, timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def call_chat_api(provider, messages, model=None, temperature=0.7, max_tokens=512):
    """Call the chat API endpoint."""
    payload = {
        "provider": provider,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    if model:
        payload["model"] = model
    
    try:
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def call_workflow_api(task_type, input_text, provider="ollama", model=None):
    """Call the workflow API endpoint."""
    payload = {
        "task_type": task_type,
        "input_text": input_text,
        "provider": provider
    }
    
    if model:
        payload["model"] = model
    
    try:
        response = requests.post(f"{API_BASE_URL}/workflow", json=payload, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, str(e)

def main():
    st.set_page_config(
        page_title="LangChain LangGraph Ollama Demo", 
        page_icon="🦜",
        layout="wide"
    )
    
    st.title("🦜 LangChain LangGraph Ollama Demo")
    st.markdown("Interactive demo for the multi-provider LLM API with LangChain, LangGraph, and Ollama support.")
    
    # Check API health
    is_healthy, health_data = check_api_health()
    
    if not is_healthy:
        st.error("❌ API is not running! Please start the FastAPI server first.")
        st.code("python main.py", language="bash")
        return
    
    st.success("✅ API is running!")
    
    # Sidebar for API information
    with st.sidebar:
        st.header("API Status")
        if health_data:
            st.json(health_data)
        
        st.header("Configuration")
        st.text(f"API URL: {API_BASE_URL}")
        
        # Environment check
        env_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "OLLAMA_BASE_URL"]
        st.subheader("Environment Variables")
        for var in env_vars:
            value = os.getenv(var, "Not set")
            if "API_KEY" in var:
                display_value = "***" if value != "Not set" else "Not set"
            else:
                display_value = value
            st.text(f"{var}: {display_value}")
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Generate", "🔄 Chat", "🔬 Workflow", "📊 Models"])
    
    with tab1:
        st.header("Text Generation")
        st.markdown("Generate text using various LLM providers.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            provider = st.selectbox("Provider", ["ollama", "openai", "anthropic", "gemini"], key="gen_provider")
            
            # Get available models for the provider
            available_models = get_available_models(provider)
            if available_models:
                model = st.selectbox("Model", ["Default"] + available_models, key="gen_model")
                model = None if model == "Default" else model
            else:
                model = st.text_input("Model (optional)", key="gen_model_input")
                model = model if model else None
        
        with col2:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="gen_temp")
            max_tokens = st.slider("Max Tokens", 50, 2000, 512, 50, key="gen_tokens")
        
        system_message = st.text_area("System Message (optional)", 
                                    placeholder="You are a helpful assistant...", 
                                    key="gen_system")
        
        prompt = st.text_area("Prompt", 
                            placeholder="Enter your prompt here...", 
                            height=100, key="gen_prompt")
        
        if st.button("Generate", type="primary", key="gen_button"):
            if not prompt:
                st.error("Please enter a prompt!")
            else:
                with st.spinner("Generating response..."):
                    start_time = time.time()
                    success, result = call_generate_api(
                        provider, prompt, model, temperature, max_tokens, 
                        system_message if system_message else None
                    )
                    end_time = time.time()
                
                if success:
                    st.success(f"Generated in {end_time - start_time:.2f} seconds")
                    st.markdown("### Response")
                    st.write(result["text"])
                    
                    with st.expander("Response Details"):
                        st.json(result)
                else:
                    st.error(f"Error: {result}")
    
    with tab2:
        st.header("Chat Interface")
        st.markdown("Have a conversation with the LLM.")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chat_provider = st.selectbox("Provider", ["ollama", "openai", "anthropic", "gemini"], key="chat_provider")
            chat_models = get_available_models(chat_provider)
            if chat_models:
                chat_model = st.selectbox("Model", ["Default"] + chat_models, key="chat_model")
                chat_model = None if chat_model == "Default" else chat_model
            else:
                chat_model = st.text_input("Model (optional)", key="chat_model_input")
                chat_model = chat_model if chat_model else None
        
        with col2:
            chat_temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, key="chat_temp")
            chat_tokens = st.slider("Max Tokens", 50, 2000, 512, 50, key="chat_tokens")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### Conversation")
            for i, msg in enumerate(st.session_state.chat_history):
                role = msg["role"]
                content = msg["content"]
                
                if role == "user":
                    st.markdown(f"**You:** {content}")
                else:
                    st.markdown(f"**Assistant:** {content}")
        
        # Chat input
        user_input = st.text_input("Your message:", key="chat_input")
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("Send", type="primary", key="chat_send"):
                if user_input:
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    
                    # Prepare messages for API
                    messages = []
                    for msg in st.session_state.chat_history:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    with st.spinner("Getting response..."):
                        success, result = call_chat_api(chat_provider, messages, chat_model, chat_temp, chat_tokens)
                    
                    if success:
                        assistant_response = result["response"]
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        st.rerun()
                    else:
                        st.error(f"Error: {result}")
        
        with col_clear:
            if st.button("Clear Chat", key="chat_clear"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab3:
        st.header("LangGraph Workflows")
        st.markdown("Run complex multi-step workflows using LangGraph.")
        
        workflow_type = st.selectbox("Workflow Type", ["research"], key="workflow_type")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            workflow_provider = st.selectbox("Provider", ["ollama", "openai", "anthropic", "gemini"], key="workflow_provider")
            workflow_models = get_available_models(workflow_provider)
            if workflow_models:
                workflow_model = st.selectbox("Model", ["Default"] + workflow_models, key="workflow_model")
                workflow_model = None if workflow_model == "Default" else workflow_model
            else:
                workflow_model = st.text_input("Model (optional)", key="workflow_model_input")
                workflow_model = workflow_model if workflow_model else None
        
        input_text = st.text_area("Research Topic", 
                                placeholder="Enter the topic you want to research...", 
                                height=100, key="workflow_input")
        
        if st.button("Run Workflow", type="primary", key="workflow_button"):
            if not input_text:
                st.error("Please enter a research topic!")
            else:
                with st.spinner("Running workflow... This may take a few minutes."):
                    start_time = time.time()
                    success, result = call_workflow_api(workflow_type, input_text, workflow_provider, workflow_model)
                    end_time = time.time()
                
                if success:
                    st.success(f"Workflow completed in {end_time - start_time:.2f} seconds")
                    
                    st.markdown("### Final Result")
                    st.write(result["result"])
                    
                    if "meta" in result:
                        with st.expander("Detailed Steps"):
                            if "research_findings" in result["meta"]:
                                st.markdown("#### Research Findings")
                                st.write(result["meta"]["research_findings"])
                            
                            if "analysis" in result["meta"]:
                                st.markdown("#### Analysis")
                                st.write(result["meta"]["analysis"])
                    
                    with st.expander("Full Response"):
                        st.json(result)
                else:
                    st.error(f"Error: {result}")
    
    with tab4:
        st.header("Available Models")
        st.markdown("Browse available models for each provider.")
        
        providers = ["ollama", "openai", "anthropic", "gemini"]
        
        for provider in providers:
            with st.expander(f"{provider.title()} Models"):
                models = get_available_models(provider)
                if models:
                    st.success(f"Found {len(models)} models")
                    for model in models:
                        st.text(f"• {model}")
                else:
                    st.warning(f"No models available or provider not accessible")
                    if provider == "ollama":
                        st.info("Make sure Ollama is running and has models installed.")
                        st.code("ollama pull llama2:7b-chat", language="bash")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit, FastAPI, LangChain, LangGraph, and Ollama 🦜🔗")

if __name__ == "__main__":
    main()