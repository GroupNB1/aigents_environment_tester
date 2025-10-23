from typing import Optional, Dict, Any, List
import os
from datetime import datetime
import asyncio

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseMessage

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()

app = FastAPI(
    title="LangChain LangGraph Ollama Demo API",
    description="Multi-provider LLM API with LangChain, LangGraph, and Ollama support",
    version="2.0.0"
)

# Request models
class GenerateRequest(BaseModel):
    provider: str = Field(..., description="one of: openai, anthropic, gemini, ollama, grok, custom")
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    system_message: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    provider: str = Field(..., description="LLM provider")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages with 'role' and 'content'")
    model: Optional[str] = None
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 512

class GraphWorkflowRequest(BaseModel):
    task_type: str = Field(..., description="one of: research, analysis, creative")
    input_text: str
    provider: Optional[str] = "ollama"
    model: Optional[str] = None

# Response models
class GenerateResponse(BaseModel):
    provider: str
    model: str
    text: str
    meta: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    provider: str
    model: str
    response: str
    conversation_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# --- LangChain Provider implementations ---

def get_langchain_llm(provider: str, model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 512):
    """Get LangChain LLM instance for the specified provider."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")
        return ChatOpenAI(
            api_key=api_key,
            model=model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")
        return ChatAnthropic(
            api_key=api_key,
            model=model or "claude-3-sonnet-20240229",
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model or "gemini-pro",
            temperature=temperature,
            max_output_tokens=max_tokens
        )
    
    elif provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            base_url=base_url,
            model=model or os.getenv("OLLAMA_MODEL", "llama2:7b-chat"),
            temperature=temperature,
            num_predict=max_tokens
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' not supported in LangChain mode")

async def generate_langchain(
    provider: str, 
    prompt: str, 
    model: Optional[str], 
    max_tokens: int, 
    temperature: float, 
    system_message: Optional[str] = None
):
    """Generate text using LangChain."""
    try:
        llm = get_langchain_llm(provider, model, temperature, max_tokens)
        
        # Create messages
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        # Generate response
        response = await llm.ainvoke(messages)
        
        return {
            "text": response.content,
            "model": model or llm.model_name if hasattr(llm, 'model_name') else "unknown",
            "provider_info": {
                "provider": provider,
                "langchain": True,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LangChain provider error: {str(e)}")

# Legacy provider implementations (keeping for compatibility)
async def generate_openai(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set")

    model = model or "gpt-4o-mini"
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if extra:
        payload.update(extra)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}") from e
        data = r.json()
        # Follow ChatCompletions response shape
        text = ""
        if "choices" in data and len(data["choices"]) > 0:
            ch = data["choices"][0]
            if "message" in ch:
                text = ch["message"].get("content", "")
            else:
                text = ch.get("text", "")
        else:
            text = data.get("error", {}).get("message", "") or str(data)
        return {"text": text, "model": model, "raw": data}


async def generate_anthropic(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not set")

    model = model or "claude-2.1"
    url = "https://api.anthropic.com/v1/complete"
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    # Anthropic expects a prompt including human/assistant tokens; keep it simple:
    anthropic_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
    payload = {
        "model": model,
        "prompt": anthropic_prompt,
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
    }
    if extra:
        payload.update(extra)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Anthropic error: {r.text}") from e
        data = r.json()
        text = data.get("completion", "")
        return {"text": text, "model": model, "raw": data}


async def generate_gemini(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    """
    Optional helper using google generative python package. If that package isn't installed,
    explain how to enable it or use the custom provider.
    """
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail="google.generativeai not installed or not configured. "
                   "Install google-generativeai and set GOOGLE_API_KEY, or use provider='custom'."
        ) from e

    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY not set for Gemini")

    genai.configure(api_key=key)
    model = model or "models/text-bison-001"
    # The package may accept different parameters; this is a simple call that works with typical installs.
    resp = genai.generate_text(model=model, prompt=prompt, temperature=temperature, max_output_tokens=max_tokens)
    # resp may have .text or .candidates
    text = getattr(resp, "text", None) or (resp.candidates[0].output if getattr(resp, "candidates", None) else str(resp))
    return {"text": text, "model": model, "raw": resp}


async def generate_grok(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    """
    Grok: generic HTTP call. Configure GROK_API_KEY and optionally GROK_API_URL.
    By default this function posts JSON {"input": prompt, "model": model, ...}.
    Adjust headers/body according to the real Grok API you use.
    """
    api_key = os.getenv("GROK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GROK_API_KEY not set")

    url = os.getenv("GROK_API_URL", "https://api.grok.ai/v1/generate")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": prompt, "model": model or "grok-1", "temperature": temperature, "max_tokens": max_tokens}
    if extra:
        payload.update(extra)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, json=payload, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Grok error: {r.text}") from e
        data = r.json()
        # Try common shapes
        text = data.get("text") or data.get("output") or (data.get("choices", [{}])[0].get("text") if data.get("choices") else str(data))
        return {"text": text, "model": model or "grok-1", "raw": data}


async def generate_custom(prompt: str, model: Optional[str], max_tokens: int, temperature: float, extra: Optional[dict]):
    """
    Call a custom HTTP endpoint provided via extra: {"url": "...", "method": "POST", "headers": {...}, "body_template": {...}}
    If body_template exists, it will be JSON-encoded and rendered with keys prompt/model/temperature/max_tokens.
    """
    if not extra or "url" not in extra:
        raise HTTPException(status_code=400, detail="custom provider requires extra.url")
    url = extra["url"]
    method = (extra.get("method") or "POST").upper()
    headers = extra.get("headers", {"Content-Type": "application/json"})
    # Basic templating
    body_template = extra.get("body_template") or {"prompt": "{prompt}", "model": "{model}", "temperature": "{temperature}", "max_tokens": "{max_tokens}"}
    # Render template
    def render(obj):
        if isinstance(obj, str):
            return obj.format(prompt=prompt, model=model or "", temperature=temperature, max_tokens=max_tokens)
        if isinstance(obj, dict):
            return {k: render(v) for k, v in obj.items()}
        return obj

    body = render(body_template)

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.request(method, url, json=body, headers=headers)
        try:
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"Custom provider error: {r.text}") from e
        # Try to parse JSON
        try:
            data = r.json()
        except Exception:
            return {"text": r.text, "model": model or "custom", "raw": {"status_code": r.status_code, "text": r.text}}
        text = data.get("text") or data.get("output") or (data.get("choices", [{}])[0].get("text") if data.get("choices") else str(data))
        return {"text": text, "model": model or "custom", "raw": data}


# --- LangGraph Workflow Definitions ---

class WorkflowState(BaseModel):
    """State for LangGraph workflows."""
    input_text: str
    task_type: str
    provider: str
    model: Optional[str] = None
    current_step: str = "start"
    results: Dict[str, Any] = {}
    final_output: str = ""

def create_research_workflow():
    """Create a research workflow using LangGraph."""
    
    async def research_step(state: Dict) -> Dict:
        """Research step in the workflow."""
        llm = get_langchain_llm(state["provider"], state.get("model"))
        
        research_prompt = f"""
        Research the following topic and provide key insights:
        Topic: {state['input_text']}
        
        Please provide:
        1. Main concepts and definitions
        2. Current trends or developments
        3. Key challenges or opportunities
        4. Relevant examples or case studies
        """
        
        response = await llm.ainvoke([HumanMessage(content=research_prompt)])
        
        state["results"]["research"] = response.content
        state["current_step"] = "analysis"
        return state
    
    async def analysis_step(state: Dict) -> Dict:
        """Analysis step in the workflow."""
        llm = get_langchain_llm(state["provider"], state.get("model"))
        
        analysis_prompt = f"""
        Based on the research findings below, provide a detailed analysis:
        
        Research Findings:
        {state['results']['research']}
        
        Please analyze:
        1. Implications and significance
        2. Strengths and weaknesses
        3. Future outlook
        4. Actionable recommendations
        """
        
        response = await llm.ainvoke([HumanMessage(content=analysis_prompt)])
        
        state["results"]["analysis"] = response.content
        state["current_step"] = "synthesis"
        return state
    
    async def synthesis_step(state: Dict) -> Dict:
        """Synthesis step to combine results."""
        llm = get_langchain_llm(state["provider"], state.get("model"))
        
        synthesis_prompt = f"""
        Create a comprehensive synthesis of the research and analysis:
        
        Research: {state['results']['research']}
        
        Analysis: {state['results']['analysis']}
        
        Provide a well-structured final report that combines all insights.
        """
        
        response = await llm.ainvoke([HumanMessage(content=synthesis_prompt)])
        
        state["final_output"] = response.content
        state["current_step"] = "complete"
        return state
    
    # Build the workflow graph
    workflow = StateGraph(dict)
    workflow.add_node("research", research_step)
    workflow.add_node("analysis", analysis_step)
    workflow.add_node("synthesis", synthesis_step)
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "analysis")
    workflow.add_edge("analysis", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# --- API Endpoints ---

@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text using LangChain providers."""
    provider = req.provider.lower()
    
    # Use LangChain for supported providers
    if provider in ["openai", "anthropic", "gemini", "ollama"]:
        try:
            result = await generate_langchain(
                provider=provider,
                prompt=req.prompt,
                model=req.model,
                max_tokens=req.max_tokens or 512,
                temperature=req.temperature or 0.0,
                system_message=req.system_message
            )
            return GenerateResponse(
                provider=provider,
                model=result["model"],
                text=result["text"],
                meta=result.get("provider_info")
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LangChain error: {str(e)}")
    
    # Fall back to legacy implementations
    handlers = {
        "grok": generate_grok,
        "custom": generate_custom,
    }
    
    if provider not in handlers:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown provider '{provider}'. Valid: openai, anthropic, gemini, ollama, grok, custom"
        )

    handler = handlers[provider]
    try:
        out = await handler(req.prompt, req.model, req.max_tokens or 512, req.temperature or 0.0, req.extra)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Provider error: {e}") from e

    return GenerateResponse(
        provider=provider, 
        model=out.get("model", req.model or ""), 
        text=out.get("text", ""), 
        meta={"raw": out.get("raw")}
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat endpoint with conversation support."""
    provider = req.provider.lower()
    
    if provider not in ["openai", "anthropic", "gemini", "ollama"]:
        raise HTTPException(status_code=400, detail=f"Chat not supported for provider: {provider}")
    
    try:
        llm = get_langchain_llm(provider, req.model, req.temperature or 0.0, req.max_tokens or 512)
        
        # Convert request messages to LangChain format
        messages = []
        for msg in req.messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                messages.append(SystemMessage(content=content))
            elif role == "assistant":
                messages.append(HumanMessage(content=f"Assistant: {content}"))  # Simplified
            else:  # user
                messages.append(HumanMessage(content=content))
        
        response = await llm.ainvoke(messages)
        
        return ChatResponse(
            provider=provider,
            model=req.model or "default",
            response=response.content,
            meta={"message_count": len(req.messages)}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/workflow")
async def run_workflow(req: GraphWorkflowRequest):
    """Run a LangGraph workflow."""
    task_type = req.task_type.lower()
    
    if task_type != "research":
        raise HTTPException(status_code=400, detail="Currently only 'research' workflow is supported")
    
    try:
        # Create and run the research workflow
        workflow = create_research_workflow()
        
        initial_state = {
            "input_text": req.input_text,
            "task_type": req.task_type,
            "provider": req.provider or "ollama",
            "model": req.model,
            "current_step": "start",
            "results": {},
            "final_output": ""
        }
        
        # Execute the workflow
        final_state = await workflow.ainvoke(initial_state)
        
        return {
            "workflow_type": task_type,
            "input": req.input_text,
            "provider": req.provider or "ollama",
            "steps_completed": ["research", "analysis", "synthesis"],
            "result": final_state["final_output"],
            "meta": {
                "research_findings": final_state["results"].get("research", ""),
                "analysis": final_state["results"].get("analysis", "")
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")


# Health and utility endpoints
@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "providers": {
            "langchain_supported": ["openai", "anthropic", "gemini", "ollama"],
            "legacy_supported": ["grok", "custom"],
            "workflows": ["research"]
        },
        "environment": {
            "ollama_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "langchain_tracing": os.getenv("LANGCHAIN_TRACING_V2", "false")
        }
    }

@app.get("/models/{provider}")
async def list_models(provider: str):
    """List available models for a provider."""
    provider = provider.lower()
    
    if provider == "ollama":
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = [model["name"] for model in data.get("models", [])]
                    return {"provider": provider, "models": models}
                else:
                    raise HTTPException(status_code=502, detail="Ollama server not available")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching Ollama models: {str(e)}")
    
    elif provider == "openai":
        return {
            "provider": provider,
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
        }
    elif provider == "anthropic":
        return {
            "provider": provider,
            "models": ["claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
        }
    elif provider == "gemini":
        return {
            "provider": provider,
            "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]
        }
    else:
        raise HTTPException(status_code=400, detail=f"Model listing not supported for provider: {provider}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple demo page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LangChain LangGraph Ollama Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦜 LangChain LangGraph Ollama Demo API</h1>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <h3>POST /generate</h3>
                <p>Generate text using various LLM providers</p>
                <code>{"provider": "ollama", "prompt": "Hello world", "model": "llama2:7b-chat"}</code>
            </div>
            
            <div class="endpoint">
                <h3>POST /chat</h3>
                <p>Chat with conversation context</p>
                <code>{"provider": "ollama", "messages": [{"role": "user", "content": "Hi there!"}]}</code>
            </div>
            
            <div class="endpoint">
                <h3>POST /workflow</h3>
                <p>Run LangGraph research workflow</p>
                <code>{"task_type": "research", "input_text": "Machine Learning trends", "provider": "ollama"}</code>
            </div>
            
            <div class="endpoint">
                <h3>GET /models/{provider}</h3>
                <p>List available models for a provider</p>
                <code>/models/ollama</code>
            </div>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Health check and system status</p>
            </div>
            
            <p><strong>Interactive API Documentation:</strong> <a href="/docs">/docs</a></p>
            <p><strong>ReDoc Documentation:</strong> <a href="/redoc">/redoc</a></p>
        </div>
    </body>
    </html>
    """


# Run with: python main.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
