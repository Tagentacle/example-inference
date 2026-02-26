# Example Inference Node

Stateless LLM gateway for Tagentacle. Exposes a `/inference/chat` Service that accepts OpenAI-compatible format and proxies to any OpenAI-compatible API.

## Configuration

Set credentials in bringup `config/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_BASE_URL = "https://api.openai.com/v1"   # or OpenRouter, DeepSeek, Ollama, etc.
```

## Usage

```bash
tagentacle run --pkg .
```

Or as part of the system launch:

```bash
tagentacle launch path/to/system_launch.toml
```

## Service Interface

**Service**: `/inference/chat`

**Request**:
```json
{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [],
    "temperature": 0.7
}
```

**Response**: Standard OpenAI completion format.
