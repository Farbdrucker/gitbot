# GitBot
A simple AI agent that will create git commit messages for you with a local LLM model served with `ollama

## Installation
### Ollama
#### On macOS
brew install ollama

#### On Linux
curl -fsSL https://ollama.ai/install.sh | sh

#### Start Ollama service
ollama serve

## Example
```shell
uv run main.py analyze --commit --unstaged
```

### Help
See all possible command with 
```json
uv run main.py --help
```

and for a specific command
```shell
uv run main.py analyze --help
```