# GitBot
A simple AI agent that will create git commit messages for you with a local LLM model served with `ollama

## Installation
### Ollama
The easiest way to run this project with a local LLM model is to use [Ollama](https://ollama.com/), which provides a simple way to run large language models locally.
#### On macOS
```shell
brew install ollama
```

#### On Linux
```shell
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Start Ollama service
ollama serve

### Project with `uv`
The easiest way to run the project is to use [uv](https://astral.sh/), which is a command line tool that helps to run Python scripts with a flexible command line interface.
#### On Linux/macOS
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### On Windows
```shell
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Example
```shell
uv run main.py analyze --commit --unstaged
```

### Help
See all possible command with 
```shell
uv run main.py --help
```

and for a specific command
```shell
uv run main.py analyze --help
```