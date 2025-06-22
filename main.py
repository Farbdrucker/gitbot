#!/usr/bin/env python3
"""
Functional Git Diff Analyzer - Generates structured commit messages using functional programming
"""

import subprocess
import logging
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple, Callable, Union, Coroutine
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


# Data Models (immutable by design)
class FileChange(BaseModel):
    """Represents changes to a single file"""

    file_path: str = Field(description="Path to the changed file")
    change_type: str = Field(
        description="Type of change: added, modified, deleted, renamed"
    )
    summary: str = Field(description="Brief summary of what changed in this file")
    lines_added: int = Field(default=0, description="Number of lines added")
    lines_removed: int = Field(default=0, description="Number of lines removed")


class CommitSummary(BaseModel):
    """Complete commit summary with structured information"""

    overall_summary: str = Field(description="One-line summary of all changes")
    detailed_description: Optional[str] = Field(
        default=None, description="Detailed description if needed"
    )
    file_changes: List[FileChange] = Field(
        description="List of individual file changes"
    )
    commit_type: str = Field(
        description="Type of commit: feat, fix, docs, style, refactor, test, chore"
    )


@dataclass(frozen=True)
class AnalyzerConfig:
    """Immutable configuration for the analyzer"""

    model_name: str = "llama3.2"
    base_url: str = "http://localhost:11434/v1"
    supports_structured_output: bool = True


@dataclass(frozen=True)
class GitContext:
    """Immutable context about git repository state"""

    is_git_repo: bool
    has_staged_changes: bool
    diff_output: str


# Pure Functions for Git Operations
def run_git_command(cmd: List[str]) -> str:
    """Pure function to run git command and return output"""
    logger.debug(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e}")


def check_git_repo() -> bool:
    """Check if current directory is a git repository"""
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def check_staged_changes() -> bool:
    """Check if there are staged changes"""
    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def get_git_diff(staged: bool = True) -> str:
    """Get git diff output"""
    cmd_base = ["git", "diff", "--staged"] if staged else ["git", "diff"]

    # Get file status and stats
    status_output = run_git_command(cmd_base + ["--stat", "--name-status"])

    # Get actual diff content
    diff_content = run_git_command(cmd_base)

    return f"File Status:\n{status_output}\n\nDiff Content:\n{diff_content}"


def create_git_context(staged: bool = True) -> GitContext:
    """Create immutable git context"""
    is_repo = check_git_repo()
    has_staged = check_staged_changes() if staged else True
    diff_output = get_git_diff(staged) if is_repo and has_staged else ""

    return GitContext(
        is_git_repo=is_repo, has_staged_changes=has_staged, diff_output=diff_output
    )


# Pure Functions for Validation
def validate_git_context(
    context: GitContext, staged: bool
) -> Union[GitContext, RuntimeError]:
    """Validate git context and return it or an error"""
    if not context.is_git_repo:
        return RuntimeError("Not in a git repository")

    if staged and not context.has_staged_changes:
        return RuntimeError(
            "No staged changes found. Use 'git add' to stage files first."
        )

    if not context.diff_output.strip():
        return RuntimeError("No changes found")

    return context


# Pure Functions for AI Model Creation
def create_system_prompt() -> str:
    """Create system prompt for AI analysis"""
    return """
    You are an expert software developer who writes excellent commit messages.

    Analyze git diff output and create structured commit summaries that follow conventional commit format.

    Guidelines:
    - Use conventional commit types: feat, fix, docs, style, refactor, test, chore
    - Keep overall summary under 50 characters when possible
    - Be specific about what changed, not just which files
    - Focus on the "why" and "what" rather than technical details
    - For each file, summarize the purpose/impact of changes
    - Use EXACT change types: "added", "modified", "deleted", "renamed" (not "M", "A", "D")

    IMPORTANT: The file_changes field must be an array of objects, not a string.
    Each file change object must have:
    - file_path: string
    - change_type: one of "added", "modified", "deleted", "renamed" 
    - summary: string describing what changed
    - lines_added: integer (default 0)
    - lines_removed: integer (default 0)

    Examples of good summaries:
    - "Add user authentication middleware"
    - "Fix memory leak in data processing loop"
    - "Update API documentation for v2 endpoints"

    Return your analysis in the exact JSON structure requested.
    """


def create_fallback_prompt() -> str:
    """Create fallback prompt for models without structured output"""
    return """
    You are an expert software developer who writes excellent commit messages.

    Analyze git diff output and create a commit message that follows conventional commit format.

    Guidelines:
    - Use conventional commit types: feat, fix, docs, style, refactor, test, chore
    - Keep the first line under 50 characters when possible
    - Be specific about what changed, not just which files
    - Focus on the "why" and "what" rather than technical details
    - Use EXACT change types: "added", "modified", "deleted", "renamed" (not "M", "A", "D")

    Format your response as JSON with this exact structure:
    {
      "overall_summary": "brief one-line summary",
      "detailed_description": "optional longer description",
      "commit_type": "feat|fix|docs|style|refactor|test|chore",
      "file_changes": [
        {
          "file_path": "path/to/file",
          "change_type": "added|modified|deleted|renamed",
          "summary": "what changed in this file",
          "lines_added": 0,
          "lines_removed": 0
        }
      ]
    }

    CRITICAL: file_changes must be an array of objects, NOT a string.

    Examples of good summaries:
    - "Add user authentication middleware"
    - "Fix memory leak in data processing loop"
    - "Update API documentation for v2 endpoints"

    IMPORTANT: Return ONLY the JSON, no other text.
    """


def create_agent(config: AnalyzerConfig) -> Tuple[Agent, bool]:
    """Create AI agent with configuration"""
    model = OpenAIModel(
        model_name=config.model_name, provider=OpenAIProvider(base_url=config.base_url)
    )

    try:
        agent = Agent(
            model=model,
            result_type=CommitSummary,
            system_prompt=create_system_prompt(),
            retries=2,  # Add retries for validation errors
        )
        return agent, True
    except Exception as e:
        logger.warning(
            f"Model doesn't support structured output, falling back to text mode: {e}"
        )
        agent = Agent(model=model, system_prompt=create_fallback_prompt(), retries=2)
        return agent, False


# Pure Functions for Response Processing
def clean_json_response(response: str) -> str:
    """Clean JSON response from markdown formatting"""
    response = response.strip()
    if response.startswith("```json"):
        response = response.split("```json")[1].split("```")[0].strip()
    elif response.startswith("```"):
        response = response.split("```")[1].split("```")[0].strip()
    return response


def normalize_change_type(change_type: str) -> str:
    """Normalize git change type to expected format"""
    mapping = {
        "M": "modified",
        "A": "added",
        "D": "deleted",
        "R": "renamed",
        "modified": "modified",
        "added": "added",
        "deleted": "deleted",
        "renamed": "renamed",
    }
    return mapping.get(change_type, "modified")


def fix_file_changes(file_changes: Any) -> List[Dict[str, Any]]:
    """Fix file_changes if it's a string or has incorrect format"""
    if isinstance(file_changes, str):
        try:
            # Try to parse as JSON string
            parsed = json.loads(file_changes)
            if isinstance(parsed, list):
                file_changes = parsed
            else:
                file_changes = [parsed]
        except json.JSONDecodeError:
            # Create a basic file change entry
            file_changes = [
                {
                    "file_path": "unknown",
                    "change_type": "modified",
                    "summary": "File updated",
                    "lines_added": 0,
                    "lines_removed": 0,
                }
            ]

    if not isinstance(file_changes, list):
        file_changes = [file_changes] if file_changes else []

    # Normalize each file change
    normalized = []
    for change in file_changes:
        if isinstance(change, dict):
            normalized_change = {
                "file_path": change.get("file_path", "unknown"),
                "change_type": normalize_change_type(
                    change.get("change_type", "modified")
                ),
                "summary": change.get("summary", change.get("purpose", "File updated")),
                "lines_added": change.get("lines_added", 0),
                "lines_removed": change.get("lines_removed", 0),
            }
            normalized.append(normalized_change)

    return (
        normalized
        if normalized
        else [
            {
                "file_path": "unknown",
                "change_type": "modified",
                "summary": "File updated",
                "lines_added": 0,
                "lines_removed": 0,
            }
        ]
    )


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON response with error handling and normalization"""
    try:
        cleaned = clean_json_response(response)
        data = json.loads(cleaned)

        # Fix file_changes if needed
        if "file_changes" in data:
            data["file_changes"] = fix_file_changes(data["file_changes"])

        # Ensure required fields exist
        if "overall_summary" not in data:
            data["overall_summary"] = "Update files"
        if "commit_type" not in data:
            data["commit_type"] = "chore"
        if "file_changes" not in data:
            data["file_changes"] = []

        return data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response}")
        return {
            "overall_summary": "Update files",
            "commit_type": "chore",
            "file_changes": [
                {
                    "file_path": "unknown",
                    "change_type": "modified",
                    "summary": "File updated",
                    "lines_added": 0,
                    "lines_removed": 0,
                }
            ],
            "detailed_description": f"AI parsing failed: {response[:200]}...",
        }


def create_commit_summary_from_dict(data: Dict[str, Any]) -> CommitSummary:
    """Create CommitSummary from dictionary"""
    return CommitSummary(**data)


# Higher-order Functions for Analysis Pipeline
def create_analysis_pipeline(
    agent: Agent, supports_structured: bool
) -> Callable[[str], Coroutine[Any, Any, CommitSummary]]:
    """Create analysis pipeline function"""

    async def analyze_diff(diff_output: str) -> CommitSummary:
        result = await agent.run(
            f"Analyze this git diff and create a structured commit summary:\n\n{diff_output}"
        )

        if supports_structured:
            return result.output
        else:
            # Functional composition for fallback processing
            return (lambda x: create_commit_summary_from_dict(parse_json_response(x)))(
                result.data
            )

    return analyze_diff


# Pure Functions for Formatting
def format_commit_message(summary: CommitSummary) -> str:
    """Format the commit summary as a conventional commit message"""
    lines = [f"{summary.commit_type}: {summary.overall_summary}"]

    if summary.detailed_description:
        lines.extend(["", summary.detailed_description])

    if summary.file_changes:
        lines.extend(["", "Files changed:"])
        for change in summary.file_changes:
            line = f"- {change.file_path}: {change.summary}"
            if change.lines_added or change.lines_removed:
                line += f" (+{change.lines_added}/-{change.lines_removed})"
            lines.append(line)

    return "\n".join(lines)


# Higher-order Functions for Effects
def with_progress(description: str) -> Callable:
    """Higher-order function to wrap async operations with progress"""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(description, total=None)
                result = await func(*args, **kwargs)
                progress.update(task, completed=True)
                return result

        return wrapper

    return decorator


# Main Analysis Function (Functional Composition)
async def analyze_git_changes(
    config: AnalyzerConfig, staged: bool = True
) -> CommitSummary:
    """Main analysis function using functional composition"""
    # Create immutable context
    context = create_git_context(staged)

    # Validate context
    validated = validate_git_context(context, staged)
    if isinstance(validated, RuntimeError):
        raise validated

    # Create agent
    agent, supports_structured = create_agent(config)

    # Create analysis pipeline
    analyze = create_analysis_pipeline(agent, supports_structured)

    # Apply progress wrapper and run analysis
    analyze_with_progress = with_progress("Analyzing changes with AI...")(analyze)

    return await analyze_with_progress(context.diff_output)


# Utility Functions for CLI
def run_subprocess_command(cmd: List[str]) -> str:
    """Run subprocess command and return output"""
    logger.debug(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command failed: {e}")


def check_ollama_available() -> bool:
    """Check if Ollama is available"""
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# CLI Application
app = typer.Typer(
    name="git-analyzer",
    help="Analyze git changes and generate commit messages using local AI models",
    rich_markup_mode="rich",
)


@app.command("analyze", short_help="Analyze git changes and generate commit messages")
def analyze(
    unstaged: bool = typer.Option(
        False, "--unstaged", help="Analyze unstaged changes instead of staged"
    ),
    model: str = typer.Option("llama3.1:8b", "--model", help="Ollama model to use"),
    base_url: str = typer.Option(
        "http://localhost:11434/v1", "--base-url", help="Ollama base URL"
    ),
    commit: bool = typer.Option(
        False, "--commit", help="Automatically commit with generated message"
    ),
    output_format: str = typer.Option("text", "--output", help="Output format"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging"
    ),
):
    """Analyze git changes and generate commit messages"""

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print(
        Panel.fit(
            f"[bold blue]Git Diff Analyzer[/bold blue]\n"
            f"Model: [cyan]{model}[/cyan]\n"
            f"Analyzing: [yellow]{'unstaged' if unstaged else 'staged'}[/yellow] changes",
            border_style="blue",
        )
    )

    config = AnalyzerConfig(model_name=model, base_url=base_url)

    try:
        # Functional composition for the main workflow
        summary = asyncio.run(analyze_git_changes(config, staged=not unstaged))

        if output_format == "json":
            console.print_json(summary.model_dump_json(indent=2))
        else:
            commit_message = format_commit_message(summary)

            console.print("\n[bold green]Generated Commit Message:[/bold green]")
            console.print(
                Panel(commit_message, border_style="green", title="Commit Message")
            )

            if commit:
                confirm = typer.confirm("\nCommit with this message?")
                if confirm:
                    commit_cmd = ["git", "commit", "-m"]
                    if unstaged:
                        commit_cmd.append("-a")
                    commit_cmd.append(commit_message)
                    run_subprocess_command(commit_cmd)
                    console.print(
                        "✅ [bold green]Changes committed successfully![/bold green]"
                    )
                else:
                    console.print("❌ [yellow]Commit cancelled[/yellow]")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def models():
    """List available Ollama models"""
    console.print("[bold blue]Checking available Ollama models...[/bold blue]")

    try:
        if not check_ollama_available():
            raise RuntimeError("Ollama not available")

        output = run_subprocess_command(["ollama", "list"])
        console.print("\n[bold green]Available Models:[/bold green]")
        console.print(output)

    except Exception:
        console.print(
            "❌ [bold red]Failed to list models. Is Ollama running?[/bold red]"
        )
        console.print("Try running: [cyan]ollama serve[/cyan]")
        raise typer.Exit(1)


@app.command()
def pull(model_name: str = typer.Argument(help="Model name to pull")):
    """Pull a new Ollama model"""
    console.print(f"[bold blue]Pulling model: {model_name}[/bold blue]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {model_name}...", total=None)
            subprocess.run(["ollama", "pull", model_name], check=True)
            progress.update(task, completed=True)

        console.print(f"✅ [bold green]Successfully pulled {model_name}[/bold green]")

    except subprocess.CalledProcessError as e:
        console.print(f"❌ [bold red]Failed to pull model: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
