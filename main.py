#!/usr/bin/env python3
"""
Git Diff Analyzer - Generates structured commit messages using PydanticAI with Ollama
"""

import subprocess
import logging
from typing import List, Optional, Annotated
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
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

class FileChange(BaseModel):
    """Represents changes to a single file"""
    file_path: str = Field(description="Path to the changed file")
    change_type: str = Field(description="Type of change: added, modified, deleted, renamed")
    summary: str = Field(description="Brief summary of what changed in this file")
    lines_added: int = Field(default=0, description="Number of lines added")
    lines_removed: int = Field(default=0, description="Number of lines removed")

class CommitSummary(BaseModel):
    """Complete commit summary with structured information"""
    overall_summary: str = Field(description="One-line summary of all changes")
    detailed_description: Optional[str] = Field(default=None, description="Detailed description if needed")
    file_changes: List[FileChange] = Field(description="List of individual file changes")
    commit_type: str = Field(description="Type of commit: feat, fix, docs, style, refactor, test, chore")

class GitDiffAnalyzer:
    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434/v1"):
        """Initialize the analyzer with PydanticAI agent using Ollama"""
        logger.info(f"Initializing analyzer with model: {model_name}")

        self.ollama_model = OpenAIModel(
            model_name=model_name,
            provider=OpenAIProvider(base_url=base_url)
        )

        # Try to create agent with structured output, fallback to text if tools not supported
        try:
            self.agent = Agent(
                model=self.ollama_model,
                result_type=CommitSummary,
                system_prompt="""
                You are an expert software developer who writes excellent commit messages.

                Analyze git diff output and create structured commit summaries that follow conventional commit format.

                Guidelines:
                - Use conventional commit types: feat, fix, docs, style, refactor, test, chore
                - Keep overall summary under 50 characters when possible
                - Be specific about what changed, not just which files
                - Focus on the "why" and "what" rather than technical details
                - For each file, summarize the purpose/impact of changes
                - Identify the change type accurately (added/modified/deleted/renamed)

                Examples of good summaries:
                - "Add user authentication middleware"
                - "Fix memory leak in data processing loop"
                - "Update API documentation for v2 endpoints"

                Return your analysis in the exact JSON structure requested.
                """
            )
            self.supports_structured_output = True
        except Exception as e:
            logger.warning(f"Model doesn't support structured output, falling back to text mode: {e}")
            self.agent = Agent(
                model=self.ollama_model,
                system_prompt="""
                You are an expert software developer who writes excellent commit messages.

                Analyze git diff output and create a commit message that follows conventional commit format.

                Guidelines:
                - Use conventional commit types: feat, fix, docs, style, refactor, test, chore
                - Keep the first line under 50 characters when possible
                - Be specific about what changed, not just which files
                - Focus on the "why" and "what" rather than technical details

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

                Examples of good summaries:
                - "Add user authentication middleware"
                - "Fix memory leak in data processing loop"
                - "Update API documentation for v2 endpoints"

                IMPORTANT: Return ONLY the JSON, no other text.
                """
            )
            self.supports_structured_output = False
        logger.info("Agent initialized successfully")

    def get_git_diff(self, staged: bool = True) -> str:
        """Get git diff output"""
        logger.info(f"Getting git diff (staged: {staged})")

        try:
            # Get staged changes by default, or all changes if staged=False
            cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]

            # Add --name-status for file change types and --stat for line counts
            logger.debug(f"Running command: {' '.join(cmd + ['--stat', '--name-status'])}")
            diff_output = subprocess.run(
                cmd + ["--stat", "--name-status"],
                capture_output=True,
                text=True,
                check=True
            )

            # Also get the actual diff content
            logger.debug(f"Running command: {' '.join(cmd)}")
            diff_content = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            combined_diff = f"File Status:\n{diff_output.stdout}\n\nDiff Content:\n{diff_content.stdout}"
            logger.info(f"Retrieved diff with {len(combined_diff)} characters")
            return combined_diff

        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e}")
            raise RuntimeError(f"Git command failed: {e}")

    def check_git_repo(self) -> bool:
        """Check if current directory is a git repository"""
        logger.debug("Checking if current directory is a git repository")
        try:
            subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                check=True
            )
            logger.debug("Confirmed: in git repository")
            return True
        except subprocess.CalledProcessError:
            logger.warning("Not in a git repository")
            return False

    def has_staged_changes(self) -> bool:
        """Check if there are staged changes"""
        logger.debug("Checking for staged changes")
        try:
            result = subprocess.run(
                ["git", "diff", "--staged", "--name-only"],
                capture_output=True,
                text=True,
                check=True
            )
            has_changes = bool(result.stdout.strip())
            logger.debug(f"Staged changes found: {has_changes}")
            return has_changes
        except subprocess.CalledProcessError:
            logger.warning("Failed to check for staged changes")
            return False

    async def analyze_changes(self, staged: bool = True) -> CommitSummary:
        """Analyze git changes and generate commit summary"""
        logger.info("Starting analysis of git changes")

        if not self.check_git_repo():
            raise RuntimeError("Not in a git repository")

        if staged and not self.has_staged_changes():
            raise RuntimeError("No staged changes found. Use 'git add' to stage files first.")

        # Get the diff
        diff_output = self.get_git_diff(staged)

        if not diff_output.strip():
            raise RuntimeError("No changes found")

        # Use PydanticAI to analyze the diff
        logger.info("Sending diff to AI model for analysis")
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
        ) as progress:
            task = progress.add_task("Analyzing changes with AI...", total=None)

            if self.supports_structured_output:
                result = await self.agent.run(
                    f"Analyze this git diff and create a structured commit summary:\n\n{diff_output}"
                )
                summary = result.data
            else:
                # Fallback for models without structured output support
                result = await self.agent.run(
                    f"Analyze this git diff and create a commit message:\n\n{diff_output}"
                )

                # Parse the JSON response manually
                import json
                try:
                    response_text = result.data.strip()
                    # Remove any markdown code blocks if present
                    if response_text.startswith('```json'):
                        response_text = response_text.split('```json')[1].split('```')[0].strip()
                    elif response_text.startswith('```'):
                        response_text = response_text.split('```')[1].split('```')[0].strip()

                    json_data = json.loads(response_text)
                    summary = CommitSummary(**json_data)
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.debug(f"Raw response: {result.data}")
                    # Create a basic summary as fallback
                    summary = CommitSummary(
                        overall_summary="Update files",
                        commit_type="chore",
                        file_changes=[],
                        detailed_description=f"AI response: {result.data[:200]}..."
                    )

            progress.update(task, completed=True)

        logger.info("AI analysis completed successfully")
        logger.debug(f"Token usage: {result.usage()}")

        return summary

    def format_commit_message(self, summary: CommitSummary) -> str:
        """Format the commit summary as a conventional commit message"""
        logger.debug("Formatting commit message")

        commit_msg = f"{summary.commit_type}: {summary.overall_summary}"

        if summary.detailed_description:
            commit_msg += f"\n\n{summary.detailed_description}"

        if summary.file_changes:
            commit_msg += "\n\nFiles changed:"
            for change in summary.file_changes:
                commit_msg += f"\n- {change.file_path}: {change.summary}"
                if change.lines_added or change.lines_removed:
                    commit_msg += f" (+{change.lines_added}/-{change.lines_removed})"

        return commit_msg

# Typer CLI app
app = typer.Typer(
    name="git-analyzer",
    help="Analyze git changes and generate commit messages using local AI models",
    rich_markup_mode="rich"
)

@app.command()
def analyze(
        unstaged: Annotated[bool, typer.Option("--unstaged", help="Analyze unstaged changes instead of staged")] = False,
        model: Annotated[str, typer.Option("--model", help="Ollama model to use")] = " llama3.1:8b",
        base_url: Annotated[str, typer.Option("--base-url", help="Ollama base URL")] = "http://localhost:11434/v1",
        commit: Annotated[bool, typer.Option("--commit", help="Automatically commit with generated message")] = False,
        output_format: Annotated[str, typer.Option("--output", help="Output format")] = "text",
        verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
):
    """Analyze git changes and generate commit messages"""

    # Set logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    console.print(Panel.fit(
        f"[bold blue]Git Diff Analyzer[/bold blue]\n"
        f"Model: [cyan]{model}[/cyan]\n"
        f"Analyzing: [yellow]{'unstaged' if unstaged else 'staged'}[/yellow] changes",
        border_style="blue"
    ))

    analyzer = GitDiffAnalyzer(model_name=model, base_url=base_url)

    try:
        import asyncio

        # Analyze the changes
        summary = asyncio.run(analyzer.analyze_changes(staged=not unstaged))

        if output_format == "json":
            console.print_json(summary.model_dump_json(indent=2))
        else:
            commit_message = analyzer.format_commit_message(summary)

            console.print("\n[bold green]Generated Commit Message:[/bold green]")
            console.print(Panel(
                commit_message,
                border_style="green",
                title="Commit Message"
            ))

            if commit:
                # Ask for confirmation
                confirm = typer.confirm("\nCommit with this message?")
                if confirm:
                    logger.info("Committing changes with generated message")
                    subprocess.run(["git", "commit", "-m", commit_message], check=True)
                    console.print("✅ [bold green]Changes committed successfully![/bold green]")
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
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )

        console.print("\n[bold green]Available Models:[/bold green]")
        console.print(result.stdout)

    except subprocess.CalledProcessError:
        console.print("❌ [bold red]Failed to list models. Is Ollama running?[/bold red]")
        console.print("Try running: [cyan]ollama serve[/cyan]")
        raise typer.Exit(1)
    except FileNotFoundError:
        console.print("❌ [bold red]Ollama not found. Please install Ollama first.[/bold red]")
        console.print("Visit: [cyan]https://ollama.ai[/cyan]")
        raise typer.Exit(1)

@app.command()
def pull(
        model_name: Annotated[str, typer.Argument(help="Model name to pull")]
):
    """Pull a new Ollama model"""
    console.print(f"[bold blue]Pulling model: {model_name}[/bold blue]")

    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
        ) as progress:
            task = progress.add_task(f"Downloading {model_name}...", total=None)

            result = subprocess.run(
                ["ollama", "pull", model_name],
                check=True
            )

            progress.update(task, completed=True)

        console.print(f"✅ [bold green]Successfully pulled {model_name}[/bold green]")

    except subprocess.CalledProcessError as e:
        console.print(f"❌ [bold red]Failed to pull model: {e}[/bold red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()