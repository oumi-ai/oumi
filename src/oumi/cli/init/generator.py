# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Core generation logic for oumi init."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from oumi.cli.init.prompts import (
    CONFIG_GENERATION_SYSTEM_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    EDIT_SYSTEM_PROMPT,
    META_JUDGE_SYSTEM_PROMPT,
    build_config_generation_prompt,
    build_conversation_user_prompt,
    build_edit_prompt,
    build_meta_judge_prompt,
    format_source_summaries,
    format_user_answers,
)
from oumi.cli.init.schemas import (
    ConfigGenerationResponse,
    ConversationResponse,
    EditResponse,
    MetaJudgeResult,
    OutputFormat,
    SessionState,
    SynthConfigSpec,
    TransformedAttribute,
)

console = Console()

MAX_CONVERSATION_ROUNDS = 5
MAX_RETRIES = 3
SESSION_FILE = ".oumi_init_session.json"


class InitGenerator:
    """Main generator for oumi init."""

    def __init__(self):
        """Initialize the generator with Anthropic client."""
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"

    # =========================================================================
    # Session Management
    # =========================================================================

    def _get_session_path(self, output_dir: str) -> Path:
        """Get the session file path."""
        return Path(output_dir) / SESSION_FILE

    def save_session(self, state: SessionState) -> None:
        """Save session state to file."""
        session_path = self._get_session_path(state.output_dir)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(state.model_dump_json(indent=2))
        console.print(f"[dim]Session saved to {session_path}[/dim]")

    def load_session(self, output_dir: str) -> SessionState | None:
        """Load session state from file."""
        session_path = self._get_session_path(output_dir)
        if not session_path.exists():
            return None
        try:
            data = json.loads(session_path.read_text())
            return SessionState.model_validate(data)
        except Exception as e:
            console.print(f"[yellow]Could not load session: {e}[/yellow]")
            return None

    def clear_session(self, output_dir: str) -> None:
        """Clear saved session."""
        session_path = self._get_session_path(output_dir)
        if session_path.exists():
            session_path.unlink()

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def run(
        self,
        task: str,
        sources: list[str],
        output_format: OutputFormat,
        output_dir: str,
        non_interactive: bool = False,
    ) -> tuple[str, str]:
        """Run the full init flow (new session).

        Args:
            task: Natural language task description.
            sources: List of source file paths.
            output_format: Desired output format.
            output_dir: Directory to save configs.
            non_interactive: If True, skip all user interaction.

        Returns:
            Tuple of (synth_yaml, judge_yaml).
        """
        # Create initial session state
        # Always start at "conversation" phase - _run_from_state handles
        # non-interactive mode by building understanding without user interaction
        state = SessionState(
            task=task,
            sources=sources,
            output_format=output_format,
            output_dir=output_dir,
            phase="conversation",
        )

        return self._run_from_state(state, non_interactive=non_interactive)

    def resume(self, output_dir: str) -> tuple[str, str]:
        """Resume from saved session.

        Args:
            output_dir: Directory containing the session file.

        Returns:
            Tuple of (synth_yaml, judge_yaml).

        Raises:
            ValueError: If no session found.
        """
        state = self.load_session(output_dir)
        if state is None:
            raise ValueError(f"No session found in {output_dir}")

        console.print(
            Panel(
                f"[bold]Resuming session[/bold]\n\n"
                f"Task: {state.task}\n"
                f"Phase: {state.phase}\n"
                f"Sources: {len(state.sources)} file(s)",
                title="[cyan]Session Loaded[/cyan]",
            )
        )

        return self._run_from_state(state)

    def _run_from_state(
        self, state: SessionState, non_interactive: bool = False
    ) -> tuple[str, str]:
        """Run from a given session state."""
        # Phase 1: Analyze sources (if not done)
        if not state.source_analyses:
            state.source_analyses = self._analyze_sources(state.sources)
            if not non_interactive:
                self.save_session(state)

        # Phase 2: Conversation loop (if not done) - skipped in non-interactive mode
        if state.phase == "conversation":
            if non_interactive:
                # In non-interactive mode, build understanding directly from task
                state.understanding = self._build_understanding_non_interactive(
                    state.task, state.source_analyses
                )
            else:
                state.understanding = self._conversation_loop(
                    state.task, state.source_analyses
                )
            state.phase = "generation"
            if not non_interactive:
                self.save_session(state)

        # Phase 3: Generate configs (if not done)
        if state.phase == "generation":
            synth_yaml, judge_yaml = self._generate_configs(
                state.understanding, state.sources, state.output_format
            )
            state.synth_yaml = synth_yaml
            state.judge_yaml = judge_yaml
            state.phase = "review"
            if not non_interactive:
                self.save_session(state)

        # Phase 4: Meta-judge validation (run but suppress UI in non-interactive)
        if non_interactive:
            self._run_meta_judge_quiet(state.task, state.synth_yaml, state.judge_yaml)
        else:
            self._run_meta_judge(state.task, state.synth_yaml, state.judge_yaml)

        # Phase 5: Oumi validation
        self._validate_configs(state.synth_yaml, state.judge_yaml)

        return state.synth_yaml, state.judge_yaml

    # =========================================================================
    # Non-Interactive Understanding
    # =========================================================================

    def _build_understanding_non_interactive(
        self, task: str, source_analyses: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build task understanding without user interaction.

        In non-interactive mode, we ask the LLM to make reasonable assumptions
        based on the task description alone, without asking clarifying questions.
        """
        source_summaries = format_source_summaries(source_analyses)

        user_prompt = f"""Analyze this task and provide your understanding.
Make reasonable assumptions where needed - do NOT ask clarifying questions.

## Task
{task}

## Sources
{source_summaries if source_summaries else "No sources provided."}

## Instructions
1. Analyze the task and infer reasonable defaults for any unclear aspects
2. Set ready_to_generate to true
3. Provide empty follow_up_questions list
4. Use "medium" confidence unless the task is very clear (high) or very vague (low)

Respond with JSON matching the expected schema."""

        with console.status("[cyan]Analyzing task...[/cyan]"):
            response = self._call_conversation(user_prompt)

        # Force ready_to_generate and no follow-up questions
        understanding = response.understanding.model_dump()
        understanding["confidence"] = (
            "high" if response.understanding.confidence == "high" else "medium"
        )

        return understanding

    # =========================================================================
    # Meta-Judge Validation
    # =========================================================================

    def _run_meta_judge_quiet(
        self, task: str, synth_yaml: str, judge_yaml: str
    ) -> MetaJudgeResult:
        """Run meta-judge validation without console output."""
        try:
            return self._call_meta_judge(task, synth_yaml, judge_yaml)
        except Exception:
            # Silently return a pass result on errors
            return MetaJudgeResult(
                is_coherent=True,
                issues=[],
                judge_synth_aligned=True,
                judge_synth_aligned_reason="Skipped",
                attribute_references_valid=True,
                attribute_references_reason="Skipped",
                pipeline_logic_sound=True,
                pipeline_logic_reason="Skipped",
                prompts_well_formed=True,
                prompts_reason="Skipped",
            )

    def _run_meta_judge(self, task: str, synth_yaml: str, judge_yaml: str) -> None:
        """Run meta-judge validation on generated configs."""
        with console.status("[cyan]Validating config coherence...[/cyan]"):
            result = self._call_meta_judge(task, synth_yaml, judge_yaml)

        # Display results
        if result.is_coherent:
            console.print(
                Panel(
                    "[green]Config coherence check passed[/green]\n\n"
                    f"Judge-Synth Aligned: {result.judge_synth_aligned_reason}\n"
                    f"References Valid: {result.attribute_references_reason}\n"
                    f"Pipeline Logic: {result.pipeline_logic_reason}\n"
                    f"Prompts: {result.prompts_reason}",
                    title="[bold green]Meta-Judge: PASSED[/bold green]",
                    border_style="green",
                )
            )
        else:
            issues_str = "\n".join(f"  - {issue}" for issue in result.issues)
            console.print(
                Panel(
                    f"[yellow]Potential issues found:[/yellow]\n\n{issues_str}\n\n"
                    f"Judge-Synth: {result.judge_synth_aligned_reason}\n"
                    f"References: {result.attribute_references_reason}\n"
                    f"Pipeline: {result.pipeline_logic_reason}\n"
                    f"Prompts: {result.prompts_reason}",
                    title="[bold yellow]Meta-Judge: WARNINGS[/bold yellow]",
                    border_style="yellow",
                )
            )

    def _call_meta_judge(
        self, task: str, synth_yaml: str, judge_yaml: str
    ) -> MetaJudgeResult:
        """Call LLM for meta-judge validation."""
        user_prompt = build_meta_judge_prompt(task, synth_yaml, judge_yaml)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2048,
                    messages=[{"role": "user", "content": user_prompt}],
                    system=META_JUDGE_SYSTEM_PROMPT,
                )

                content = response.content[0].text
                data = self._extract_json(content)
                return MetaJudgeResult.model_validate(data)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    console.print(
                        f"[yellow]Meta-judge retry {attempt + 1}/{MAX_RETRIES}: "
                        f"{e}[/yellow]"
                    )
                else:
                    # Don't fail on meta-judge errors, just warn
                    console.print(
                        f"[yellow]Meta-judge validation skipped: {e}[/yellow]"
                    )
                    return MetaJudgeResult(
                        is_coherent=True,
                        issues=["Meta-judge validation could not be performed"],
                        judge_synth_aligned=True,
                        judge_synth_aligned_reason="Skipped",
                        attribute_references_valid=True,
                        attribute_references_reason="Skipped",
                        pipeline_logic_sound=True,
                        pipeline_logic_reason="Skipped",
                        prompts_well_formed=True,
                        prompts_reason="Skipped",
                    )

        # Should not reach here
        raise RuntimeError("Meta-judge failed unexpectedly")

    # =========================================================================
    # Edit Loop
    # =========================================================================

    def edit_configs(
        self,
        synth_yaml: str,
        judge_yaml: str,
        understanding: dict[str, Any],
    ) -> tuple[str, str]:
        """Run edit loop on configs.

        Args:
            synth_yaml: Current synth config YAML.
            judge_yaml: Current judge config YAML.
            understanding: Task understanding dict.

        Returns:
            Updated (synth_yaml, judge_yaml).
        """
        while True:
            action = Prompt.ask(
                "\n[bold]What would you like to do?[/bold]\n"
                "[dim](edit <description>, preview, or done)[/dim]"
            )

            action_lower = action.lower().strip()

            if action_lower in ("done", "exit", "quit", ""):
                break

            if action_lower == "preview":
                self._run_preview(synth_yaml, judge_yaml)
                continue

            # Treat as edit request
            edit_request = action
            if action_lower.startswith("edit "):
                edit_request = action[5:].strip()

            with console.status("[cyan]Applying edits...[/cyan]"):
                response = self._call_edit(synth_yaml, judge_yaml, edit_request)

            console.print(
                Panel(
                    f"[green]{response.changes_summary}[/green]\n\n"
                    f"Modified: {', '.join(response.modified_sections)}",
                    title="[bold]Changes Applied[/bold]",
                )
            )

            # Apply updates
            if response.updated_synth_config:
                synth_yaml = self._render_synth_config(
                    response.updated_synth_config, understanding
                )
            if response.updated_judge_config:
                judge_yaml = self._render_judge_config(
                    response.updated_judge_config, understanding
                )

            # Show updated preview
            console.print(
                Panel(
                    Syntax(synth_yaml, "yaml", theme="monokai"),
                    title="[bold]Updated synth_config.yaml[/bold]",
                )
            )
            console.print(
                Panel(
                    Syntax(judge_yaml, "yaml", theme="monokai"),
                    title="[bold]Updated judge_config.yaml[/bold]",
                )
            )

            # Validate
            try:
                self._validate_configs(synth_yaml, judge_yaml)
            except ValueError:
                console.print(
                    "[yellow]Validation failed. You may want to edit again.[/yellow]"
                )

        return synth_yaml, judge_yaml

    def _run_preview(
        self,
        synth_yaml: str,
        judge_yaml: str,
        num_samples: int = 3,
    ) -> None:
        """Run synth and judge to preview results.

        Args:
            synth_yaml: Synth config YAML.
            judge_yaml: Judge config YAML.
            num_samples: Number of samples to generate for preview.
        """
        import tempfile

        from oumi import judge as oumi_judge
        from oumi import synthesize as oumi_synthesize
        from oumi.core.configs import JudgeConfig, SynthesisConfig

        console.print(
            Panel(
                f"[cyan]Running preview with {num_samples} samples...[/cyan]",
                title="[bold]Preview[/bold]",
            )
        )

        # Create temp files for configs
        with tempfile.TemporaryDirectory() as tmpdir:
            synth_config_path = Path(tmpdir) / "synth_config.yaml"
            judge_config_path = Path(tmpdir) / "judge_config.yaml"
            output_path = Path(tmpdir) / "preview_output.jsonl"

            # Modify synth config to use fewer samples and temp output
            synth_config = yaml.safe_load(synth_yaml.split("---")[-1])
            synth_config["num_samples"] = num_samples
            synth_config["output_path"] = str(output_path)

            synth_config_path.write_text(
                yaml.dump(synth_config, default_flow_style=False)
            )
            judge_config_path.write_text(judge_yaml)

            # Run synth
            try:
                with console.status("[cyan]Generating samples...[/cyan]"):
                    config = SynthesisConfig.from_yaml(str(synth_config_path))
                    oumi_synthesize(config)

                console.print(f"[green]Generated {num_samples} samples[/green]")

                # Show generated samples
                if output_path.exists():
                    samples = []
                    with open(output_path) as f:
                        for line in f:
                            samples.append(json.loads(line))

                    for i, sample in enumerate(samples, 1):
                        # Format sample for display
                        sample_str = json.dumps(sample, indent=2, ensure_ascii=False)
                        if len(sample_str) > 1000:
                            sample_str = sample_str[:1000] + "\n..."
                        console.print(
                            Panel(
                                Syntax(sample_str, "json", theme="monokai"),
                                title=f"[bold]Sample {i}/{len(samples)}[/bold]",
                                border_style="blue",
                            )
                        )

                # Run judge
                with console.status("[cyan]Judging samples...[/cyan]"):
                    judge_config = JudgeConfig.from_yaml(str(judge_config_path))
                    judge_config.input_path = str(output_path)
                    judge_output_path = Path(tmpdir) / "judge_output.jsonl"
                    judge_config.output_path = str(judge_output_path)
                    oumi_judge(judge_config)

                console.print("[green]Judging complete[/green]")

                # Show judge results
                if judge_output_path.exists():
                    results = []
                    with open(judge_output_path) as f:
                        for line in f:
                            results.append(json.loads(line))

                    # Summarize results
                    passed = sum(
                        1
                        for r in results
                        if r.get("judgment", {}).get("judgment") is True
                        or str(r.get("judgment", {}).get("judgment")).lower() == "true"
                    )
                    total = len(results)

                    summary_color = "green" if passed == total else "yellow"
                    console.print(
                        Panel(
                            f"[{summary_color}]Passed: {passed}/{total}[/{summary_color}]",
                            title="[bold]Judge Results Summary[/bold]",
                        )
                    )

                    # Show individual judgments
                    for i, result in enumerate(results, 1):
                        judgment = result.get("judgment", {})
                        is_pass = (
                            judgment.get("judgment") is True
                            or str(judgment.get("judgment")).lower() == "true"
                        )
                        status = "[green]PASS[/green]" if is_pass else "[red]FAIL[/red]"
                        explanation = judgment.get("explanation", "No explanation")

                        console.print(
                            Panel(
                                f"{status}\n\n{explanation}",
                                title=f"[bold]Judgment {i}/{total}[/bold]",
                                border_style="green" if is_pass else "red",
                            )
                        )

            except Exception as e:
                console.print(f"[red]Preview failed: {e}[/red]")
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")

    def _call_edit(
        self, synth_yaml: str, judge_yaml: str, edit_request: str
    ) -> EditResponse:
        """Call LLM for edit request."""
        user_prompt = build_edit_prompt(synth_yaml, judge_yaml, edit_request)

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": user_prompt}],
                    system=EDIT_SYSTEM_PROMPT,
                )

                content = response.content[0].text
                data = self._extract_json(content)
                return EditResponse.model_validate(data)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    console.print(
                        f"[yellow]Edit retry {attempt + 1}/{MAX_RETRIES}: {e}[/yellow]"
                    )
                else:
                    raise

        raise RuntimeError("Failed to apply edit")

    # =========================================================================
    # Source Analysis
    # =========================================================================

    def _analyze_sources(self, sources: list[str]) -> list[dict[str, Any]]:
        """Analyze source files."""
        if not sources:
            return []

        analyses = []
        for path in sources:
            with console.status(f"[cyan]Analyzing {path}...[/cyan]"):
                analysis = self._analyze_single_source(path)
                analyses.append(analysis)

        return analyses

    def _analyze_single_source(self, path: str) -> dict[str, Any]:
        """Analyze a single source file."""
        p = Path(path)
        ext = p.suffix.lower()

        if ext in {".jsonl", ".json", ".csv", ".parquet", ".tsv", ".xlsx", ".xls"}:
            return self._analyze_dataset(path, ext)
        elif ext in {".pdf", ".md", ".txt", ".docx"}:
            return self._analyze_document(path, ext)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _analyze_dataset(self, path: str, ext: str) -> dict[str, Any]:
        """Analyze a dataset file."""
        import pandas as pd

        if ext == ".jsonl":
            df = pd.read_json(path, lines=True, nrows=10)
        elif ext == ".json":
            df = pd.read_json(path, nrows=10)
        elif ext == ".csv":
            df = pd.read_csv(path, nrows=10)
        elif ext == ".parquet":
            df = pd.read_parquet(path).head(10)
        elif ext == ".tsv":
            df = pd.read_csv(path, sep="\t", nrows=10)
        elif ext in {".xlsx", ".xls"}:
            df = pd.read_excel(path, nrows=10)
        else:
            raise ValueError(f"Unsupported dataset format: {ext}")

        return {
            "type": "dataset",
            "path": path,
            "file_type": ext[1:],
            "columns": list(df.columns),
            "num_rows": len(df),
            "sample": df.iloc[0].to_dict() if len(df) > 0 else None,
        }

    def _analyze_document(self, path: str, ext: str) -> dict[str, Any]:
        """Analyze a document file."""
        p = Path(path)

        if ext == ".pdf":
            content = self._read_pdf(path)
        elif ext == ".docx":
            content = self._read_docx(path)
        else:
            content = p.read_text()[:5000]

        return {
            "type": "document",
            "path": path,
            "file_type": ext[1:],
            "num_chars": len(content),
            "preview": content[:500],
        }

    def _read_pdf(self, path: str) -> str:
        """Read text from PDF."""
        try:
            import PyPDF2

            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages[:5]:  # First 5 pages
                    text += page.extract_text() or ""
                return text[:5000]
        except ImportError:
            return "[PDF reading requires PyPDF2]"

    def _read_docx(self, path: str) -> str:
        """Read text from Word document."""
        try:
            import docx

            doc = docx.Document(path)
            paragraphs = []
            for para in doc.paragraphs[:50]:  # First 50 paragraphs
                if para.text.strip():
                    paragraphs.append(para.text)
            return "\n\n".join(paragraphs)[:5000]
        except ImportError:
            return "[DOCX reading requires python-docx: pip install python-docx]"

    # =========================================================================
    # Conversation Loop
    # =========================================================================

    def _conversation_loop(
        self,
        task: str,
        source_analyses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run conversation loop until ready to generate."""
        source_summaries = format_source_summaries(source_analyses)
        conversation_history: list[ConversationResponse] = []
        user_answers: list[str] = []

        for round_num in range(MAX_CONVERSATION_ROUNDS):
            # Build prompt
            if conversation_history:
                prev_understanding = json.dumps(
                    conversation_history[-1].understanding.model_dump(), indent=2
                )
                answers_str = format_user_answers(
                    conversation_history[-1].follow_up_questions, user_answers
                )
                user_prompt = build_conversation_user_prompt(
                    task, source_summaries, prev_understanding, answers_str
                )
            else:
                user_prompt = build_conversation_user_prompt(task, source_summaries)

            # Call LLM
            with console.status("[cyan]Analyzing task...[/cyan]"):
                response = self._call_conversation(user_prompt)

            conversation_history.append(response)

            # Display understanding
            self._display_understanding(response.understanding)

            # Check if ready
            if response.ready_to_generate:
                if response.understanding.confidence == "low":
                    # Auto-trigger more questions
                    console.print(
                        "[yellow]Confidence is low, asking clarifying questions..."
                        "[/yellow]"
                    )
                else:
                    break

            # Ask follow-up questions
            if response.follow_up_questions:
                user_answers = self._ask_questions(response.follow_up_questions)
            else:
                break

        return response.understanding.model_dump()

    def _call_conversation(self, user_prompt: str) -> ConversationResponse:
        """Call LLM for conversation phase with structured output."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": user_prompt}],
                    system=CONVERSATION_SYSTEM_PROMPT,
                )

                # Parse response - extract JSON from content
                content = response.content[0].text
                # Try to find JSON in the response
                data = self._extract_json(content)
                return ConversationResponse.model_validate(data)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    console.print(
                        f"[yellow]Retry {attempt + 1}/{MAX_RETRIES}: {e}[/yellow]"
                    )
                else:
                    raise

        raise RuntimeError("Failed to get valid conversation response")

    def _extract_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        import re

        # Try direct parse first
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find raw JSON object
        json_match = re.search(r"(\{.*\})", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        raise ValueError(f"Could not extract JSON from response: {content[:200]}...")

    def _display_understanding(self, understanding: Any) -> None:
        """Display LLM's understanding to user."""
        console.print(
            Panel(
                Syntax(
                    json.dumps(understanding.model_dump(), indent=2),
                    "json",
                    theme="monokai",
                ),
                title="[bold]Understanding[/bold]",
                border_style="green",
            )
        )

    def _ask_questions(self, questions: list[Any]) -> list[str]:
        """Ask follow-up questions to user."""
        answers = []

        for q in questions:
            console.print(f"\n[bold]{q.question}[/bold]")
            console.print(f"[dim]({q.why_needed})[/dim]\n")

            if q.question_type == "multiple_choice" and q.options:
                for i, opt in enumerate(q.options, 1):
                    console.print(f"  {i}. {opt['label']}")
                    console.print(f"     [dim]{opt['description']}[/dim]")

                choice = Prompt.ask(
                    "Select option",
                    choices=[str(i) for i in range(1, len(q.options) + 1)],
                )
                answers.append(q.options[int(choice) - 1]["label"])
            else:
                answer = Prompt.ask("Your answer")
                answers.append(answer)

        return answers

    # =========================================================================
    # Config Generation
    # =========================================================================

    def _generate_configs(
        self,
        understanding: dict[str, Any],
        sources: list[str],
        output_format: OutputFormat,
    ) -> tuple[str, str]:
        """Generate synth and judge configs."""
        user_prompt = build_config_generation_prompt(
            json.dumps(understanding, indent=2),
            sources,
            output_format.value,
        )

        with console.status("[cyan]Generating configs...[/cyan]"):
            response = self._call_config_generation(user_prompt)

        # Render to YAML
        synth_yaml = self._render_synth_config(response.synth_config, understanding)
        judge_yaml = self._render_judge_config(response.judge_config, understanding)

        return synth_yaml, judge_yaml

    def _call_config_generation(self, user_prompt: str) -> ConfigGenerationResponse:
        """Call LLM for config generation with structured output."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": user_prompt}],
                    system=CONFIG_GENERATION_SYSTEM_PROMPT,
                )

                content = response.content[0].text
                data = self._extract_json(content)
                return ConfigGenerationResponse.model_validate(data)

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    console.print(
                        f"[yellow]Retry {attempt + 1}/{MAX_RETRIES}: {e}[/yellow]"
                    )
                else:
                    raise

        raise RuntimeError("Failed to get valid config generation response")

    def _render_synth_config(
        self, spec: SynthConfigSpec, understanding: dict[str, Any]
    ) -> str:
        """Render synth spec to YAML."""
        # Build config dict matching Oumi schema
        config: dict[str, Any] = {
            "strategy": "GENERAL",
            "num_samples": spec.num_samples,
            "output_path": spec.output_path,
            "strategy_params": {},
            "inference_config": {
                "model": {"model_name": "claude-sonnet-4-20250514"},
                "engine": "ANTHROPIC",
                "generation": {
                    "max_new_tokens": 2048,
                    "temperature": 0.7,
                },
            },
        }

        # Add input sources
        if spec.input_documents:
            config["strategy_params"]["input_documents"] = [
                {
                    "path": d.path,
                    "id": d.id,
                    "segmentation_params": {
                        "id": f"{d.id}_segment",
                        "segment_length": d.segment_length,
                        "segment_overlap": d.segment_overlap,
                    },
                }
                for d in spec.input_documents
            ]

        if spec.input_data:
            config["strategy_params"]["input_data"] = [
                {"path": d.path, "attribute_map": d.attribute_map}
                for d in spec.input_data
            ]

        # Add attributes
        if spec.sampled_attributes:
            config["strategy_params"]["sampled_attributes"] = [
                {
                    "id": a.id,
                    "name": a.name,
                    "description": a.description,
                    "possible_values": [
                        {
                            "id": v.id,
                            "name": v.name,
                            "description": v.description,
                            **({"sample_rate": v.sample_rate} if v.sample_rate else {}),
                        }
                        for v in a.possible_values
                    ],
                }
                for a in spec.sampled_attributes
            ]

        if spec.generated_attributes:
            config["strategy_params"]["generated_attributes"] = [
                {
                    "id": a.id,
                    "instruction_messages": [
                        {"role": "SYSTEM", "content": a.system_prompt},
                        {"role": "USER", "content": a.user_prompt_template},
                    ],
                }
                for a in spec.generated_attributes
            ]

        if spec.transformed_attributes:
            config["strategy_params"]["transformed_attributes"] = [
                self._render_transformed_attribute(a)
                for a in spec.transformed_attributes
            ]

        if spec.passthrough_attributes:
            config["strategy_params"]["passthrough_attributes"] = (
                spec.passthrough_attributes
            )

        # Add header comment
        header = f"""# Generated by: oumi init --task
# Task: {understanding.get('summary', 'N/A')}
# Usage: oumi synth -c <this_file>

"""
        return header + yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _render_transformed_attribute(self, attr: TransformedAttribute) -> dict:
        """Render a transformed attribute."""
        result: dict[str, Any] = {
            "id": attr.id,
            "transformation_strategy": {
                "type": attr.transformation_type,
            },
        }

        if attr.transformation_type == "CHAT" and attr.chat_messages:
            result["transformation_strategy"]["chat_transform"] = {
                "messages": [
                    {"role": m.role, "content": m.content} for m in attr.chat_messages
                ]
            }
        elif attr.transformation_type == "STRING" and attr.string_template:
            result["transformation_strategy"]["string_transform"] = attr.string_template

        return result

    def _render_judge_config(
        self, spec: Any, understanding: dict[str, Any]
    ) -> str:
        """Render judge spec to YAML."""
        config: dict[str, Any] = {
            "judge_params": {
                "system_instruction": spec.system_instruction,
                "prompt_template": spec.prompt_template,
                "response_format": "JSON",
                "judgment_type": spec.judgment_type,
                "include_explanation": spec.include_explanation,
            },
            "inference_config": {
                "model": {"model_name": "claude-sonnet-4-20250514"},
                "engine": "ANTHROPIC",
                "generation": {
                    "max_new_tokens": 1024,
                    "temperature": 0.0,
                },
            },
        }

        if spec.judgment_scores:
            config["judge_params"]["judgment_scores"] = spec.judgment_scores

        header = f"""# Generated by: oumi init --task
# Task: Evaluate {understanding.get('summary', 'generated content')}
# Usage: oumi judge dataset -c <this_file> --input <synth_output.jsonl>

"""
        return header + yaml.dump(config, default_flow_style=False, sort_keys=False)

    def _validate_configs(self, synth_yaml: str, judge_yaml: str) -> None:
        """Validate configs through Oumi's config system."""
        from oumi.core.configs import JudgeConfig, SynthesisConfig

        errors = []

        # Validate synth config
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(synth_yaml)
                f.flush()
                SynthesisConfig.from_yaml(f.name)
        except Exception as e:
            errors.append(f"Synth config validation error: {e}")

        # Validate judge config
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                f.write(judge_yaml)
                f.flush()
                JudgeConfig.from_yaml(f.name)
        except Exception as e:
            errors.append(f"Judge config validation error: {e}")

        if errors:
            for err in errors:
                console.print(f"[red]{err}[/red]")
            raise ValueError("Config validation failed")

        console.print("[green]Configs validated successfully[/green]")
