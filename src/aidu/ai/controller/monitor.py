import sys
import termios
import tty
from time import sleep

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from aidu.ai.core.processor_result import ProcessorResult
from aidu.ai.core.context import Context


class Monitor:
    """
    Owns all Rich-based display logic for the controller cockpit.

    Responsibilities:
        - Build and own the Layout tree.
        - Render per-step state into the cockpit panels.
        - Collect interactive footer input via raw keystrokes.
    """

    def __init__(self, context: Context, show_trace: bool = False):
        self.context = context
        self.show_trace = show_trace
        self.layout = self._create_layout(show_trace=show_trace)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    @staticmethod
    def _create_layout(show_trace: bool = False) -> Layout:
        layout = Layout(name="root")

        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=5),
        )

        if show_trace:
            layout["body"].split_row(
                Layout(name="trace", ratio=2),
                Layout(name="runtime", ratio=3),
            )
        else:
            layout["body"].split_column(
                Layout(name="runtime"),
            )

        return layout

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compact_text(value, max_len: int = 36) -> str:
        text = str(value)
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _artifacts_table(self) -> Table:
        table = Table(expand=True, box=ROUNDED, show_lines=False)
        table.add_column("id", no_wrap=True, ratio=2)
        table.add_column("type", no_wrap=True, ratio=1)
        table.add_column("producer", no_wrap=True, ratio=1)
        table.add_column("step", no_wrap=True, ratio=1)
        table.add_column("content", ratio=3)

        artifacts = list(self.context.artifacts.values())[-8:]
        for artifact in artifacts:
            table.add_row(
                self._compact_text(artifact.id, max_len=10),
                self._compact_text(artifact.type, max_len=12),
                self._compact_text(artifact.producer, max_len=12),
                str(artifact.step),
                self._compact_text(artifact.content, max_len=48),
            )

        if not artifacts:
            table.add_row("-", "-", "-", "-", "no artifacts")

        return table

    @staticmethod
    def _result_summary(result: ProcessorResult) -> dict:
        return {
            "artifacts": [a.id for a in result.artifacts],
            "recommendations": [r.model_dump() for r in result.recommendations],
        }

    # ------------------------------------------------------------------
    # Cockpit
    # ------------------------------------------------------------------

    def cockpit(
        self,
        console: Console,
        event,
        result: ProcessorResult,
        auto: bool = False,
        live: Live | None = None,
    ) -> None:
        self.layout["header"].update(
            Panel(
                f"Step {self.context.step}",
                title="Controller",
                border_style="cyan",
            )
        )

        if self.show_trace:
            self.layout["trace"].update(self.context.trace.pretty())

        self.layout["runtime"].update(
            Group(
                self.context.state.pretty(),
                self.context.control.pretty(),
                Panel(
                    self._artifacts_table(),
                    title="Artifacts",
                ),
                Panel(
                    Pretty(self._result_summary(result)),
                    title="Last Result",
                ),
            )
        )

        self.layout["footer"].update(
            Panel(
                f"Current artifacts: {len(self.context.artifacts)}",
                border_style="cyan",
            )
        )

        if live is not None:
            live.update(self.layout, refresh=True)
        else:
            console.clear(home=True)
            console.print(self.layout)

        if auto:
            sleep(1)
        else:
            console.input("[dim]Press Enter for next step...[/dim]\n")

    # ------------------------------------------------------------------
    # Interactive footer input
    # ------------------------------------------------------------------

    def read_footer_input(self, console: Console, live: Live | None = None) -> str:
        if live is None or not sys.stdin.isatty():
            return console.input("[bold green]> [/bold green]")

        prompt = "Type input and press Enter"
        buffer = ""

        while True:
            self.layout["footer"].update(
                Panel(
                    f"{prompt}\n[bold green]> [/bold green]{buffer}",
                    title="Input",
                    border_style="green",
                )
            )
            live.update(self.layout, refresh=True)

            ch = self._read_key()
            if ch in ("\r", "\n"):
                break
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\x7f", "\b"):
                buffer = buffer[:-1]
                continue
            if ch.startswith("\x1b"):
                # Ignore escape sequences (arrows, function keys, etc.).
                continue
            if ch.isprintable():
                buffer += ch

        return buffer

    @staticmethod
    def _read_key() -> str:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
