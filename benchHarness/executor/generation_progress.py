"""Sample completion and active stages; rendering never makes model requests."""
from __future__ import annotations

from collections import Counter
import sys
from threading import RLock
import time


class GenerationProgress:
    def __init__(self, subtask_id, total, results=(), cached=0, stream=None, interactive=None):
        self.subtask_id, self.total = subtask_id, total
        self.completed = len(results)
        self.accepted = sum(r.get('status') == 'accepted' for r in results)
        self.rejected = self.completed - self.accepted
        self.cached = cached
        self.active = {}
        self.stream = sys.stdout if stream is None else stream
        self.interactive = (getattr(self.stream, 'isatty', lambda: False)()
                            if interactive is None else interactive)
        self.lock = RLock()
        self.live = None
        self.started = time.monotonic()

    def __enter__(self):
        if self.interactive:
            from rich.console import Console
            from rich.console import Group
            from rich.text import Text
            from rich.progress import (Progress, TextColumn, BarColumn, TaskProgressColumn,
                                       MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn)
            class DetailedProgress(Progress):
                def get_renderable(self):
                    tasks = self.tasks
                    status = tasks[0].fields.get('status', '') if tasks else ''
                    return Group(super().get_renderable(), Text(status))

            self.live = DetailedProgress(
                TextColumn('{task.description}', markup=False), BarColumn(bar_width=18), TaskProgressColumn(),
                MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
                console=Console(file=self.stream, markup=False), refresh_per_second=2)
            self.task = self.live.add_task(f'Generator[{self.subtask_id}]', total=self.total,
                                           completed=self.completed, status=self._status())
            self.live.start()
        else:
            self._render('start')
        return self

    def _status(self):
        stages = ','.join(f'{name}:{count}' for name, count in sorted(Counter(self.active.values()).items()))
        return (f'accepted={self.accepted} rejected={self.rejected} cached={self.cached} '
                f'active={stages or "-"}')

    def _render(self, event):
        if self.live is not None:
            self.live.update(self.task, completed=self.completed, status=self._status())
        else:
            fraction = self.completed / self.total if self.total else 1
            filled = min(20, int(fraction * 20))
            bar = '#' * filled + '-' * (20 - filled)
            elapsed = time.monotonic() - self.started
            print(f'[Generation][{self.subtask_id}] [{bar}] {self.completed}/{self.total} '
                  f'({fraction:.0%}) {self._status()} elapsed={elapsed:.1f}s; {event}',
                  file=self.stream, flush=True)

    def stage(self, key, name):
        with self.lock:
            self.active[key] = name
            self._render(f'sample={key[:8]} stage={name}')

    def finish(self, key, result):
        with self.lock:
            self.active.pop(key, None)
            self.completed += 1
            if result.get('status') == 'accepted':
                self.accepted += 1
            else:
                self.rejected += 1
            self._render(f'sample={key[:8]} {result.get("status", "failed")}')

    def __exit__(self, exc_type, exc_value, traceback):
        with self.lock:
            if exc_type is not None:
                # Aborted/queued samples must not be presented as completed.
                if self.live is not None:
                    self.live.update(self.task, status='stopped; ' + self._status())
                else:
                    self._render('stopped')
            elif self.live is None:
                self._render('finished')
            if self.live is not None:
                self.live.stop()
        return False
