"""File dialogs shared by the command-line tools.

tkinter is imported only when a dialog is actually shown, so the tools run
without Tk as long as all arguments are given on the command line.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import tkinter as tk

FileTypes = Sequence[tuple[str, str]]


class LazyRoot:
    """A hidden Tk root that is only created when a dialog needs it."""

    def __init__(self) -> None:
        self._root: tk.Tk | None = None

    def get(self) -> tk.Tk:
        """Return the root, creating it on first use."""
        if self._root is None:
            import tkinter
            root = tkinter.Tk()
            root.withdraw()
            self._root = root
            return root
        return self._root

    def destroy(self) -> None:
        """Destroy the root, if it was created."""
        if self._root is not None:
            self._root.destroy()
            self._root = None


def ask_open_file(
    root: LazyRoot, title: str, file_types: FileTypes, directory: Path = Path(".")
) -> Path | None:
    """Ask for an existing file. Returns None if cancelled."""
    from tkinter import filedialog

    path = filedialog.askopenfilename(
        parent=root.get(),
        title=title,
        initialdir=str(directory),
        filetypes=(*file_types, ("All files", "*.*")),
    )
    return Path(path) if path else None


def ask_directory(root: LazyRoot, title: str, directory: Path = Path(".")) -> Path | None:
    """Ask for a directory. Returns None if cancelled."""
    from tkinter import filedialog

    path = filedialog.askdirectory(parent=root.get(), title=title, initialdir=str(directory))
    return Path(path) if path else None


def ask_save_file(
    root: LazyRoot,
    title: str,
    file_types: FileTypes,
    default_name: str,
    directory: Path = Path("."),
) -> Path | None:
    """Ask where to save a file. Returns None if cancelled.

    The extension of `default_name` is appended if the user omits one.
    """
    from tkinter import filedialog

    path = filedialog.asksaveasfilename(
        parent=root.get(),
        title=title,
        initialdir=str(directory),
        initialfile=default_name,
        defaultextension=Path(default_name).suffix,
        filetypes=(*file_types, ("All files", "*.*")),
    )
    return Path(path) if path else None