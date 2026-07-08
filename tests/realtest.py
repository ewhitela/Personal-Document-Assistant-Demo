"""
Capstone demo GUI: file panel (left) + prompt input (center) + answer output (right).

Layout:
    [ Add Files ]          Ask a question:            Answer:
    file1.pdf [x]          [   text box   ]           [ scrolled  ]
    file2.pdf [x]          [   Send       ]           [ output    ]

Run with: python demo_app.py
Requires: pdva (DocumentIndex, LocalLLM, RAGPipeline) importable, tkinter.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from pdva import DocumentIndex, LocalLLM, RAGPipeline


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Personal Assistant Demo")
        self.root.geometry("1000x600")
        self.root.minsize(820, 480)

        # backend state
        self.index = DocumentIndex()
        self.rag = RAGPipeline(index=self.index, llm=LocalLLM())
        self.files = []  # list of file paths currently added to the index

        self._build_layout()

    # UI Construction

    def _build_layout(self):
        # three columns: files | input | output
        self.root.columnconfigure(0, weight=1, minsize=220)
        self.root.columnconfigure(1, weight=2, minsize=300)
        self.root.columnconfigure(2, weight=2, minsize=300)
        self.root.rowconfigure(0, weight=1)

        self._build_file_panel()
        self._build_input_panel()
        self._build_output_panel()

    def _build_file_panel(self):
        frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        tk.Label(frame, text="Documents", font=("Segoe UI", 11, "bold")).pack(pady=(5, 0))
        self.add_files_button = tk.Button(frame, text="Add Files", command=self.on_add_files)
        self.add_files_button.pack(pady=5, fill=tk.X, padx=5)

        # scrollable list of files
        list_container = tk.Frame(frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        canvas = tk.Canvas(list_container, highlightthickness=0)
        scrollbar = tk.Scrollbar(list_container, orient="vertical", command=canvas.yview)
        self.file_list_frame = tk.Frame(canvas)

        self.file_list_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.file_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.status_label = tk.Label(frame, text="0 documents indexed", fg="gray")
        self.status_label.pack(pady=(0, 5))

    def _build_input_panel(self):
        frame = tk.Frame(self.root)
        frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        tk.Label(frame, text="Ask a question", font=("Segoe UI", 11, "bold")).pack(pady=(5, 0))

        self.prompt_text = tk.Text(frame, height=8, wrap=tk.WORD)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.prompt_text.bind("<Control-Return>", lambda e: self.on_send())

        self.send_button = tk.Button(frame, text="Send", command=self.on_send, width=15)
        self.send_button.pack(pady=(0, 10))

        self.progress_label = tk.Label(frame, text="", fg="gray")
        self.progress_label.pack()

    def _build_output_panel(self):
        frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        tk.Label(frame, text="Answer", font=("Segoe UI", 11, "bold")).pack(pady=(5, 0))
        self.output_box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, state="disabled")
        self.output_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # file panel

    def on_add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[("Documents", "*.pdf *.txt *.md"), ("All files", "*.*")],
        )
        if not paths:
            return
        new_paths = [p for p in paths if p not in self.files]
        if not new_paths:
            return

        self.status_label.config(text="Indexing...")
        self.add_files_button.config(state="disabled")
        self.root.update_idletasks()

        def worker():
            try:
                self.index.add_documents(list(new_paths))
                self.root.after(0, lambda: self._on_files_indexed(new_paths))
            except Exception as e:
                self.root.after(0, lambda e=e: self._on_index_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_files_indexed(self, new_paths):
        for p in new_paths:
            self.files.append(p)
            self._add_file_row(p)
        self.status_label.config(text=f"{len(self.files)} documents indexed")
        self.add_files_button.config(state="normal")

    def _on_index_error(self, error):
        self.status_label.config(text=f"{len(self.files)} documents indexed")
        self.add_files_button.config(state="normal")
        messagebox.showerror("Indexing failed", str(error))

    def _add_file_row(self, path):
        row = tk.Frame(self.file_list_frame)
        row.pack(fill=tk.X, pady=2)

        name = os.path.basename(path)
        label = tk.Label(row, text=name, anchor="w")
        label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        remove_btn = tk.Button(
            row, text="x", fg="red", width=2,
            command=lambda: self.on_remove_file(path, row),
        )
        remove_btn.pack(side=tk.RIGHT)

    def on_remove_file(self, path, row):
        # pdva.DocumentIndex has no per-document delete, only reset() (wipes the
        # whole collection). So removal means: reset, then re-add whatever's left.
        if path in self.files:
            self.files.remove(path)
        row.destroy()
        remaining = list(self.files)

        self.status_label.config(text="Rebuilding index...")
        self.add_files_button.config(state="disabled")
        self.send_button.config(state="disabled")

        def worker():
            try:
                self.index.reset()
                if remaining:
                    self.index.add_documents(remaining)
                self.root.after(0, self._on_rebuild_done)
            except Exception as e:
                self.root.after(0, lambda e=e: self._on_rebuild_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_rebuild_done(self):
        self.status_label.config(text=f"{len(self.files)} documents indexed")
        self.add_files_button.config(state="normal")
        self.send_button.config(state="normal")

    def _on_rebuild_error(self, error):
        self.status_label.config(text=f"{len(self.files)} documents indexed (rebuild failed)")
        self.add_files_button.config(state="normal")
        self.send_button.config(state="normal")
        messagebox.showerror("Rebuild failed", str(error))

    # prompt/send logic

    def on_send(self):
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            return
        if not self.files:
            messagebox.showinfo("No documents", "Add at least one document before asking a question.")
            return

        self.send_button.config(state="disabled")
        self.progress_label.config(text="Thinking...")
        self._append_output(f"Q: {prompt}\n")

        def worker():
            try:
                result = self.rag.answer(prompt)
                answer = getattr(result, "answer", result)
                self.root.after(0, lambda: self._on_answer(answer))
            except Exception as e:
                self.root.after(0, lambda e=e: self._on_answer_error(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_answer(self, answer):
        self._append_output(f"A: {answer}\n\n")
        self.progress_label.config(text="")
        self.send_button.config(state="normal")

    def _on_answer_error(self, error):
        self._append_output(f"[Error: {error}]\n\n")
        self.progress_label.config(text="")
        self.send_button.config(state="normal")

    def _append_output(self, text):
        self.output_box.config(state="normal")
        self.output_box.insert(tk.END, text)
        self.output_box.see(tk.END)
        self.output_box.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = DemoApp(root)
    root.mainloop()