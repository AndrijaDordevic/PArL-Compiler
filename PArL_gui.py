from pathlib import Path
import tempfile, traceback, tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk   

from parser_PArL import Parser, ParseError
from semantic_analyser_PArL import SemanticAnalyser
from code_generator_PArL import IRCodeGenerator
from visitor_print_PArL import PrintNodesVisitor     
from visitor_dot_PArL import DotVisitor               


def compile_source(src: str, want_ast: bool):
    # 1. Parser
    try:
        p = Parser(src)
        p.Parse()
        ast = p.ASTroot
    except ParseError as e:
        lineno = getattr(e.token, "line", "?")
        col    = getattr(e.token, "col",  "?" )
        return False, f"Syntax error (line {lineno}, col {col}): {e}", None, None
    except Exception:
        return False, f"Parser crashed:\n{traceback.format_exc()}", None, None

    png_path, pretty = None, None
    if want_ast:
        pretty = PrintNodesVisitor().format_tree(ast)
        try:
            from graphviz import Source
            dot = DotVisitor().generate(ast)
            tmp_png = Path(tempfile.mkdtemp()) / "ast.png"
            Source(dot).render(tmp_png.with_suffix(""), format="png", cleanup=True)
            png_path = tmp_png
        except Exception as e:
            print(f"[Graphviz error] {e}")

    # 2. semantics
    try:
        sema = SemanticAnalyser(); sema.check_semantics(ast)
    except Exception:
        return False, f"Semantic phase crashed:\n{traceback.format_exc()}", png_path, pretty
    if getattr(sema, "error_count", 0):
        return False, f"{sema.error_count} semantic error(s) – aborting", png_path, pretty

    # 3. IR generation
    try:
        ir_gen = IRCodeGenerator(); ir_gen.generate_code(ast)
        return True, ir_gen.code, png_path, pretty
    except Exception:
        return False, f"IR generation crashed:\n{traceback.format_exc()}", png_path, pretty


def open_file():
    fn = filedialog.askopenfilename(filetypes=[("PArL files", "*.parl"), ("All files", "*.*")])
    if fn:
        text_src.delete("1.0", tk.END)
        text_src.insert(tk.END, Path(fn).read_text(encoding="utf-8"))

def save_ir():
    ir = text_out.get("1.0", tk.END).strip()
    if not ir:
        return messagebox.showinfo("Save IR", "Nothing to save.")
    fn = filedialog.asksaveasfilename(defaultextension=".ir",
                                      filetypes=[("IR files", "*.ir"), ("All files", "*.*")])
    if fn:
        Path(fn).write_text(ir, encoding="utf-8")

def show_png(png: Path):
    win = tk.Toplevel(root)
    win.title("AST graph")
    img = Image.open(png)

    # available screen real estate (leave a bit of margin)
    max_w = root.winfo_screenwidth()  - 100
    max_h = root.winfo_screenheight() - 120

    w, h = img.size
    if w > max_w or h > max_h:               
        scale = min(max_w / w, max_h / h)    
        img   = img.resize(
            (int(w * scale), int(h * scale)),
            Image.LANCZOS                   
        )

    photo = ImageTk.PhotoImage(img)
    tk.Label(win, image=photo).pack()
    win.image = photo                           

def run_compile():
    src, want_ast = text_src.get("1.0", tk.END), var_ast.get()
    ok, msg, png, pretty = compile_source(src, want_ast)
    text_out.delete("1.0", tk.END)
    text_out.insert(tk.END, msg)               

    if want_ast and pretty:
        print("\n" + "═" * 70 + "\nAST (text view):\n")
        print(pretty)
        print("\n" + "═" * 70 + "\n")

    if not ok:
        messagebox.showerror("Compilation failed", msg)
    if png:
        show_png(png)

# GUI 
root = tk.Tk(); root.title("PArL mini-compiler")

frame = tk.Frame(root); frame.pack(padx=10, pady=10, fill="both", expand=True)

left = tk.Frame(frame); left.grid(row=0, column=0, sticky="nsew", padx=(0,12))
tk.Label(left, text="Source PArL code").pack(anchor="w")
text_src = scrolledtext.ScrolledText(left, width=70, height=16); text_src.pack()

var_ast = tk.BooleanVar()
tk.Checkbutton(left, text="Show AST (Graphviz) and print text tree to console",
               variable=var_ast).pack(anchor="w", pady=4)

tk.Label(left, text="Output / IR / errors").pack(anchor="w", pady=(8,0))
text_out = scrolledtext.ScrolledText(left, width=70, height=16); text_out.pack()

right = tk.Frame(frame); right.grid(row=0, column=1, sticky="ns")
for label, cmd in [("Open file", open_file),
                   ("Compile",   run_compile),
                   ("Save IR",   save_ir)]:
    tk.Button(right, text=label, width=18, command=cmd).pack(pady=6, anchor="e")

frame.columnconfigure(0, weight=1); frame.rowconfigure(0, weight=1)
left.rowconfigure(1, weight=1); left.rowconfigure(3, weight=1)

root.mainloop()
