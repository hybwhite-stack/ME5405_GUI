"""Visual, per-image preprocessing workflow for the ME5405 OCR project."""
from __future__ import annotations

from copy import deepcopy
import importlib
from pathlib import Path
import sys
import threading
from types import ModuleType
import webbrowser

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk


def _load_available_preprocessing() -> None:
    """Load the current preprocess sources without their stale package imports."""
    package = ModuleType("preprocess")
    package.__path__ = [str(Path(__file__).resolve().parent / "preprocess")]
    package.__package__ = "preprocess"
    sys.modules["preprocess"] = package

    # pipeline.py still imports the deleted module, but the matching operations
    # are removed from the GUI catalog before any recipe can use them.
    morphology = ModuleType("preprocess.morphology")
    sys.modules[morphology.__name__] = morphology
    package.morphology = morphology

    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        basic = importlib.import_module("preprocess.basic")
        pipeline = importlib.import_module("preprocess.pipeline")
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting

    unavailable = {"open", "close"}
    for operation_id in unavailable:
        pipeline.OPERATIONS.pop(operation_id, None)
    for name in ("OPERATIONS", "PipelineUnavailable", "normalize_recipe", "recipe_label", "run_pipeline"):
        setattr(package, name, getattr(pipeline, name))
    package.save_image = basic.save_image


_load_available_preprocessing()

from evaluation import load_ground_truth
from manual_workflow import execute_attempt, export_manual_run, load_recipes, save_ground_truth, save_recipes
from ocr_pipeline import find_images, load_image
from preprocess import OPERATIONS, PipelineUnavailable, normalize_recipe, recipe_label, run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent

# A single palette keeps native dialogs and the workspace visually consistent.
COLORS = {
    "background": "#F3F6FA", "surface": "#FFFFFF", "text": "#17243B",
    "muted": "#65748B", "border": "#DCE4EF", "accent": "#2563EB",
    "accent_hover": "#1D4ED8", "soft": "#EBF2FF", "success": "#16805D",
}


def configure_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    root.configure(background=COLORS["background"])
    root.option_add("*Font", ("Microsoft YaHei UI", 10))
    root.option_add("*TCombobox*Listbox.background", COLORS["surface"])
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent"])
    style.configure(".", font=("Microsoft YaHei UI", 10), foreground=COLORS["text"])
    style.configure("TFrame", background=COLORS["background"])
    style.configure("Card.TFrame", background=COLORS["surface"])
    style.configure("TLabel", background=COLORS["background"])
    style.configure("Card.TLabel", background=COLORS["surface"])
    style.configure("Muted.TLabel", foreground=COLORS["muted"])
    style.configure("CardMuted.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("Microsoft YaHei UI", 9))
    style.configure("Title.TLabel", font=("Microsoft YaHei UI", 20, "bold"))
    style.configure("Heading.TLabel", background=COLORS["surface"], font=("Microsoft YaHei UI", 11, "bold"))
    style.configure("Badge.TLabel", background=COLORS["soft"], foreground=COLORS["accent"], padding=(9, 4), font=("Segoe UI", 9, "bold"))
    style.configure("Success.TLabel", background=COLORS["surface"], foreground=COLORS["success"], padding=(0, 6))
    style.configure("TButton", background=COLORS["surface"], bordercolor=COLORS["border"], lightcolor=COLORS["surface"], darkcolor=COLORS["border"], padding=(10, 7), relief="flat")
    style.map("TButton", background=[("active", COLORS["soft"]), ("disabled", "#EEF1F6")], foreground=[("disabled", "#98A3B4")])
    style.configure("Primary.TButton", background=COLORS["accent"], foreground="white", bordercolor=COLORS["accent"], lightcolor=COLORS["accent"], darkcolor=COLORS["accent"], font=("Microsoft YaHei UI", 10, "bold"))
    style.map("Primary.TButton", background=[("disabled", "#C7D6F0"), ("active", COLORS["accent_hover"])], foreground=[("disabled", "#F8FAFC"), ("!disabled", "white")])
    style.configure("Small.TButton", padding=(7, 5), font=("Microsoft YaHei UI", 9))
    style.configure("TEntry", fieldbackground=COLORS["surface"], bordercolor=COLORS["border"], padding=6)
    style.configure("TCombobox", fieldbackground=COLORS["surface"], background=COLORS["surface"], bordercolor=COLORS["border"], arrowsize=13, padding=5)
    style.map("TCombobox", fieldbackground=[("readonly", COLORS["surface"])], foreground=[("readonly", COLORS["text"])])
    style.configure("TNotebook", background=COLORS["surface"], borderwidth=0, tabmargins=(0, 0, 0, 6))
    style.configure("TNotebook.Tab", background=COLORS["background"], padding=(16, 9), borderwidth=0)
    style.map("TNotebook.Tab", background=[("selected", COLORS["surface"])], foreground=[("selected", COLORS["accent"]), ("!selected", COLORS["muted"])])
    style.configure("TSeparator", background=COLORS["border"])
    style.configure("Vertical.TScrollbar", background="#C5D1E0", troughcolor=COLORS["surface"], borderwidth=0, arrowsize=11)
    style.configure("Horizontal.TScrollbar", background="#C5D1E0", troughcolor=COLORS["surface"], borderwidth=0, arrowsize=11)
    style.configure("TProgressbar", background=COLORS["accent"], troughcolor=COLORS["soft"], borderwidth=0)


def styled_listbox(master, **kwargs) -> tk.Listbox:
    return tk.Listbox(master, background=COLORS["surface"], foreground=COLORS["text"],
                      selectbackground=COLORS["soft"], selectforeground=COLORS["accent"],
                      relief="flat", borderwidth=0, highlightthickness=0,
                      activestyle="none", exportselection=False, **kwargs)


def styled_text(master, **kwargs) -> tk.Text:
    return tk.Text(master, background="#F8FAFD", foreground=COLORS["text"],
                   insertbackground=COLORS["accent"], selectbackground="#DBEAFE",
                   relief="flat", borderwidth=0, highlightthickness=1,
                   highlightbackground=COLORS["border"], highlightcolor=COLORS["accent"],
                   padx=10, pady=10, spacing1=2, spacing3=3, **kwargs)


class ScrollPanel(ttk.Frame):
    """Keep configuration reachable on short windows and high-DPI displays."""

    def __init__(self, master):
        super().__init__(master, style="Card.TFrame")
        self.canvas = tk.Canvas(self, background=COLORS["surface"], highlightthickness=0, width=1, height=1)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.content = ttk.Frame(self.canvas, style="Card.TFrame", padding=(14, 12))
        self.item = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.content.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.item, width=event.width))
        self.canvas.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        # Do not steal scrolling from editable text, lists, or dropdown menus.
        if isinstance(event.widget, (tk.Text, tk.Listbox, ttk.Combobox)):
            return
        x, y = event.x_root, event.y_root
        if (self.canvas.winfo_rootx() <= x < self.canvas.winfo_rootx() + self.canvas.winfo_width()
                and self.canvas.winfo_rooty() <= y < self.canvas.winfo_rooty() + self.canvas.winfo_height()
                and self.content.winfo_reqheight() > self.canvas.winfo_height()):
            self.canvas.yview_scroll(-int(event.delta / 120), "units")
            return "break"


# key, Chinese label, value kind, concise input hint
PARAMETER_FIELDS: dict[str, list[tuple[str, str, str, str]]] = {
    "contrast": [
        ("low_percentile", "低分位点 (%)", "float", "建议 0–10"),
        ("high_percentile", "高分位点 (%)", "float", "建议 90–100"),
    ],
    "clahe": [
        ("clip_limit", "对比度限制", "float", "必须大于 0，常用 1–4"),
        ("tile_grid", "分块大小（宽,高）", "pair", "例如 8,8"),
    ],
    "median": [("kernel", "中值核大小", "odd", "奇数，至少 3")],
    "gaussian": [
        ("kernel", "高斯核大小", "odd", "奇数，至少 3"),
        ("sigma", "Sigma", "positive_float", "必须大于 0"),
    ],
    "bilateral": [
        ("diameter", "邻域直径", "positive_int", "至少 1"),
        ("sigma_color", "颜色 Sigma", "positive_float", "必须大于 0"),
        ("sigma_space", "空间 Sigma", "positive_float", "必须大于 0"),
    ],
    "adaptive": [
        ("block_size", "局部窗口大小", "odd", "奇数，至少 3"),
        ("constant", "阈值常数 C", "float", "可为负数"),
    ],
    "sharpen": [
        ("sigma", "模糊 Sigma", "positive_float", "必须大于 0"),
        ("amount", "锐化强度", "nonnegative_float", "至少 0，常用 0.3–1.5"),
    ],
}


NO_PARAMETER_HELP = {
    "raw": "原图不执行任何处理，因此没有参数。",
    "gray": "灰度化使用标准颜色转换，不需要手工参数。",
    "equalize": "全局直方图均衡由图像直方图自动计算，不需要手工参数。",
    "otsu": "Otsu 阈值由当前图像自动计算，不需要手工输入阈值。",
}


def _photo(image_bgr: np.ndarray, maximum: tuple[int, int]) -> ImageTk.PhotoImage:
    if image_bgr.ndim == 2:
        pil = Image.fromarray(image_bgr)
    else:
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    width, height = pil.size
    scale = min(maximum[0] / max(1, width), maximum[1] / max(1, height), 1.0)
    if scale < 1:
        pil = pil.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(pil)


class ImageView(ttk.Frame):
    def __init__(self, master, title: str):
        super().__init__(master, padding=12, style="Card.TFrame")
        row = ttk.Frame(self, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text=title, style="Heading.TLabel").pack(side="left")
        self.info = ttk.Label(row, style="CardMuted.TLabel")
        self.info.pack(side="right")
        self.canvas = tk.Canvas(self, background="#EDF2F8", highlightthickness=0, width=1, height=1)
        self.canvas.pack(fill="both", expand=True)
        self._photo = None
        self._image = None
        self._resize_job = None
        self.canvas.bind("<Configure>", self._schedule_render)
        self.bind("<Map>", self._schedule_render)

    def show(self, image: np.ndarray | None) -> None:
        self._image = image
        self._schedule_render()

    def _schedule_render(self, _event=None) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(50, self._render)

    def _render(self) -> None:
        self._resize_job = None
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            return
        self.canvas.delete("all")
        if self._image is None:
            self._photo = None
            self.info.configure(text="")
            self.canvas.create_text(width / 2, height / 2, text="暂无图像\n选择图片或运行 OCR 后在这里查看", fill=COLORS["muted"], font=("Microsoft YaHei UI", 11), justify="center")
            return
        self._photo = _photo(self._image, (max(1, width - 24), max(1, height - 24)))
        self.canvas.create_image(width / 2, height / 2, image=self._photo, anchor="center")
        h, w = self._image.shape[:2]
        self.info.configure(text=f"{w} × {h} px  ·  自适应预览")


class ParameterDialog(tk.Toplevel):
    """Edit all numeric parameters of one preprocessing operation at once."""

    def __init__(self, master, operation_id: str, values: dict):
        super().__init__(master)
        spec = OPERATIONS[operation_id]
        self.title(f"设置参数 - {spec.label}")
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)
        self.operation_id = operation_id
        self.result = None
        self.entries: dict[str, tuple[ttk.Entry, str]] = {}

        body = ttk.Frame(self, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=spec.label, font=("Microsoft YaHei", 11, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 4)
        )
        ttk.Label(body, text=spec.description, foreground="#475569").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        for row, (key, label, kind, hint) in enumerate(PARAMETER_FIELDS[operation_id], 2):
            ttk.Label(body, text=label).grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            entry = ttk.Entry(body, width=18)
            entry.insert(0, self._format_value(values[key]))
            entry.grid(row=row, column=1, sticky="ew", pady=4)
            ttk.Label(body, text=hint, foreground="#64748b").grid(row=row, column=2, sticky="w", padx=(8, 0), pady=4)
            self.entries[key] = (entry, kind)

        buttons = ttk.Frame(body)
        buttons.grid(row=2 + len(PARAMETER_FIELDS[operation_id]), column=0, columnspan=3, pady=(12, 0))
        ttk.Button(buttons, text="恢复默认", command=lambda: self._load_values(OPERATIONS[operation_id].default_params)).pack(side="left", padx=4)
        ttk.Button(buttons, text="确定", command=self._save).pack(side="left", padx=4)
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="left", padx=4)
        self.bind("<Return>", lambda _event: self._save())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.update_idletasks()
        self.geometry(f"+{master.winfo_rootx() + 180}+{master.winfo_rooty() + 120}")
        next(iter(self.entries.values()))[0].focus_set()

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, (list, tuple)):
            return ",".join(str(part) for part in value)
        return str(value)

    def _load_values(self, values: dict) -> None:
        for key, (entry, _kind) in self.entries.items():
            entry.delete(0, "end")
            entry.insert(0, self._format_value(values[key]))

    @staticmethod
    def _parse_value(text: str, kind: str):
        text = text.strip().replace("，", ",")
        if kind == "pair":
            parts = [part.strip() for part in text.split(",")]
            if len(parts) != 2:
                raise ValueError("请输入两个整数，例如 3,3")
            value = [int(part) for part in parts]
            if any(part < 1 for part in value):
                raise ValueError("宽和高必须至少为 1")
            return value
        if kind in {"odd", "positive_int"}:
            value = int(text)
            if value < (3 if kind == "odd" else 1):
                raise ValueError("数值过小")
            if kind == "odd" and value % 2 == 0:
                raise ValueError("该数值必须是奇数")
            return value
        value = float(text)
        if kind == "positive_float" and value <= 0:
            raise ValueError("数值必须大于 0")
        if kind == "nonnegative_float" and value < 0:
            raise ValueError("数值不能小于 0")
        return value

    def _save(self) -> None:
        try:
            result = {
                key: self._parse_value(entry.get(), kind)
                for key, (entry, kind) in self.entries.items()
            }
            if self.operation_id == "contrast":
                low, high = result["low_percentile"], result["high_percentile"]
                if not (0 <= low < high <= 100):
                    raise ValueError("低分位点和高分位点必须满足 0 ≤ 低 < 高 ≤ 100")
        except (TypeError, ValueError) as exc:
            messagebox.showerror("参数无效", str(exc), parent=self)
            return
        self.result = result
        self.destroy()


class PointSelector(tk.Toplevel):
    def __init__(self, master, image: np.ndarray, mode: str):
        super().__init__(master)
        self.title("选择ROI" if mode == "roi" else "选择透视四角")
        self.transient(master)
        self.grab_set()
        self.mode = mode
        self.required = 2 if mode == "roi" else 4
        self.points: list[tuple[int, int]] = []
        self.result = None
        max_width, max_height = 1050, 700
        height, width = image.shape[:2]
        self.scale = min(max_width / width, max_height / height, 1.0)
        display = cv2.resize(image, (round(width * self.scale), round(height * self.scale))) if self.scale < 1 else image
        self.photo = _photo(display, (max_width, max_height))
        self.canvas = tk.Canvas(self, width=self.photo.width(), height=self.photo.height(), cursor="cross")
        self.canvas.pack(padx=10, pady=10)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.bind("<Button-1>", self._click)
        text = "依次点击左上与右下" if mode == "roi" else "依次点击左上、右上、右下、左下"
        self.status = ttk.Label(self, text=text)
        self.status.pack(pady=(0, 6))
        buttons = ttk.Frame(self)
        buttons.pack(pady=(0, 10))
        ttk.Button(buttons, text="重新选择", command=self._clear).pack(side="left", padx=4)
        ttk.Button(buttons, text="确认", command=self._confirm).pack(side="left", padx=4)
        ttk.Button(buttons, text="取消", command=self.destroy).pack(side="left", padx=4)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _click(self, event) -> None:
        if len(self.points) >= self.required:
            return
        point = (round(event.x / self.scale), round(event.y / self.scale))
        self.points.append(point)
        radius = 4
        self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill="#22c55e", tags="mark")
        self.canvas.create_text(event.x + 10, event.y - 10, text=str(len(self.points)), fill="#dc2626", font=("Arial", 12, "bold"), tags="mark")
        if len(self.points) > 1:
            previous = self.points[-2]
            self.canvas.create_line(previous[0] * self.scale, previous[1] * self.scale, event.x, event.y, fill="#22c55e", width=2, tags="mark")
        self.status.configure(text=f"已选择 {len(self.points)}/{self.required} 个点")

    def _clear(self) -> None:
        self.points.clear()
        self.canvas.delete("mark")
        self.status.configure(text="请重新点击")

    def _confirm(self) -> None:
        if len(self.points) != self.required:
            messagebox.showwarning("点数不足", f"需要选择 {self.required} 个点", parent=self)
            return
        if self.mode == "roi":
            (x1, y1), (x2, y2) = self.points
            x, y = min(x1, x2), min(y1, y2)
            self.result = [x, y, abs(x2 - x1), abs(y2 - y1)]
        else:
            self.result = [[x, y] for x, y in self.points]
        self.destroy()


class OCRPreprocessApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        configure_style(root)
        self.root.title("ME5405 · OCR 图像工作台")
        self.root.geometry("1500x900")
        self.root.minsize(1180, 720)
        self.input_dir = PROJECT_ROOT / "images" / "raw"
        self.output_root = PROJECT_ROOT / "outputs" / "manual_preprocessing"
        self.gt_path = PROJECT_ROOT / "ground_truth.csv"
        self.recipe_path = PROJECT_ROOT / "manual_recipes.json"
        self.entries: dict[str, dict] = {}
        self.current_name: str | None = None
        self.preview_result = None
        self.current_attempt_index: int | None = None
        self.reader = None
        self.reader_lock = threading.Lock()
        self.busy = False
        self.operation_display = {f"[{spec.category}] {spec.label}": operation_id for operation_id, spec in OPERATIONS.items() if operation_id != "raw"}
        self._build_ui()
        self.reload_images()

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(20, 14, 20, 10))
        header.pack(fill="x")
        title_row = ttk.Frame(header)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="OCR 图像工作台", style="Title.TLabel").pack(side="left")
        ttk.Label(title_row, text="ME5405", style="Badge.TLabel").pack(side="left", padx=12)
        ttk.Label(title_row, text="预处理  →  文字识别  →  结果对比", style="Muted.TLabel").pack(side="right")
        toolbar = ttk.Frame(header)
        toolbar.pack(fill="x", pady=(12, 0))
        ttk.Label(toolbar, text="图片目录", style="Muted.TLabel").pack(side="left", padx=(0, 10))
        self.input_var = tk.StringVar(value=str(self.input_dir))
        path_entry = ttk.Entry(toolbar, textvariable=self.input_var, width=30)
        path_entry.pack(side="left", fill="x", expand=True)
        path_entry.bind("<Return>", lambda _event: self.reload_images())
        ttk.Button(toolbar, text="选择目录", command=self.choose_input).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="重新加载", command=self.reload_images).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="真值 CSV", command=self.choose_ground_truth).pack(side="left", padx=(8, 0))

        # Fixed sidebars give the image the remaining width; no oversized controls.
        body = ttk.Frame(self.root, padding=(20, 0, 20, 0))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, width=220, style="Card.TFrame", padding=14)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.grid_propagate(False)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        center = ttk.Frame(body, style="Card.TFrame", padding=(12, 14, 12, 4))
        center.grid(row=0, column=1, sticky="nsew")
        right = ttk.Frame(body, width=390, style="Card.TFrame")
        right.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
        right.pack_propagate(False)

        ttk.Label(left, text="图片库", style="Heading.TLabel").grid(row=0, column=0, sticky="w")
        self.image_count_var = tk.StringVar(value="0 张图片")
        ttk.Label(left, textvariable=self.image_count_var, style="CardMuted.TLabel").grid(row=1, column=0, sticky="w", pady=(4, 12))
        list_frame = ttk.Frame(left, style="Card.TFrame")
        list_frame.grid(row=2, column=0, sticky="nsew")
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)
        self.image_list = styled_listbox(list_frame, width=1, height=1, font=("Microsoft YaHei UI", 9))
        self.image_list.grid(row=0, column=0, sticky="nsew")
        image_scroll = ttk.Scrollbar(list_frame, command=self.image_list.yview)
        image_scroll.grid(row=0, column=1, sticky="ns")
        image_xscroll = ttk.Scrollbar(list_frame, orient="horizontal", command=self.image_list.xview)
        image_xscroll.grid(row=1, column=0, sticky="ew")
        self.image_list.configure(yscrollcommand=image_scroll.set, xscrollcommand=image_xscroll.set)
        self.image_list.bind("<<ListboxSelect>>", self.select_image)
        ttk.Label(left, text="逐张预览 · 独立选择流程", style="CardMuted.TLabel").grid(row=3, column=0, sticky="w", pady=(12, 0))

        self.image_name_var = tk.StringVar(value="图像预览")
        self.image_heading = ttk.Label(center, textvariable=self.image_name_var, style="Heading.TLabel")
        self.image_heading.pack(anchor="w", pady=(0, 4))
        ttk.Label(center, text="先检查处理效果，再运行 OCR。预览不会触发识别。", style="CardMuted.TLabel").pack(anchor="w", pady=(0, 12))
        center.bind("<Configure>", lambda event: self.image_heading.configure(wraplength=max(100, event.width - 24)))
        self.notebook = ttk.Notebook(center)
        self.notebook.pack(fill="both", expand=True)
        self.original_view = ImageView(self.notebook, "原图")
        self.processed_view = ImageView(self.notebook, "处理预览")
        self.annotated_view = ImageView(self.notebook, "识别标注")
        self.notebook.add(self.original_view, text="原图")
        self.notebook.add(self.processed_view, text="处理预览")
        self.notebook.add(self.annotated_view, text="OCR 标注")

        self.controls_notebook = ttk.Notebook(right)
        self.controls_notebook.pack(fill="both", expand=True, padx=4, pady=(8, 0))
        configuration = ScrollPanel(self.controls_notebook)
        self.configuration_tab = configuration
        form = configuration.content
        self.results_tab = ttk.Frame(self.controls_notebook, style="Card.TFrame", padding=14)
        self.controls_notebook.add(configuration, text="流程与真值")
        self.controls_notebook.add(self.results_tab, text="识别结果")

        ttk.Label(form, text="01  自选预处理步骤", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(form, text="逐个添加处理方式，并按执行顺序排列。需要灰度图的操作请先添加灰度化。", style="CardMuted.TLabel", wraplength=320).pack(anchor="w", pady=(4, 10))

        recipe_frame = ttk.Frame(form, style="Card.TFrame")
        recipe_frame.pack(fill="x", pady=(12, 10))
        recipe_frame.columnconfigure(0, weight=1)
        self.recipe_list = styled_listbox(recipe_frame, height=5, width=1, font=("Microsoft YaHei UI", 10))
        self.recipe_list.grid(row=0, column=0, sticky="ew")
        recipe_scroll = ttk.Scrollbar(recipe_frame, command=self.recipe_list.yview)
        recipe_scroll.grid(row=0, column=1, sticky="ns")
        recipe_xscroll = ttk.Scrollbar(recipe_frame, orient="horizontal", command=self.recipe_list.xview)
        recipe_xscroll.grid(row=1, column=0, sticky="ew")
        self.recipe_list.configure(yscrollcommand=recipe_scroll.set, xscrollcommand=recipe_xscroll.set)
        self.recipe_list.bind("<Double-Button-1>", lambda _event: self.configure_operation())
        op_row = ttk.Frame(form, style="Card.TFrame")
        op_row.pack(fill="x")
        self.operation_var = tk.StringVar(value=next(iter(self.operation_display)))
        ttk.Combobox(op_row, textvariable=self.operation_var, values=list(self.operation_display), state="readonly", width=16).pack(side="left", fill="x", expand=True)
        ttk.Button(op_row, text="添加", style="Small.TButton", command=self.add_operation).pack(side="left", padx=(6, 0))
        edit_row = ttk.Frame(form, style="Card.TFrame")
        edit_row.pack(fill="x", pady=(8, 10))
        for column, (text, command) in enumerate([("↑ 上移", lambda: self.move_operation(-1)), ("↓ 下移", lambda: self.move_operation(1)), ("删除", self.remove_operation), ("参数", self.configure_operation)]):
            edit_row.columnconfigure(column, weight=1)
            ttk.Button(edit_row, text=text, style="Small.TButton", command=command).grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 0))
        self.pipeline_text = ttk.Label(form, text="流程：raw", style="CardMuted.TLabel", wraplength=320)
        self.pipeline_text.pack(fill="x")

        ttk.Separator(form).pack(fill="x", pady=16)
        ttk.Label(form, text="02  人工正确文字", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(form, text="用于计算字符错误率 CER，可在识别前填写。", style="CardMuted.TLabel", wraplength=320).pack(anchor="w", pady=(4, 10))
        self.gt_text = styled_text(form, height=3, width=1, wrap="word", undo=True)
        self.gt_text.pack(fill="x")
        gt_row = ttk.Frame(form, style="Card.TFrame")
        gt_row.pack(fill="x", pady=(8, 0))
        self.gt_file_var = tk.StringVar(value=self.gt_path.name)
        ttk.Label(gt_row, textvariable=self.gt_file_var, style="CardMuted.TLabel").pack(side="left")
        ttk.Button(gt_row, text="保存真值", style="Small.TButton", command=self.save_current_ground_truth).pack(side="right")

        results = self.results_tab
        ttk.Label(results, text="03  识别与评价", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(results, text="对照原图文字，检查识别质量。", style="CardMuted.TLabel").pack(anchor="w", pady=(4, 12))
        self.metric_var = tk.StringVar(value="confidence：待运行　CER：待运行")
        ttk.Label(results, textvariable=self.metric_var, style="Card.TLabel", wraplength=330).pack(fill="x", pady=(0, 10))
        ttk.Label(results, text="confidence 不是准确率；CER 越低越好。", style="CardMuted.TLabel", wraplength=330).pack(anchor="w", pady=(0, 12))
        text_frame = ttk.Frame(results, style="Card.TFrame")
        text_frame.pack(fill="both", expand=True)
        self.result_text = styled_text(text_frame, height=4, width=1, wrap="word", state="disabled")
        self.result_text.pack(side="left", fill="both", expand=True)
        result_scroll = ttk.Scrollbar(text_frame, command=self.result_text.yview)
        result_scroll.pack(side="right", fill="y")
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.final_var = tk.StringVar(value="最终方案：尚未选择")
        ttk.Label(results, textvariable=self.final_var, style="Success.TLabel", wraplength=330).pack(fill="x", pady=(10, 4))
        self.final_button = ttk.Button(results, text="设为最终方案", command=self.select_current_attempt, state="disabled")
        self.final_button.pack(fill="x", pady=(0, 8))
        ttk.Button(results, text="导出总览与最终结果", command=self.export_results).pack(fill="x")
        self._set_result_text("")

        # Recognition actions stay visible even when the configuration scrolls.
        footer = ttk.Frame(right, style="Card.TFrame", padding=(14, 10, 14, 14))
        footer.pack(fill="x", side="bottom", before=self.controls_notebook)
        ttk.Separator(footer).pack(fill="x", pady=(0, 10))
        self.progress = ttk.Progressbar(footer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))
        action_row = ttk.Frame(footer, style="Card.TFrame")
        action_row.pack(fill="x")
        self.run_button = ttk.Button(action_row, text="运行 OCR", style="Primary.TButton", command=self.run_current_ocr)
        self.run_button.pack(side="left", fill="x", expand=True, padx=(0, 3))

        self.status_var = tk.StringVar(value="就绪")
        status = ttk.Frame(self.root, padding=(20, 9))
        status.pack(fill="x", side="bottom")
        ttk.Label(status, text="●", foreground=COLORS["accent"]).pack(side="left", padx=(0, 8))
        status_label = ttk.Label(status, textvariable=self.status_var, style="Muted.TLabel", anchor="w")
        status_label.pack(side="left", fill="x", expand=True)
        status.bind("<Configure>", lambda event: status_label.configure(wraplength=max(100, event.width - 75)))

    def choose_input(self) -> None:
        chosen = filedialog.askdirectory(initialdir=self.input_dir)
        if chosen:
            self.input_var.set(chosen)
            self.reload_images()

    def choose_ground_truth(self) -> None:
        chosen = filedialog.askopenfilename(initialdir=PROJECT_ROOT, filetypes=[("CSV", "*.csv")])
        if chosen:
            self.gt_path = Path(chosen)
            self.gt_file_var.set(self.gt_path.name)
            self.reload_images()

    def reload_images(self) -> None:
        if self.busy:
            self.status_var.set("OCR 正在运行，请完成后再重新加载图片。")
            return
        try:
            self.input_dir = Path(self.input_var.get()).resolve()
            images = find_images(self.input_dir)
            answers = load_ground_truth(self.gt_path)
            saved = load_recipes(self.recipe_path).get("images", {})
            self.entries = {}
            for index, path in enumerate(images, 1):
                saved_image = saved.get(path.name, {})
                recipe = saved_image.get("draft_recipe") or saved_image.get("recipe") or ["raw"]
                try:
                    recipe = normalize_recipe(recipe, auto_gray=False)
                except ValueError:
                    recipe = normalize_recipe(["raw"], auto_gray=False)
                self.entries[path.name] = {
                    "image_name": path.name,
                    "image_path": str(path),
                    "artifact_id": f"{index:03}",
                    "original": None,
                    "recipe": recipe,
                    "ground_truth": answers.get(path.name, ""),
                    "attempts": [],
                    "selected_attempt": None,
                }
            self.image_list.delete(0, "end")
            for name in self.entries:
                self.image_list.insert("end", name)
            self.image_count_var.set(f"{len(images)} 张图片")
            if images:
                self.image_list.selection_set(0)
                self.select_image()
                self.status_var.set(f"已加载 {len(images)} 张图片；预处理预览和OCR运行是分开的。")
            else:
                self.current_name = None
                self.current_attempt_index = None
                self.preview_result = None
                self.image_name_var.set("图像预览")
                self.original_view.show(None)
                self.processed_view.show(None)
                self.annotated_view.show(None)
                self.recipe_list.delete(0, "end")
                self.pipeline_text.configure(text="请选择图片以配置流程。")
                self.gt_text.delete("1.0", "end")
                self._set_result_text("")
                self.metric_var.set("confidence：待运行　CER：待运行")
                self._refresh_final_label()
                self.run_button.configure(state="disabled")
                self.status_var.set(f"目录中没有支持的图片：{self.input_dir}")
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))

    def current_entry(self) -> dict | None:
        return self.entries.get(self.current_name) if self.current_name else None

    def select_image(self, _event=None) -> None:
        selection = self.image_list.curselection()
        if not selection:
            return
        name = self.image_list.get(selection[0])
        self.current_name = name
        self.image_name_var.set(name)
        self.controls_notebook.select(self.configuration_tab)
        entry = self.entries[name]
        if entry["original"] is None:
            try:
                entry["original"] = load_image(Path(entry["image_path"]))
            except Exception as exc:
                messagebox.showerror("读取失败", f"{name}: {exc}")
                return
        self.original_view.show(entry["original"])
        self.gt_text.delete("1.0", "end")
        self.gt_text.insert("1.0", entry["ground_truth"])
        self.current_attempt_index = None
        self.annotated_view.show(None)
        self._set_result_text("")
        self.metric_var.set("confidence：—　CER：—")
        self._refresh_recipe_list()
        self._refresh_final_label()
        self.preview_pipeline()

    def _refresh_recipe_list(self) -> None:
        entry = self.current_entry()
        self.recipe_list.delete(0, "end")
        if not entry:
            return
        for index, item in enumerate(entry["recipe"], 1):
            spec = OPERATIONS[item["op"]]
            suffix = "（自动补入）" if item.get("auto_inserted") else ""
            params = deepcopy(spec.default_params)
            params.update(item.get("params", {}))
            parameter_text = self._parameter_summary(item["op"], params)
            if parameter_text:
                suffix += f"【{parameter_text}】"
            self.recipe_list.insert("end", f"{index}. {spec.label}{suffix}")
        self.pipeline_text.configure(text="流程：" + recipe_label(entry["recipe"]))

    @staticmethod
    def _parameter_summary(operation_id: str, params: dict) -> str:
        if operation_id == "rotate":
            return f"角度={params.get('angle')}°" if params.get("angle") is not None else "角度待设置"
        if operation_id == "roi":
            return "已选区域" if params.get("roi") else "区域待选"
        if operation_id == "perspective":
            return "已选四角" if params.get("corners") else "四角待选"
        labels = {
            "low_percentile": "低分位", "high_percentile": "高分位", "clip_limit": "限制",
            "tile_grid": "分块", "kernel": "核", "sigma": "sigma", "diameter": "直径",
            "sigma_color": "颜色σ", "sigma_space": "空间σ", "block_size": "窗口",
            "constant": "C", "iterations": "次数", "amount": "强度",
        }
        if operation_id not in PARAMETER_FIELDS:
            return ""
        parts = []
        for key, _label, _kind, _hint in PARAMETER_FIELDS[operation_id]:
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                value = "×".join(str(part) for part in value)
            parts.append(f"{labels.get(key, key)}={value}")
        return "，".join(parts)

    def _save_draft_recipe(self) -> None:
        entry = self.current_entry()
        if not entry:
            return
        data = load_recipes(self.recipe_path)
        stored = data["images"].setdefault(entry["image_name"], {})
        stored["draft_recipe"] = entry["recipe"]
        save_recipes(self.recipe_path, data)

    def _set_recipe(self, recipe: list, selected_index: int | None = None) -> bool:
        entry = self.current_entry()
        if not entry:
            return False
        try:
            entry["recipe"] = normalize_recipe(recipe, auto_gray=False)
        except ValueError as exc:
            messagebox.showwarning("流程不合法", str(exc))
            return False
        self._refresh_recipe_list()
        if selected_index is not None and entry["recipe"]:
            self.recipe_list.selection_set(min(selected_index, len(entry["recipe"]) - 1))
        self._save_draft_recipe()
        self.preview_pipeline()
        return True

    def add_operation(self) -> None:
        entry = self.current_entry()
        if not entry:
            return
        operation_id = self.operation_display[self.operation_var.get()]
        recipe = deepcopy(entry["recipe"])
        if len(recipe) == 1 and recipe[0]["op"] == "raw":
            recipe = []
        if operation_id in {"otsu", "adaptive"}:
            existing = next((i for i, item in enumerate(recipe) if item["op"] in {"otsu", "adaptive"}), None)
            if existing is not None:
                if not messagebox.askyesno("更换二值化", "同一流程只能使用一种二值化。是否替换现有方法？"):
                    return
                recipe.pop(existing)
        item = {"op": operation_id, "params": {}}
        if OPERATIONS[operation_id].spatial:
            recipe.insert(0, item)
        else:
            recipe.append(item)
        if self._set_recipe(recipe, len(recipe) - 1) and operation_id in {"roi", "perspective", "rotate"}:
            self.recipe_list.selection_clear(0, "end")
            index = next(i for i, value in enumerate(self.current_entry()["recipe"]) if value["op"] == operation_id)
            self.recipe_list.selection_set(index)
            self.configure_operation()

    def _selected_recipe_index(self) -> int | None:
        selection = self.recipe_list.curselection()
        return selection[0] if selection else None

    def move_operation(self, direction: int) -> None:
        entry = self.current_entry()
        index = self._selected_recipe_index()
        if not entry or index is None:
            return
        target = index + direction
        if not 0 <= target < len(entry["recipe"]):
            return
        recipe = deepcopy(entry["recipe"])
        recipe[index], recipe[target] = recipe[target], recipe[index]
        self._set_recipe(recipe, target)

    def remove_operation(self) -> None:
        entry = self.current_entry()
        index = self._selected_recipe_index()
        if not entry or index is None:
            return
        recipe = deepcopy(entry["recipe"])
        recipe.pop(index)
        self._set_recipe(recipe or ["raw"], max(0, index - 1))

    def configure_operation(self) -> None:
        entry = self.current_entry()
        index = self._selected_recipe_index()
        if not entry or index is None:
            return
        item = deepcopy(entry["recipe"][index])
        operation_id = item["op"]
        if operation_id == "rotate":
            value = simpledialog.askfloat("旋转角度", "输入逆时针旋转角度（-180到180）：", parent=self.root, minvalue=-180, maxvalue=180)
            if value is None:
                return
            item["params"]["angle"] = value
        elif operation_id in {"roi", "perspective"}:
            selector = PointSelector(self.root, entry["original"], operation_id)
            self.root.wait_window(selector)
            if selector.result is None:
                return
            item["params"]["roi" if operation_id == "roi" else "corners"] = selector.result
        elif operation_id in PARAMETER_FIELDS:
            params = deepcopy(OPERATIONS[operation_id].default_params)
            params.update(item.get("params", {}))
            dialog = ParameterDialog(self.root, operation_id, params)
            self.root.wait_window(dialog)
            if dialog.result is None:
                return
            item["params"] = dialog.result
        else:
            messagebox.showinfo("无需设置参数", NO_PARAMETER_HELP.get(operation_id, "此操作没有需要手工设置的参数。"))
            return
        recipe = deepcopy(entry["recipe"])
        recipe[index] = item
        self._set_recipe(recipe, index)

    def preview_pipeline(self) -> None:
        entry = self.current_entry()
        if not entry:
            return
        try:
            self.preview_result = run_pipeline(entry["original"], entry["recipe"], {})
            entry["recipe"] = self.preview_result.recipe
            self.processed_view.show(self.preview_result.image)
            self.status_var.set("预处理预览已更新；OCR结果待刷新。")
            self.notebook.select(self.processed_view)
        except (ValueError, PipelineUnavailable) as exc:
            self.preview_result = None
            self.processed_view.show(None)
            self.status_var.set(f"流程暂不能预览：{exc}")
        self.current_attempt_index = None
        self.annotated_view.show(None)
        self._set_result_text("")
        self.metric_var.set("confidence：待运行　CER：待运行")
        self.controls_notebook.select(self.configuration_tab)
        # 预处理预览本身不运行 OCR。只要没有后台 OCR 任务，
        # 两个 OCR 按钮应当始终可用。
        if not self.busy:
            self.run_button.configure(state="normal")

    def save_current_ground_truth(self) -> None:
        entry = self.current_entry()
        if not entry:
            return
        text = self.gt_text.get("1.0", "end-1c")
        try:
            save_ground_truth(self.gt_path, entry["image_name"], text)
            entry["ground_truth"] = text
            self.status_var.set(f"人工真值已保存：{self.gt_path}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))

    def _get_reader(self):
        with self.reader_lock:
            if self.reader is None:
                import easyocr
                model_dir = PROJECT_ROOT / "models" / "easyocr"
                self.reader = easyocr.Reader(
                    ["ch_sim", "en"],
                    gpu=False,
                    model_storage_directory=str(model_dir),
                    user_network_directory=str(model_dir / "user_network"),
                    verbose=False,
                )
            return self.reader

    def _set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        state = "disabled" if busy or not self.current_entry() else "normal"
        self.run_button.configure(state=state)
        self.run_button.configure(text="识别中…" if busy else "运行 OCR")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(value=0)
        self.status_var.set(status)

    def run_current_ocr(self) -> None:
        entry = self.current_entry()
        if not entry or self.busy:
            return
        self.save_current_ground_truth()
        name = entry["image_name"]
        recipe = deepcopy(entry["recipe"])
        original = entry["original"].copy()
        truth = entry["ground_truth"]
        recipe_ops = [item["op"] for item in recipe]
        has_raw = any([item["op"] for item in attempt.get("recipe", [])] == ["raw"] for attempt in entry["attempts"])
        needs_raw = recipe_ops != ["raw"] and not has_raw
        status = "正在先运行原图基准，再运行当前流程……" if needs_raw else "正在初始化/运行OCR，请稍候……"
        self._set_busy(True, status)

        def work():
            try:
                reader = self._get_reader()
                results = []
                if needs_raw:
                    results.append(execute_attempt(reader, Path(entry["image_path"]), original, ["raw"], truth, label="原图基准"))
                results.append(execute_attempt(reader, Path(entry["image_path"]), original, recipe, truth, label="自定义流程"))
                self.root.after(0, lambda completed=results: self._finish_attempts(name, completed))
            except Exception as exc:
                # Python 会在 except 块结束后清理 exc，因此必须在此处
                # 绑定到默认参数，否则 UI 回调会再次报错并让按钮永久变灰。
                self.root.after(0, lambda error=exc: self._finish_error(error))

        threading.Thread(target=work, daemon=True).start()

    def _finish_attempt(self, name: str, attempt: dict) -> None:
        self._finish_attempts(name, [attempt])

    def _finish_attempts(self, name: str, attempts: list[dict]) -> None:
        entry = self.entries[name]
        entry["attempts"].extend(attempts)
        index = len(entry["attempts"]) - 1
        self._set_busy(False, "OCR完成。请检查文字和CER，再决定是否设为最终方案。")
        if self.current_name == name:
            self.current_attempt_index = index
            self._display_attempt(entry["attempts"][index])

    def _finish_error(self, error: Exception) -> None:
        self._set_busy(False, f"运行失败：{error}")
        messagebox.showerror("运行失败", str(error))

    def _display_attempt(self, attempt: dict) -> None:
        self.annotated_view.show(attempt["annotated"])
        self.notebook.select(self.annotated_view)
        self.controls_notebook.select(self.results_tab)
        entry = self.current_entry()
        raw = next(
            (item for item in (entry or {}).get("attempts", []) if [step["op"] for step in item.get("recipe", [])] == ["raw"]),
            None,
        )
        if raw and [step["op"] for step in attempt.get("recipe", [])] != ["raw"]:
            self._set_result_text(f"原图OCR识别文字：\n{raw['text']}\n\n当前流程OCR识别文字：\n{attempt['text']}")
        else:
            self._set_result_text("OCR识别文字：\n" + attempt["text"])
        metric = attempt.get("evaluation") or {}
        cer = f"{metric['cer']:.2%}" if metric.get("status") == "ok" else "未填写真值"
        if raw and raw is not attempt:
            raw_metric = raw.get("evaluation") or {}
            raw_cer = f"{raw_metric['cer']:.2%}" if raw_metric.get("status") == "ok" else "未填写真值"
            self.metric_var.set(
                f"原图 confidence/CER：{raw['weighted_confidence']:.3f} / {raw_cer}\n"
                f"当前 confidence/CER：{attempt['weighted_confidence']:.3f} / {cer}"
            )
        else:
            self.metric_var.set(f"confidence：{attempt['weighted_confidence']:.3f}　CER：{cer}")

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text or "尚未运行 OCR\n\n先在「流程与真值」中确认预处理效果，然后点击下方「运行 OCR」。\n\n识别文字和原图对比结果会显示在这里。")
        self.result_text.configure(state="disabled")
        self.final_button.configure(state="normal" if text else "disabled")

    def _select_attempt(self, index: int) -> None:
        entry = self.current_entry()
        if not entry:
            return
        entry["selected_attempt"] = index
        self.current_attempt_index = index
        attempt = entry["attempts"][index]
        entry["recipe"] = deepcopy(attempt["recipe"])
        self._refresh_recipe_list()
        self._display_attempt(attempt)
        data = load_recipes(self.recipe_path)
        data["images"][entry["image_name"]] = {
            "label": attempt["label"],
            "recipe": attempt["recipe"],
            "draft_recipe": attempt["recipe"],
        }
        save_recipes(self.recipe_path, data)
        self._refresh_final_label()
        self.status_var.set("已保存该图片的最终流程。")

    def select_current_attempt(self) -> None:
        if self.current_attempt_index is None:
            messagebox.showinfo("尚无OCR结果", "请先运行当前流程 OCR。")
            return
        self._select_attempt(self.current_attempt_index)

    def _refresh_final_label(self) -> None:
        entry = self.current_entry()
        if not entry or entry["selected_attempt"] is None:
            self.final_var.set("最终方案：尚未选择")
            return
        attempt = entry["attempts"][entry["selected_attempt"]]
        self.final_var.set("最终方案：" + attempt["pipeline_label"])

    def export_results(self) -> None:
        selected = [entry for entry in self.entries.values() if entry.get("selected_attempt") is not None]
        if not selected:
            messagebox.showinfo("没有最终结果", "请至少为一张图片运行OCR并设为最终方案。")
            return
        try:
            run_dir = export_manual_run(self.output_root, selected)
            overview = run_dir / "overview.html"
            self.status_var.set(f"已导出：{run_dir}")
            webbrowser.open(overview.as_uri())
            messagebox.showinfo("导出完成", f"已生成直观总览和最终结果：\n{run_dir}")
        except Exception as exc:
            messagebox.showerror("导出失败", str(exc))


def main() -> None:
    root = tk.Tk()
    OCRPreprocessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
