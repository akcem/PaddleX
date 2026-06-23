from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config import MODES
from core.paths import REPO_ROOT, DET_CONFIG, REC_CONFIG, now_stamp


class ConfigPanel(QWidget):
    """A self-contained config panel for one model type (DET or REC)."""

    def __init__(self, model_type, default_config_path, parent=None):
        super().__init__(parent)
        self.model_type = model_type
        self.field_widgets = {}
        self._build(default_config_path)

    def _build(self, default_path):
        layout = QVBoxLayout(self)
        self.field_specs = []
        self.browse_widgets = {}

        # Config path row (the only top bar item kept)
        top = QGridLayout()
        layout.addLayout(top)
        self.config_path = QLineEdit(str(default_path))
        self.config_browse_btn = QPushButton("浏览")
        top.addWidget(QLabel("配置文件"), 0, 0)
        top.addWidget(self.config_path, 0, 1, 1, 2)
        top.addWidget(self.config_browse_btn, 0, 3)

        # Scroll area with form fields
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_container = QWidget()
        self.form_layout = QFormLayout(form_container)
        self.form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        scroll.setWidget(form_container)
        layout.addWidget(scroll, 1)

        # Bottom buttons
        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        self.load_config_btn = QPushButton("加载配置")
        self.save_config_btn = QPushButton("保存并覆盖配置")
        self.run_mode_btn = QPushButton("运行当前 mode")
        buttons.addWidget(self.load_config_btn)
        buttons.addWidget(self.save_config_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.run_mode_btn)

    def set_fields(self, field_specs):
        self._clear_form()
        self.field_specs = list(field_specs)
        self.field_widgets = {}
        self.browse_widgets = {}

        current_section = None
        for spec in self.field_specs:
            if spec.section != current_section:
                current_section = spec.section
                header = QLabel(spec.section)
                header.setStyleSheet("font-weight: 600; margin-top: 8px;")
                self.form_layout.addRow(header)
            if spec.editor_kind == "mode":
                editor = QComboBox()
                editor.addItems(MODES)
                editor.setEditable(False)
            elif spec.editor_kind == "bool":
                editor = QComboBox()
                editor.addItems(["True", "False"])
            elif spec.editor_kind == "yaml":
                editor = QTextEdit()
                editor.setAcceptRichText(False)
                editor.setMinimumHeight(88)
            else:
                editor = QLineEdit()
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(editor, 1)
            if spec.browse_kind:
                browse = QPushButton("浏览")
                browse.setFixedWidth(60)
                row_layout.addWidget(browse)
                self.browse_widgets[spec.path] = browse
            self.field_widgets[spec.path] = editor
            self.form_layout.addRow(spec.label, row_widget)

    def _clear_form(self):
        while self.form_layout.rowCount():
            self.form_layout.removeRow(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PaddleX OCR 训练 GUI")
        self.resize(1280, 900)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Top bar: only project root
        top = QHBoxLayout()
        root.addLayout(top)
        top.addWidget(QLabel("项目根目录"))
        self.repo_root = QLineEdit(str(REPO_ROOT))
        top.addWidget(self.repo_root, 1)
        self.repo_browse_btn = QPushButton("浏览")
        top.addWidget(self.repo_browse_btn)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)
        self.dataset_tab = QWidget()
        self.config_tab = QWidget()
        self.validation_tab = QWidget()
        self.tabs.addTab(self.dataset_tab, "数据集预处理")
        self.tabs.addTab(self.config_tab, "配置参数")
        self.tabs.addTab(self.validation_tab, "验证")

        self._build_dataset_tab()
        self._build_config_tab()
        self._build_validation_tab()
        self._build_log(root)

    def _build_dataset_tab(self):
        layout = QVBoxLayout(self.dataset_tab)
        intro = QLabel(
            "增量训练数据预处理：最新数据 + 多个既往数据集按比例抽样，"
            "输出 PaddleOCR/PaddleX 可直接读取的 train.txt、val.txt 和 images 目录。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QGridLayout()
        layout.addLayout(form)
        self.latest_dataset = QLineEdit()
        self.latest_browse_btn = QPushButton("浏览")
        self.latest_ratio = QLineEdit("1.0")
        self.dataset_kind = QComboBox()
        self.dataset_kind.addItems(["det", "rec"])
        self._add_path_row(form, 0, "最新数据集", self.latest_dataset, self.latest_browse_btn)
        form.addWidget(QLabel("读取子目录"), 1, 0)
        form.addWidget(self.dataset_kind, 1, 1)
        form.addWidget(QLabel("最新数据抽样比例"), 1, 2)
        form.addWidget(self.latest_ratio, 1, 3)

        old_box = QGroupBox("既往数据集")
        old_layout = QVBoxLayout(old_box)
        old_tools = QHBoxLayout()
        old_layout.addLayout(old_tools)
        old_tools.addWidget(QLabel("抽样比例（1=全量，0.25=25%）"))
        self.old_ratio = QLineEdit("1.0")
        self.old_ratio.setFixedWidth(100)
        old_tools.addWidget(self.old_ratio)
        self.add_old_btn = QPushButton("添加既往数据集")
        self.add_old_batch_btn = QPushButton("批量添加")
        self.remove_old_btn = QPushButton("移除选中")
        old_tools.addWidget(self.add_old_btn)
        old_tools.addWidget(self.add_old_batch_btn)
        old_tools.addWidget(self.remove_old_btn)
        old_tools.addStretch(1)
        self.old_list = QListWidget()
        old_layout.addWidget(self.old_list)
        layout.addWidget(old_box, 1)

        out_form = QGridLayout()
        layout.addLayout(out_form)
        self.merge_output = QLineEdit(
            str(REPO_ROOT / "mytools" / "merged_datasets" / f"ocr_{now_stamp()}")
        )
        self.merge_output_browse_btn = QPushButton("浏览")
        self._add_path_row(out_form, 0, "合并输出目录", self.merge_output, self.merge_output_browse_btn)
        self.val_ratio = QLineEdit("0.1")
        self.random_seed = QLineEdit("2026")
        out_form.addWidget(QLabel("仅 label.txt 时验证集比例"), 1, 0)
        out_form.addWidget(self.val_ratio, 1, 1)
        out_form.addWidget(QLabel("随机种子"), 1, 2)
        out_form.addWidget(self.random_seed, 1, 3)

        buttons = QHBoxLayout()
        layout.addLayout(buttons)
        buttons.addStretch(1)
        self.merge_only_btn = QPushButton("生成本次训练数据")
        self.merge_save_btn = QPushButton("生成并写入配置")
        buttons.addWidget(self.merge_only_btn)
        buttons.addWidget(self.merge_save_btn)

    def _build_config_tab(self):
        layout = QVBoxLayout(self.config_tab)

        # Sub-tabs for DET and REC
        self.config_tabs = QTabWidget()
        self.config_panels = {}
        for mt, cp in [("DET", DET_CONFIG), ("REC", REC_CONFIG)]:
            panel = ConfigPanel(mt, cp)
            self.config_panels[mt] = panel
            self.config_tabs.addTab(panel, mt)
        layout.addWidget(self.config_tabs, 1)

        # Shared bottom row
        bottom = QHBoxLayout()
        layout.addLayout(bottom)
        self.backup_config = QCheckBox("覆盖前备份 .bak")
        self.backup_config.setChecked(True)
        self.stop_process_btn = QPushButton("停止")
        bottom.addWidget(self.backup_config)
        bottom.addStretch(1)
        bottom.addWidget(self.stop_process_btn)

    def _build_validation_tab(self):
        layout = QVBoxLayout(self.validation_tab)
        intro = QLabel("使用 PaddleOCR 对单张图片做验证，保存可视化图片和 JSON。")
        layout.addWidget(intro)

        grid = QGridLayout()
        layout.addLayout(grid)
        self.val_image = QLineEdit()
        self.val_image_browse_btn = QPushButton("浏览")
        self.val_output = QLineEdit(str(REPO_ROOT / "mytools" / "validation_output"))
        self.val_output_browse_btn = QPushButton("浏览")
        self.val_det_name = QComboBox()
        self.val_det_name.setEditable(True)
        self.val_det_name.addItems(["PP-OCRv5_server_det", "PP-OCRv5_mobile_det"])
        self.val_det_dir = QLineEdit(str(REPO_ROOT / "mytools" / "exports" / "det"))
        self.val_det_browse_btn = QPushButton("浏览")
        self.val_use_rec = QCheckBox("启用识别")
        self.val_use_rec.setChecked(True)
        self.val_rec_name = QComboBox()
        self.val_rec_name.setEditable(True)
        self.val_rec_name.addItems(["PP-OCRv5_server_rec", "PP-OCRv5_mobile_rec"])
        self.val_rec_dir = QLineEdit(str(REPO_ROOT / "mytools" / "exports" / "rec"))
        self.val_rec_browse_btn = QPushButton("浏览")
        self.val_device = QComboBox()
        self.val_device.addItems(["gpu", "cpu"])
        self._add_path_row(grid, 0, "输入图片", self.val_image, self.val_image_browse_btn)
        self._add_path_row(grid, 1, "验证输出目录", self.val_output, self.val_output_browse_btn)
        grid.addWidget(QLabel("检测模型名"), 2, 0)
        grid.addWidget(self.val_det_name, 2, 1)
        self._add_path_row(grid, 3, "检测模型目录", self.val_det_dir, self.val_det_browse_btn)
        grid.addWidget(self.val_use_rec, 4, 0)
        grid.addWidget(self.val_rec_name, 4, 1)
        self._add_path_row(grid, 5, "识别模型目录", self.val_rec_dir, self.val_rec_browse_btn)
        grid.addWidget(QLabel("Device"), 6, 0)
        grid.addWidget(self.val_device, 6, 1)

        run_row = QHBoxLayout()
        layout.addLayout(run_row)
        self.run_validation_btn = QPushButton("运行验证并绘图")
        self.stop_validation_btn = QPushButton("停止验证")
        self.open_validation_output_btn = QPushButton("打开输出目录")
        run_row.addWidget(self.run_validation_btn)
        run_row.addWidget(self.stop_validation_btn)
        run_row.addStretch(1)
        run_row.addWidget(self.open_validation_output_btn)

        preview_tools = QHBoxLayout()
        layout.addLayout(preview_tools)
        preview_tools.addWidget(QLabel("绘图预览"))
        preview_tools.addStretch(1)
        self.preview_original_btn = QPushButton("原始大小")
        self.preview_fit_btn = QPushButton("适应窗口")
        self.preview_zoom_out_btn = QPushButton("缩小")
        self.preview_zoom_in_btn = QPushButton("放大")
        for widget in (
            self.preview_original_btn,
            self.preview_fit_btn,
            self.preview_zoom_out_btn,
            self.preview_zoom_in_btn,
        ):
            preview_tools.addWidget(widget)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_label = QLabel("验证后会在这里显示绘图结果")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_scroll.setWidget(self.preview_label)
        layout.addWidget(self.preview_scroll, 1)

    def _build_log(self, root_layout):
        log_tools = QHBoxLayout()
        root_layout.addLayout(log_tools)
        log_tools.addWidget(QLabel("日志"))
        log_tools.addStretch(1)
        self.clear_log_btn = QPushButton("清空")
        log_tools.addWidget(self.clear_log_btn)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(130)
        root_layout.addWidget(self.log_text)

    def _add_path_row(self, layout, row, label, edit, button):
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(edit, row, 1, 1, 2)
        layout.addWidget(button, row, 3)
