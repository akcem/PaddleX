import locale
import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, QSize, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTreeView,
    QWidget,
)

from core import config as cfg
from core import dataset
from core.paths import REPO_ROOT, RUNTIME_CONFIG_DIR, default_python, norm_path
from core.validation import latest_image, validation_code


class MergeWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str, dict)
    failed = pyqtSignal(str)

    def __init__(self, sources, out_dir, val_ratio, seed):
        super().__init__()
        self.sources = sources
        self.out_dir = out_dir
        self.val_ratio = val_ratio
        self.seed = seed

    def run(self):
        try:
            out, manifest = dataset.merge_datasets(
                self.sources,
                self.out_dir,
                self.val_ratio,
                self.seed,
                log=self.log.emit,
            )
            self.done.emit(str(out), manifest)
        except Exception as exc:
            self.failed.emit(str(exc))


class AppController:
    def __init__(self, window):
        self.w = window
        self.old_datasets = []
        self.process = None
        self.validation_process = None
        self.merge_worker = None
        self.preview_pixmap = None
        self.preview_scale = 1.0
        self.runtime_config_path = None
        self._connect()
        self.load_config("DET")
        self.load_config("REC")

    def _connect(self):
        w = self.w

        # Top bar
        w.repo_browse_btn.clicked.connect(self.pick_repo_root)

        # Dataset tab
        w.latest_browse_btn.clicked.connect(self.pick_latest_dataset)
        w.dataset_kind.currentTextChanged.connect(self.on_dataset_kind_changed)
        w.old_ratio.editingFinished.connect(self.update_old_ratios_from_input)
        w.merge_output_browse_btn.clicked.connect(lambda: self.pick_dir(w.merge_output))
        w.add_old_btn.clicked.connect(self.add_old_dataset)
        w.add_old_batch_btn.clicked.connect(self.add_old_datasets_batch)
        w.remove_old_btn.clicked.connect(self.remove_old_dataset)
        w.merge_only_btn.clicked.connect(lambda: self.start_merge(False))
        w.merge_save_btn.clicked.connect(lambda: self.start_merge(True))

        # Config panels
        for model_type in ("DET", "REC"):
            panel = w.config_panels[model_type]
            panel.config_browse_btn.clicked.connect(
                lambda checked, mt=model_type: self.pick_config(mt)
            )
            panel.load_config_btn.clicked.connect(
                lambda checked, mt=model_type: self.load_config(mt)
            )
            panel.save_config_btn.clicked.connect(
                lambda checked, mt=model_type: self.save_config(mt)
            )
            panel.run_mode_btn.clicked.connect(
                lambda checked, mt=model_type: self.run_current_mode(mt)
            )

        # Config tab shared
        w.stop_process_btn.clicked.connect(self.stop_process)

        # Validation tab
        w.val_image_browse_btn.clicked.connect(
            lambda: self.pick_file(w.val_image, images=True)
        )
        w.val_output_browse_btn.clicked.connect(lambda: self.pick_dir(w.val_output))
        w.val_det_browse_btn.clicked.connect(lambda: self.pick_dir(w.val_det_dir))
        w.val_rec_browse_btn.clicked.connect(lambda: self.pick_dir(w.val_rec_dir))
        w.run_validation_btn.clicked.connect(self.start_validation)
        w.stop_validation_btn.clicked.connect(self.stop_validation)
        w.open_validation_output_btn.clicked.connect(self.open_validation_output)
        w.preview_zoom_in_btn.clicked.connect(lambda: self.zoom_preview(1.25))
        w.preview_zoom_out_btn.clicked.connect(lambda: self.zoom_preview(0.8))
        w.preview_fit_btn.clicked.connect(self.fit_preview)
        w.preview_original_btn.clicked.connect(self.original_preview)

        # Log
        w.clear_log_btn.clicked.connect(w.log_text.clear)

    def append_log(self, text):
        self.w.log_text.append(str(text))

    def _decode_process_bytes(self, raw):
        if not raw:
            return ""
        candidates = ["utf-8"]
        preferred = locale.getpreferredencoding(False)
        for enc in (preferred, "gbk", "cp936"):
            if enc and enc.lower() not in {x.lower() for x in candidates}:
                candidates.append(enc)
        for enc in candidates:
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode(candidates[0], errors="replace")

    def pick_dir(self, edit):
        path = QFileDialog.getExistingDirectory(
            self.w, "选择目录", edit.text() or str(REPO_ROOT)
        )
        if path:
            edit.setText(path)

    def pick_file(self, edit, weights=False, images=False):
        if images:
            filters = "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;All files (*.*)"
        elif weights:
            filters = "Paddle training weights (*.pdparams);;All files (*.*)"
        else:
            filters = "All files (*.*)"
        path, _ = QFileDialog.getOpenFileName(
            self.w, "选择文件", edit.text() or str(REPO_ROOT), filters
        )
        if path:
            edit.setText(path)

    def pick_repo_root(self):
        self.pick_dir(self.w.repo_root)

    def dataset_kind(self):
        return self.w.dataset_kind.currentText().strip().lower()

    def resolve_dataset_dir(self, path):
        root = Path(norm_path(path))
        kind = self.dataset_kind()
        child = root / kind
        if child.is_dir():
            return str(child)
        if root.name.lower() in ("det", "rec") and root.name.lower() != kind:
            sibling = root.parent / kind
            if sibling.is_dir():
                return str(sibling)
        return str(root)

    def on_dataset_kind_changed(self):
        latest = self.w.latest_dataset.text().strip()
        if latest:
            self.w.latest_dataset.setText(self.resolve_dataset_dir(latest))
        self.old_datasets = [
            (self.resolve_dataset_dir(path), ratio)
            for path, ratio in self.old_datasets
        ]
        self.refresh_old_list()

    def pick_latest_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self.w, "选择最新数据集", self.w.latest_dataset.text() or str(REPO_ROOT)
        )
        if path:
            self.w.latest_dataset.setText(self.resolve_dataset_dir(path))

    def pick_dirs(self, title, start_dir):
        dialog = QFileDialog(self.w, title, start_dir or str(REPO_ROOT))
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        path_row = QWidget(dialog)
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_edit = QLineEdit(str(start_dir or REPO_ROOT))
        go_btn = QPushButton("转到")
        path_layout.addWidget(QLabel("路径"))
        path_layout.addWidget(path_edit, 1)
        path_layout.addWidget(go_btn)

        def go_to_path():
            target = Path(norm_path(path_edit.text()))
            if target.is_dir():
                dialog.setDirectory(str(target))
            else:
                QMessageBox.warning(self.w, "路径无效", f"目录不存在：{target}")

        go_btn.clicked.connect(go_to_path)
        path_edit.returnPressed.connect(go_to_path)

        layout = dialog.layout()
        if layout is not None:
            try:
                layout.addWidget(path_row)
            except TypeError:
                row = layout.rowCount() if hasattr(layout, "rowCount") else 0
                columns = layout.columnCount() if hasattr(layout, "columnCount") else 1
                layout.addWidget(path_row, row, 0, 1, max(1, columns))
        for view in dialog.findChildren(QListView) + dialog.findChildren(QTreeView):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if dialog.exec():
            return dialog.selectedFiles()
        return []

    def pick_config(self, model_type):
        panel = self.w.config_panels[model_type]
        start_dir = str(Path(panel.config_path.text()).parent)
        path, _ = QFileDialog.getOpenFileName(
            self.w,
            "选择配置文件",
            start_dir,
            "YAML (*.yaml *.yml);;All files (*.*)",
        )
        if path:
            panel.config_path.setText(path)
            self.load_config(model_type)

    def load_config(self, model_type):
        panel = self.w.config_panels[model_type]
        try:
            data = cfg.read_config(panel.config_path.text(), model_type)
            panel.set_fields(cfg.build_field_specs(data))
            self._connect_panel_browse_buttons(panel)
            self.populate_form(panel, data)
            self.append_log(f"已加载配置：{panel.config_path.text()}")
        except Exception as exc:
            QMessageBox.critical(self.w, "加载失败", str(exc))

    def populate_form(self, panel, data):
        for spec in panel.field_specs:
            editor = panel.field_widgets[spec.path]
            self.set_editor_value(editor, cfg.display_value(cfg.get_by_path(data, spec.path)))

    def _connect_panel_browse_buttons(self, panel):
        for spec in panel.field_specs:
            browse = panel.browse_widgets.get(spec.path)
            if browse is None:
                continue
            editor = panel.field_widgets[spec.path]
            if spec.browse_kind == "dir":
                browse.clicked.connect(lambda checked, e=editor: self.pick_dir(e))
            else:
                browse.clicked.connect(
                    lambda checked, e=editor, s=spec: self.pick_file(
                        e,
                        weights=cfg.is_weight_field(s.path),
                        images=cfg.is_image_field(s.path),
                    )
                )

    def set_editor_value(self, editor, text):
        if hasattr(editor, "setPlainText"):
            editor.setPlainText(text)
        elif hasattr(editor, "setCurrentText"):
            editor.setCurrentText(text)
        else:
            editor.setText(text)

    def editor_value(self, editor):
        if hasattr(editor, "toPlainText"):
            return editor.toPlainText()
        if hasattr(editor, "currentText"):
            return editor.currentText()
        return editor.text()

    def _panel_config_path(self, model_type):
        panel = self.w.config_panels[model_type]
        path = Path(norm_path(panel.config_path.text()))
        if path.is_absolute():
            return path
        return Path(norm_path(self.w.repo_root.text())) / path

    def collect_config(self, model_type):
        panel = self.w.config_panels[model_type]
        cfg_path = self._panel_config_path(model_type)
        base = cfg.read_config(str(cfg_path), model_type)
        for spec in panel.field_specs:
            try:
                value = cfg.parse_editor_value(
                    self.editor_value(panel.field_widgets[spec.path]), spec
                )
            except Exception as exc:
                dotted = ".".join(spec.path)
                raise ValueError(f"{dotted} 填写无效：{exc}") from exc
            cfg.set_by_path(base, spec.path, value)
        return base

    def save_config(self, model_type):
        panel = self.w.config_panels[model_type]
        try:
            data = self.collect_config(model_type)
            cfg_path = self._panel_config_path(model_type)
            backup = cfg.write_config(
                str(cfg_path), data, self.w.backup_config.isChecked()
            )
            if backup:
                self.append_log(f"已备份原配置：{backup}")
            self.append_log(f"已覆盖保存配置：{cfg_path}")
            return True
        except Exception as exc:
            QMessageBox.critical(self.w, "保存失败", str(exc))
            return False

    def _write_runtime_config(self, model_type):
        data = self.collect_config(model_type)
        runtime_dir = Path(norm_path(RUNTIME_CONFIG_DIR))
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_path = runtime_dir / f"{model_type.lower()}_runtime.yaml"
        cfg.write_config(str(runtime_path), data, backup=False)
        self.runtime_config_path = runtime_path
        return runtime_path

    def run_current_mode(self, model_type):
        if self.process is not None:
            QMessageBox.warning(self.w, "进程运行中", "当前已有进程在运行。")
            return
        repo = Path(norm_path(self.w.repo_root.text()))
        try:
            cfg_path = self._write_runtime_config(model_type)
        except Exception as exc:
            QMessageBox.critical(self.w, "运行失败", str(exc))
            return
        try:
            cfg_arg = cfg_path.relative_to(repo).as_posix()
        except ValueError:
            cfg_arg = str(cfg_path)
        self.process = QProcess(self.w)
        self.process.setWorkingDirectory(str(repo))
        self.process.setProgram(default_python())
        self.process.setArguments(["main.py", "-c", cfg_arg])
        self.process.readyReadStandardOutput.connect(
            lambda: self.read_process(self.process)
        )
        self.process.readyReadStandardError.connect(
            lambda: self.read_process(self.process)
        )
        self.process.finished.connect(self.on_process_finished)
        self.append_log(
            f"运行使用临时配置：{cfg_path}，不会覆盖正式配置；点击“保存并覆盖配置”才会写回。"
        )
        self.append_log(f"命令：{default_python()} main.py -c {cfg_arg}")
        self.process.start()

    def read_process(self, process):
        data = self._decode_process_bytes(bytes(process.readAllStandardOutput()))
        err = self._decode_process_bytes(bytes(process.readAllStandardError()))
        for text in (data, err):
            if text:
                self.append_log(text.rstrip())

    def on_process_finished(self, code, _status):
        self.append_log(f"进程结束，退出码：{code}")
        self.process = None

    def stop_process(self):
        if self.process is None:
            self.append_log("当前没有运行中的进程。")
            return
        pid = self.process.processId()
        self.append_log(f"正在停止进程 (PID: {pid})...")
        tree_killed = False
        # On Windows, kill the whole process tree first. If we kill the parent
        # QProcess before taskkill runs, the training child can become orphaned.
        if pid and sys.platform == "win32":
            try:
                cp = subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                tree_killed = cp.returncode == 0
                if not tree_killed:
                    detail = (cp.stderr or cp.stdout or "").strip()
                    if detail:
                        self.append_log(f"taskkill 未完全成功：{detail}")
            except subprocess.TimeoutExpired:
                self.append_log("taskkill 超时，改用 QProcess.kill() 兜底。")
        if not tree_killed:
            self.process.kill()
        self.process.waitForFinished(5000)
        self.process = None
        self.append_log("已停止训练进程。")

    def add_old_dataset(self):
        path = QFileDialog.getExistingDirectory(
            self.w, "选择既往数据集", str(REPO_ROOT)
        )
        if not path:
            return
        self.add_old_dataset_path(path)

    def add_old_datasets_batch(self):
        paths = self.pick_dirs("批量选择既往数据集", str(REPO_ROOT))
        if not paths:
            return
        for path in paths:
            self.add_old_dataset_path(path, show_errors=False)
        self.refresh_old_list()

    def add_old_dataset_path(self, path, show_errors=True):
        try:
            ratio = dataset.parse_ratio(self.w.old_ratio.text())
        except Exception as exc:
            if show_errors:
                QMessageBox.critical(self.w, "比例错误", str(exc))
            return
        self.old_datasets.append((self.resolve_dataset_dir(path), ratio))
        if show_errors:
            self.refresh_old_list()

    def update_old_ratios_from_input(self):
        try:
            ratio = dataset.parse_ratio(self.w.old_ratio.text())
        except Exception as exc:
            QMessageBox.critical(self.w, "比例错误", str(exc))
            return
        self.old_datasets = [(path, ratio) for path, _old_ratio in self.old_datasets]
        self.refresh_old_list()

    def remove_old_dataset(self):
        for item in self.w.old_list.selectedItems():
            row = self.w.old_list.row(item)
            self.old_datasets.pop(row)
        self.refresh_old_list()

    def refresh_old_list(self):
        self.w.old_list.clear()
        for path, ratio in self.old_datasets:
            self.w.old_list.addItem(QListWidgetItem(f"{ratio:.2f} | {path}"))

    def start_merge(self, save_after):
        latest = self.w.latest_dataset.text().strip()
        if not latest:
            QMessageBox.warning(self.w, "缺少数据集", "请选择最新数据集。")
            return
        try:
            latest_ratio = dataset.parse_ratio(self.w.latest_ratio.text())
            old_ratio = dataset.parse_ratio(self.w.old_ratio.text())
            val_ratio = dataset.parse_ratio(self.w.val_ratio.text(), 0.1)
            seed = int(self.w.random_seed.text())
        except Exception as exc:
            QMessageBox.critical(self.w, "参数错误", str(exc))
            return
        self.old_datasets = [(path, old_ratio) for path, _ratio in self.old_datasets]
        self.refresh_old_list()
        latest = self.resolve_dataset_dir(latest)
        sources = [(latest, latest_ratio, "latest")]
        sources.extend(
            (self.resolve_dataset_dir(path), old_ratio, f"old_{idx + 1}")
            for idx, (path, _ratio) in enumerate(self.old_datasets)
        )
        self.merge_worker = MergeWorker(
            sources, self.w.merge_output.text(), val_ratio, seed
        )
        self.merge_worker.log.connect(self.append_log)
        self.merge_worker.failed.connect(
            lambda msg: self.append_log(f"合并失败：{msg}")
        )
        self.merge_worker.done.connect(
            lambda out, manifest: self.on_merge_done(out, manifest, save_after)
        )
        self.merge_worker.start()

    def on_merge_done(self, out, manifest, save_after):
        model_type = self.dataset_kind().upper()
        panel = self.w.config_panels[model_type]
        self.append_log(
            f"合并完成：train={manifest['train_samples']}，val={manifest['val_samples']}，"
            f"缺失图片={manifest['missing_count']}"
        )
        applied = []
        for path, value in cfg.dataset_updates_for(model_type, out).items():
            editor = panel.field_widgets.get(path)
            if editor is None:
                continue
            self.set_editor_value(editor, cfg.display_value(value))
            applied.append(".".join(path))
        if applied:
            self.append_log(f"已填入 {model_type} 数据集参数：{', '.join(applied)}")
        else:
            self.append_log(
                f"当前 {model_type} 配置页未找到可回填的数据集字段，请手动检查 Train/Eval 参数。"
            )
        if save_after:
            self.save_config(model_type)

    def _active_model_type(self):
        return self.w.config_tabs.currentWidget().model_type

    def start_validation(self):
        if self.validation_process is not None:
            QMessageBox.warning(self.w, "验证运行中", "当前已有验证进程在运行。")
            return
        image = self.w.val_image.text().strip()
        det_dir = self.w.val_det_dir.text().strip()
        rec_dir = self.w.val_rec_dir.text().strip()
        if not image or not Path(image).exists():
            QMessageBox.warning(self.w, "缺少图片", "请选择存在的输入图片。")
            return
        if not det_dir or not Path(det_dir).exists():
            QMessageBox.warning(self.w, "缺少检测模型", "请选择导出的检测模型目录。")
            return
        if self.w.val_use_rec.isChecked() and (
            not rec_dir or not Path(rec_dir).exists()
        ):
            QMessageBox.warning(
                self.w, "缺少识别模型", "启用识别时请选择导出的识别模型目录。"
            )
            return
        out = Path(norm_path(self.w.val_output.text()))
        out.mkdir(parents=True, exist_ok=True)
        args = [
            "-c",
            validation_code(),
            image,
            str(out),
            det_dir,
            "true" if self.w.val_use_rec.isChecked() else "false",
            rec_dir,
            self.w.val_device.currentText(),
            self.w.val_det_name.currentText(),
            self.w.val_rec_name.currentText(),
            str(Path(norm_path(self.w.repo_root.text()))),
        ]
        self.validation_process = QProcess(self.w)
        self.validation_process.setWorkingDirectory(self.w.repo_root.text())
        self.validation_process.setProgram(default_python())
        self.validation_process.setArguments(args)
        self.validation_process.readyReadStandardOutput.connect(
            lambda: self.read_process(self.validation_process)
        )
        self.validation_process.readyReadStandardError.connect(
            lambda: self.read_process(self.validation_process)
        )
        self.validation_process.finished.connect(
            lambda code, status: self.on_validation_finished(code, status, out)
        )
        self.append_log(
            "验证命令："
            + default_python()
            + " -c <validation_code> ..."
        )
        self.validation_process.start()

    def on_validation_finished(self, code, _status, output_dir):
        self.append_log(f"验证进程结束，退出码：{code}")
        self.validation_process = None
        if code == 0:
            image = latest_image(output_dir)
            if image:
                self.load_preview(image)
            else:
                self.w.preview_label.setText(
                    f"未找到可视化图片：{output_dir}"
                )

    def stop_validation(self):
        if self.validation_process is None:
            self.append_log("当前没有运行中的验证进程。")
            return
        self.validation_process.terminate()

    def open_validation_output(self):
        out = Path(norm_path(self.w.val_output.text()))
        out.mkdir(parents=True, exist_ok=True)
        os.startfile(str(out))

    def load_preview(self, path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self.w.preview_label.setText(f"预览失败：{path}")
            return
        self.preview_pixmap = pixmap
        self.fit_preview()
        self.append_log(f"已预览验证图片：{path}")

    def render_preview(self):
        if self.preview_pixmap is None:
            return
        size = QSize(
            max(1, int(self.preview_pixmap.width() * self.preview_scale)),
            max(1, int(self.preview_pixmap.height() * self.preview_scale)),
        )
        scaled = self.preview_pixmap.scaled(
            size,
            aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation,
        )
        self.w.preview_label.setPixmap(scaled)
        self.w.preview_label.resize(scaled.size())

    def zoom_preview(self, factor):
        if self.preview_pixmap is None:
            return
        self.preview_scale = max(
            0.05, min(self.preview_scale * factor, 10.0)
        )
        self.render_preview()

    def fit_preview(self):
        if self.preview_pixmap is None:
            return
        viewport = self.w.preview_scroll.viewport().size()
        scale_w = viewport.width() / max(self.preview_pixmap.width(), 1)
        scale_h = viewport.height() / max(self.preview_pixmap.height(), 1)
        self.preview_scale = min(scale_w, scale_h, 1.0)
        self.render_preview()

    def original_preview(self):
        if self.preview_pixmap is None:
            return
        self.preview_scale = 1.0
        self.render_preview()
