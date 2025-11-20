import sys, os, json, re
from PySide6 import QtCore, QtWidgets, QtGui
from app.pipeline import run_weekly_pipeline, run_daily_pipeline
from app.config import load_config, save_config
from app.parse_progress import FormCount, ParseProgressTracker


class RunnerThread(QtCore.QThread):
    progress = QtCore.Signal(str)   # status updates
    finished = QtCore.Signal(str)   # done/cancelled/error

    def __init__(self, mode: str, start_stage: str, skip_fda: bool = False):
        super().__init__()
        self.mode = mode
        self.start_stage = start_stage
        self.skip_fda = skip_fda
        self.stop_flag = {"stop": False}

    def run(self):
        try:
            if self.mode == "weekly":
                run_weekly_pipeline(
                    stop_flag=self.stop_flag,
                    progress_fn=self.progress.emit,
                    start_stage=self.start_stage,
                    skip_fda=self.skip_fda,
                )
            elif self.mode == "daily":
                run_daily_pipeline(
                    stop_flag=self.stop_flag,
                    progress_fn=self.progress.emit,
                    start_stage=self.start_stage
                )

            if self.stop_flag["stop"]:
                self.finished.emit("cancelled")
            else:
                self.finished.emit("done")

        except Exception as e:
            # pipeline already logged to errorlog
            self.finished.emit(f"error: {e}")

    def request_stop(self):
        self.stop_flag["stop"] = True
        self.progress.emit("cancel requested")


class ParseStatsWidget(QtWidgets.QTreeWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.setColumnCount(5)
        self.setHeaderLabels(["Form Type", "Parsed", "Valid", "Missing", "Balanced"])
        header = self.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for col in range(1, 5):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.setRootIsDecorated(False)
        self.setAlternatingRowColors(True)
        self.setFocusPolicy(QtCore.Qt.NoFocus)

    def update_from_tracker(self, form_stats: dict[str, FormCount]) -> None:
        self.setUpdatesEnabled(False)
        self.clear()
        for form in sorted(form_stats.keys()):
            stats = form_stats[form]
            item = QtWidgets.QTreeWidgetItem([
                form,
                str(stats.parsed),
                str(stats.valid),
                str(stats.missing),
                "",
            ])
            self._apply_balance_indicator(item, 4, stats)
            self.addTopLevelItem(item)
        self.setUpdatesEnabled(True)

    def _apply_balance_indicator(self, item: QtWidgets.QTreeWidgetItem, column: int, stats: FormCount) -> None:
        balanced = stats.totals_match()
        item.setText(column, "✓" if balanced else "✗")
        delta = stats.parsed - (stats.valid + stats.missing)
        tooltip = "Totals match" if balanced else f"Mismatch by {delta:+d}"
        item.setToolTip(column, tooltip)
        color = QtGui.QColor("darkgreen") if balanced else QtGui.QColor("red")
        item.setForeground(column, QtGui.QBrush(color))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microcap Pipeline")
        self.resize(900, 600)

        self.cfg = load_config()
        self.worker = None
        self.is_running = False

        # live_buffer holds all recent progress msgs so we don't lose them
        self.live_buffer: list[str] = []
        # When True we are displaying messages from an active/most recent run
        # and should not overwrite the live log with persisted runlog.csv data.
        self.tail_from_live_run = False

        # timer to refresh offline logs (runlog.csv / errorlog.csv)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.refresh_logs)
        self.timer.start(self.cfg["GUI"]["LogRefreshSeconds"] * 1000)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # --- Run tab ---
        self.run_tab = QtWidgets.QWidget()
        tabs.addTab(self.run_tab, "Run")

        self.btn_weekly = QtWidgets.QPushButton("Run Weekly (Full)")
        self.btn_daily  = QtWidgets.QPushButton("Run Daily (Prices Only)")
        self.btn_cancel = QtWidgets.QPushButton("Cancel Run")
        self.btn_cancel.setEnabled(False)

        self.weekly_stage_combo = QtWidgets.QComboBox()
        self.weekly_stage_combo.addItem("Universe (Full Run)", "universe")
        self.weekly_stage_combo.addItem("Profiles", "profiles")
        self.weekly_stage_combo.addItem("Filings", "filings")
        self.weekly_stage_combo.addItem("Prices", "prices")
        self.weekly_stage_combo.addItem("FDA", "fda")
        self.weekly_stage_combo.addItem("Hydrate + Shortlist", "hydrate")
        self.weekly_stage_combo.addItem("Deep Research", "deep_research")
        self.weekly_stage_combo.addItem("Parse Q10", "parse_q10")
        self.weekly_stage_combo.addItem("Parse 8-K", "parse_8k")
        self.weekly_stage_combo.addItem("Populate DR Output", "dr_populate")
        self.weekly_stage_combo.addItem("Build Validated Watchlist", "build_watchlist")

        self.daily_stage_combo = QtWidgets.QComboBox()
        self.daily_stage_combo.addItem("Prices", "prices")
        self.daily_stage_combo.addItem("Hydrate + Shortlist", "hydrate")

        self.chk_skip_fda = QtWidgets.QCheckBox("Skip FDA")

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)  # spinning bar
        self.progress_bar.setVisible(False)

        self.status_label = QtWidgets.QLabel("Idle")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        # PySide6's QLabel does not expose setMinimumContentsLength, so approximate
        # the same behavior by setting a minimum width based on character width.
        min_width = self.status_label.fontMetrics().horizontalAdvance("M") * 60
        self.status_label.setMinimumWidth(min_width)
        self.status_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self.log_tail = QtWidgets.QPlainTextEdit()
        self.log_tail.setReadOnly(True)
        self.parse_stats_view = ParseStatsWidget()
        self.parse_tracker = ParseProgressTracker(self._update_parse_progress)
        self.parse_tracker.reset()

        run_layout = QtWidgets.QVBoxLayout()
        weekly_stage_row = QtWidgets.QHBoxLayout()
        weekly_stage_row.addWidget(QtWidgets.QLabel("Weekly start stage:"))
        weekly_stage_row.addWidget(self.weekly_stage_combo)
        daily_stage_row = QtWidgets.QHBoxLayout()
        daily_stage_row.addWidget(QtWidgets.QLabel("Daily start stage:"))
        daily_stage_row.addWidget(self.daily_stage_combo)

        run_layout.addLayout(weekly_stage_row)
        run_layout.addWidget(self.chk_skip_fda)
        run_layout.addWidget(self.btn_weekly)
        run_layout.addLayout(daily_stage_row)
        run_layout.addWidget(self.btn_daily)
        run_layout.addWidget(self.btn_cancel)
        run_layout.addWidget(self.progress_bar)
        run_layout.addWidget(self.status_label)
        run_layout.addWidget(QtWidgets.QLabel("Live Log Tail"))
        log_and_stats_layout = QtWidgets.QHBoxLayout()
        log_and_stats_layout.addWidget(self.log_tail, 3)

        stats_panel = QtWidgets.QVBoxLayout()
        stats_panel.addWidget(QtWidgets.QLabel("Parse Counts"))
        stats_panel.addWidget(self.parse_stats_view)
        stats_panel.addStretch(1)

        log_and_stats_layout.addLayout(stats_panel, 1)
        run_layout.addLayout(log_and_stats_layout)
        self.run_tab.setLayout(run_layout)

        self.btn_weekly.clicked.connect(self.start_weekly)
        self.btn_daily.clicked.connect(self.start_daily)
        self.btn_cancel.clicked.connect(self.cancel_run)

        # --- Config tab ---
        self.config_tab = QtWidgets.QWidget()
        tabs.addTab(self.config_tab, "Config")

        self.config_edit = QtWidgets.QPlainTextEdit()
        self.btn_load_cfg = QtWidgets.QPushButton("Load Config")
        self.btn_save_cfg = QtWidgets.QPushButton("Save Config")

        cfg_layout = QtWidgets.QVBoxLayout()
        cfg_layout.addWidget(QtWidgets.QLabel("Edit config.json below. All fields are live."))
        cfg_layout.addWidget(self.config_edit)
        cfg_layout.addWidget(self.btn_load_cfg)
        cfg_layout.addWidget(self.btn_save_cfg)
        self.config_tab.setLayout(cfg_layout)

        self.btn_load_cfg.clicked.connect(self.load_cfg_into_editor)
        self.btn_save_cfg.clicked.connect(self.save_cfg_from_editor)

        # init
        self.load_cfg_into_editor()
        self.refresh_logs()

    def start_weekly(self):
        stage = self.weekly_stage_combo.currentData()
        self._start_run("weekly", stage)

    def start_daily(self):
        stage = self.daily_stage_combo.currentData()
        self._start_run("daily", stage)

    def _start_run(self, mode: str, stage: str):
        # don't start if already running
        if self.worker and self.worker.isRunning():
            return

        self.is_running = True

        # Starting a new run resets the live log so only the current run is shown
        self.tail_from_live_run = True
        self.live_buffer = [f"=== starting {mode} run from stage '{stage}' ==="]
        self._render_live_buffer()
        self.parse_tracker.reset()

        self.status_label.setText(f"{mode} starting…")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.btn_cancel.setEnabled(True)
        self.btn_weekly.setEnabled(False)
        self.btn_daily.setEnabled(False)
        self.weekly_stage_combo.setEnabled(False)
        self.daily_stage_combo.setEnabled(False)
        self.chk_skip_fda.setEnabled(False)

        skip_fda = self.chk_skip_fda.isChecked() if mode == "weekly" else False
        self.worker = RunnerThread(mode, stage, skip_fda=skip_fda)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def cancel_run(self):
        if self.worker:
            self.worker.request_stop()

    def on_progress(self, msg: str):
        # append to live buffer
        self.tail_from_live_run = True
        self.live_buffer.append(msg)
        self._render_live_buffer(append_only=True)

        # update status label with the most recent message
        self.status_label.setText(msg)
        self._update_progress_bar_from_msg(msg)
        self.parse_tracker.process_message(msg)

    def on_finished(self, msg: str):
        self.is_running = False
        # append final state to buffer
        final_line = f"Finished: {msg}"
        self.live_buffer.append(final_line)
        self._render_live_buffer(append_only=True)

        self.status_label.setText(final_line)
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.btn_cancel.setEnabled(False)
        self.btn_weekly.setEnabled(True)
        self.btn_daily.setEnabled(True)
        self.weekly_stage_combo.setEnabled(True)
        self.daily_stage_combo.setEnabled(True)
        self.chk_skip_fda.setEnabled(True)

        # show disk logs once more at end
        self.refresh_logs()

    def _render_live_buffer(self, append_only: bool = False):
        if append_only and self.live_buffer:
            self.log_tail.appendPlainText(self.live_buffer[-1])
        else:
            self.log_tail.setPlainText("\n".join(self.live_buffer))

        # always keep the viewport scrolled to the newest message
        sb = self.log_tail.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _update_progress_bar_from_msg(self, msg: str):
        # Extract the pipeline message without the timestamp prefix so we can
        # detect stage transitions and embedded percentages consistently.
        body = msg.split("|", 1)[1].strip() if "|" in msg else msg

        match = re.search(r"\((\d{1,3})%\)", msg)
        if match:
            pct = max(0, min(100, int(match.group(1))))
            if self.progress_bar.maximum() != 100 or self.progress_bar.minimum() != 0:
                self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(pct)
        else:
            # Only reset the bar for explicit stage starts; otherwise retain
            # the last determinate value so parse progress stays visible even
            # when intermittent log messages do not include percentages.
            if re.search(r"\bstart\b", body):
                self.progress_bar.setRange(0, 0)
                self.progress_bar.setValue(0)

    def _update_parse_progress(self, form_stats: dict[str, FormCount]) -> None:
        self.parse_stats_view.update_from_tracker(form_stats)

    def refresh_logs(self):
        """
        Mirror runlog.csv into the live log when idle.
        While a run is active we never overwrite live_buffer so the tail stays
        intact for the entire execution.
        """
        paths = load_config()["Paths"]
        runlog_path = os.path.join(paths["logs"], "runlog.csv")
        runlog_txt = self._read_file(runlog_path)

        if not self.is_running and not self.tail_from_live_run:
            # when idle and not currently showing an in-memory run log,
            # mirror runlog.csv into the live panel as a convenience.
            lines = runlog_txt.splitlines()
            if lines and lines[0].startswith("timestamp"):
                lines = lines[1:]
            self.live_buffer = [line for line in lines if line]
            self.tail_from_live_run = False
            self._render_live_buffer()

    def _read_file(self, path: str) -> str:
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def load_cfg_into_editor(self):
        cfg = load_config()
        self.config_edit.setPlainText(json.dumps(cfg, indent=2))

    def save_cfg_from_editor(self):
        try:
            new_cfg = json.loads(self.config_edit.toPlainText())
            save_config(new_cfg)
            self.cfg = new_cfg
            # update timer interval dynamically
            self.timer.start(self.cfg["GUI"]["LogRefreshSeconds"] * 1000)
            self.status_label.setText("Config saved.")
        except Exception as e:
            self.status_label.setText(f"Config save error: {e}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
