import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- PyQt5 Imports ---
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QLabel, QLineEdit, QPushButton,
                           QFileDialog, QGroupBox, QGridLayout, QTextEdit,
                           QTabWidget, QSpinBox, QDoubleSpinBox, QMessageBox,
                           QProgressBar, QSizePolicy) # Added QSizePolicy
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon # Added QIcon

# --- Matplotlib Integration ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- Your Optimizer Module ---
# Ensure optimizer.py is in the same directory or Python path
try:
    from optimizer import CourtYardOptimizer
except ImportError:
    # Provide a dummy class if optimizer isn't available for UI testing
    print("Warning: optimizer.py not found. Using a dummy class.")
    class CourtYardOptimizer:
        def __init__(self, params): pass
        def set_progress_callback(self, cb): self.progress_cb = cb
        def set_log_callback(self, cb): self.log_cb = cb
        def load_data(self, f1, f2):
            self.log_cb(f"Dummy load: {f1}, {f2}")
            self.progress_cb(0.1, "Dummy loading...")
            # Simulate loading data (return dummy data for structure)
            dummy_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
            dummy_adj = {1: [2], 2: [1]}
            self.progress_cb(0.2, "Dummy load complete")
            return dummy_df, dummy_adj
        def run_optimization(self):
            self.log_cb("Dummy running optimization...")
            for i in range(1, 11):
                QTimer.singleShot(i * 100, lambda p=i/10.0: self.progress_cb(0.2 + p * 0.7, f"Dummy step {int(p*10)}"))
            QTimer.singleShot(1100, lambda: self.progress_cb(0.9, "Dummy finishing..."))
            QTimer.singleShot(1200, lambda: self.progress_cb(1.0, "Dummy done!"))
            return {'status': 'success', 'message': 'Dummy success message', 'results_df': None, 'plot_data': None} # Dummy results
        def format_results(self, results): return f"Dummy Formatted Results:\n{results['message']}"
        def generate_plots(self, results, output_dir):
             self.log_cb(f"Dummy generating plots in {output_dir}")
             # Simulate plotting
             ax = self.figure.add_subplot(111)
             ax.plot([1, 2, 3], [1, 4, 9], label="Dummy Plot")
             ax.set_title("Dummy Optimization Plot")
             ax.legend()
             # No need to save file here, just draw on canvas
             # self.canvas.draw() # Drawing is handled in MainWindow


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("院落搬迁优化系统 V1.1") # Added Version
        self.setGeometry(100, 100, 1250, 850) # Slightly larger default size
        self.apply_stylesheet() # Apply custom styles

        # Initialize optimizer placeholder
        self.optimizer = None
        self.figure = plt.figure() # Create the figure instance ONCE


        # Create center widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10) # Add margins
        main_layout.setSpacing(10) # Add spacing

        # --- Left Panel (Parameters) ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(15) # Spacing for parameter sections

        # File Selection Group
        file_group = QGroupBox("输入文件选择")
        file_layout = QGridLayout()
        file_layout.setVerticalSpacing(10)
        file_layout.setHorizontalSpacing(5)

        self.file1_path = QLineEdit()
        self.file1_path.setPlaceholderText("点击右侧按钮选择地块信息表 (.xlsx, .xls)")
        self.file2_path = QLineEdit()
        self.file2_path.setPlaceholderText("点击右侧按钮选择院落邻接表 (.xlsx, .xls)")

        # Use standard icons (SP_DirOpenIcon) or provide paths to custom icons
        icon_folder = self.style().standardIcon(QApplication.style().SP_DirOpenIcon)
        file1_btn = QPushButton(icon_folder, " 选择地块信息表") # Add icon
        file2_btn = QPushButton(icon_folder, " 选择院落邻接表") # Add icon

        file_layout.addWidget(QLabel("地块信息表:"), 0, 0)
        file_layout.addWidget(self.file1_path, 0, 1)
        file_layout.addWidget(file1_btn, 0, 2)
        file_layout.addWidget(QLabel("院落邻接表:"), 1, 0)
        file_layout.addWidget(self.file2_path, 1, 1)
        file_layout.addWidget(file2_btn, 1, 2)
        file_group.setLayout(file_layout)

        # Parameter Settings Group
        params_group = QGroupBox("优化参数设置")
        params_layout = QGridLayout()
        params_layout.setVerticalSpacing(8) # Adjust spacing
        params_layout.setHorizontalSpacing(10)

        # Create SpinBoxes (Consider adding units/tooltips)
        self.base_budget = QDoubleSpinBox()
        self.base_budget.setRange(0, 1e12) # Increased range
        self.base_budget.setValue(20000000)
        self.base_budget.setSingleStep(100000)
        self.base_budget.setSuffix(" 元") # Add unit
        self.base_budget.setToolTip("项目保底规划的总成本预算")

        self.reserve_ratio = QDoubleSpinBox()
        self.reserve_ratio.setRange(0, 1)
        self.reserve_ratio.setValue(0.3)
        self.reserve_ratio.setSingleStep(0.05) # Smaller step
        self.reserve_ratio.setDecimals(2) # Show percentage better
        self.reserve_ratio.setToolTip("预算中用于备用的资金比例 (0 到 1)")


        self.comm_cost = QDoubleSpinBox()
        self.comm_cost.setRange(0, 1e7) # Increased range
        self.comm_cost.setValue(30000)
        self.comm_cost.setSingleStep(1000)
        self.comm_cost.setSuffix(" 元/户") # Add unit
        self.comm_cost.setToolTip("与每户沟通协调所需的大致成本")


        self.move_months = QSpinBox()
        self.move_months.setRange(1, 24) # Increased range
        self.move_months.setValue(4)
        self.move_months.setSuffix(" 个月") # Add unit
        self.move_months.setToolTip("预估完成所有搬迁所需的总月数")


        self.rent_ns = QDoubleSpinBox()
        self.rent_ns.setRange(0, 1000) # Increased range
        self.rent_ns.setValue(15)
        self.rent_ns.setSuffix(" 元/平米/月") # Add unit

        self.rent_ew = QDoubleSpinBox()
        self.rent_ew.setRange(0, 1000) # Increased range
        self.rent_ew.setValue(8)
        self.rent_ew.setSuffix(" 元/平米/月") # Add unit

        self.rent_full = QDoubleSpinBox()
        self.rent_full.setRange(0, 1000) # Increased range
        self.rent_full.setValue(30)
        self.rent_full.setSuffix(" 元/平米/月") # Add unit

        self.adj_bonus = QDoubleSpinBox()
        self.adj_bonus.setRange(1, 3) # Increased range
        self.adj_bonus.setValue(1.2)
        self.adj_bonus.setSingleStep(0.05) # Smaller step
        self.adj_bonus.setDecimals(2)
        self.adj_bonus.setToolTip("相邻地块合并后的租金增益系数 (>= 1)")


        # Add labels and widgets to layout
        # Use bold labels for better readability
        params_layout.addWidget(QLabel("<b>保底规划成本:</b>"), 0, 0)
        params_layout.addWidget(self.base_budget, 0, 1)
        params_layout.addWidget(QLabel("<b>备用金比例:</b>"), 1, 0)
        params_layout.addWidget(self.reserve_ratio, 1, 1)
        params_layout.addWidget(QLabel("<b>沟通成本/户:</b>"), 2, 0)
        params_layout.addWidget(self.comm_cost, 2, 1)
        params_layout.addWidget(QLabel("<b>搬迁月数:</b>"), 3, 0)
        params_layout.addWidget(self.move_months, 3, 1)
        params_layout.addWidget(QLabel("<b>南北向租金:</b>"), 4, 0)
        params_layout.addWidget(self.rent_ns, 4, 1)
        params_layout.addWidget(QLabel("<b>东西向租金:</b>"), 5, 0)
        params_layout.addWidget(self.rent_ew, 5, 1)
        params_layout.addWidget(QLabel("<b>整院租金:</b>"), 6, 0)
        params_layout.addWidget(self.rent_full, 6, 1)
        params_layout.addWidget(QLabel("<b>毗邻增益系数:</b>"), 7, 0)
        params_layout.addWidget(self.adj_bonus, 7, 1)

        params_group.setLayout(params_layout)

        # Run Button
        # Use standard icon (SP_MediaPlay) or provide path to custom icon
        icon_run = self.style().standardIcon(QApplication.style().SP_MediaPlay)
        self.run_btn = QPushButton(icon_run, " 运行优化计算") # Add icon and text
        self.run_btn.setFixedHeight(45) # Make button taller
        # self.run_btn.setStyleSheet("font-weight: bold; font-size: 14pt;") # Specific style

        # Add widgets to left panel layout
        left_layout.addWidget(file_group)
        left_layout.addWidget(params_group)
        left_layout.addWidget(self.run_btn)
        left_layout.addStretch(1) # Push elements up

        # --- Right Panel (Results) ---
        right_panel = QTabWidget()

        # Text Results Tab
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        text_layout.setSpacing(10)

        # Progress Group
        progress_group = QGroupBox("优化进度")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True) # Show percentage
        self.progress_label = QLabel("准备运行...")
        self.progress_label.setAlignment(Qt.AlignCenter) # Center label
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        progress_group.setLayout(progress_layout)
        # Make progress group less tall initially
        progress_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        text_layout.addWidget(progress_group)


        # Log Output Group
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        # Adjust height constraints for better balance
        # self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        text_layout.addWidget(log_group, 1) # Give log some stretch factor

        # Optimization Results Group
        result_group = QGroupBox("优化结果摘要")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_layout.addWidget(self.result_text)
        result_group.setLayout(result_layout)
        text_layout.addWidget(result_group, 2) # Give results more stretch factor


        # Plot Results Tab
        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        # Create the canvas with the figure
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)

        # Add Tabs
        right_panel.addTab(text_tab, "文本结果 & 日志") # Rename tab
        right_panel.addTab(plot_tab, "图形结果") # Re-enable plot tab

        # Add panels to main layout
        main_layout.addWidget(left_panel, 2) # Proportion for left panel
        main_layout.addWidget(right_panel, 3) # Proportion for right panel

        # --- Connect Signals ---
        file1_btn.clicked.connect(lambda: self.select_file(self.file1_path, "地块信息表"))
        file2_btn.clicked.connect(lambda: self.select_file(self.file2_path, "院落邻接表"))
        self.run_btn.clicked.connect(self.run_optimization)

        # --- Initial Status ---
        self.update_progress(0, "请选择文件并设置参数")


    def apply_stylesheet(self):
        """Applies a global stylesheet to the application."""
        stylesheet = """
            QMainWindow, QWidget {
                background-color: #f0f0f0; /* Light gray background */
                font-size: 10pt; /* Base font size */
            }
            QGroupBox {
                background-color: #ffffff; /* White background for groups */
                border: 1px solid #cccccc; /* Light gray border */
                border-radius: 5px; /* Rounded corners */
                margin-top: 10px; /* Space above the group box title */
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left; /* Position title */
                padding: 0 5px 5px 5px; /* Padding around title */
                color: #333333;
                font-weight: bold;
            }
            QLabel {
                color: #333333; /* Dark gray text */
                padding: 2px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                border: 1px solid #cccccc;
                padding: 5px;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #252525; /* Highlight focus */
            }
            QPushButton {
                background-color: #252525; /* Blue background */
                color: white; /* White text */
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                min-height: 20px; /* Ensure minimum height */
            }
            QPushButton:hover {
                background-color: #252525; /* Darker blue on hover */
            }
            QPushButton:pressed {
                background-color: #252525; /* Even darker blue when pressed */
            }
            QPushButton:disabled {
                background-color: #cccccc; /* Gray out disabled button */
                color: #666666;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #ffffff;
                font-family: Consolas, monospace; /* Monospace font for logs/results */
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                text-align: center;
                color: #333333;
            }
            QProgressBar::chunk {
                background-color: #bdbdbd; /* Blue progress chunk */
                width: 10px; /* Adjust chunk width if needed */
                margin: 1px;
            }
            QTabWidget::pane { /* The tab contents area */
                border: 1px solid #cccccc;
                border-top: none;
                background-color: #f8f8f8;
                 border-bottom-left-radius: 5px;
                 border-bottom-right-radius: 5px;

            }
            QTabBar::tab {
                background: #e0e0e0; /* Light gray for inactive tabs */
                border: 1px solid #cccccc;
                border-bottom: none; /* Hides bottom border */
                padding: 8px 20px; /* Padding within tabs */
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                color: #555555;
            }
            QTabBar::tab:selected {
                background: #f8f8f8; /* Slightly lighter for active tab */
                color: #000000;
                border-bottom: 1px solid #f8f8f8; /* Match pane background */

            }
             QTabBar::tab:hover {
                 background: #d0d0d0;
             }
        """
        self.setStyleSheet(stylesheet)
        # Or apply to the whole application: QApplication.instance().setStyleSheet(stylesheet)


    def select_file(self, line_edit: QLineEdit, file_type: str):
        """Opens a file dialog to select an Excel file."""
        # Use os.path.expanduser("~") for a better default directory
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {file_type} 文件",
            desktop_path, # Start in user's desktop directory
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            line_edit.setText(file_path)


    def get_parameters(self):
        """获取所有参数值 (with validation if needed)"""
        # Add potential validation here if required
        return {
            'BASE_BUDGET': self.base_budget.value(),
            'RESERVE_RATIO': self.reserve_ratio.value(),
            'COMMUNICATION_COST': self.comm_cost.value(),
            'MOVE_DURATION_MONTHS': self.move_months.value(),
            'RENT_NS': self.rent_ns.value(),
            'RENT_EW': self.rent_ew.value(),
            'RENT_FULL_COURTYARD': self.rent_full.value(),
            'ADJACENCY_BONUS': self.adj_bonus.value()
        }

    def run_optimization(self):
        # --- 1. Input Validation ---
        file1 = self.file1_path.text()
        file2 = self.file2_path.text()
        if not file1 or not os.path.exists(file1):
            QMessageBox.warning(self, "输入错误", f"地块信息表文件无效或未选择！\n路径: {file1}")
            return
        if not file2 or not os.path.exists(file2):
            QMessageBox.warning(self, "输入错误", f"院落邻接表文件无效或未选择！\n路径: {file2}")
            return

        # --- 2. UI Preparation ---
        self.run_btn.setEnabled(False) # Disable button during run
        self.progress_bar.setValue(0)
        self.progress_label.setText("正在初始化...")
        self.log_text.clear()
        self.result_text.clear()
        self.figure.clear() # Clear previous plot
        self.canvas.draw() # Update canvas to show cleared plot
        QApplication.processEvents() # Update UI immediately

        # --- 3. Setup Optimizer ---
        try:
            params = self.get_parameters()
            self.optimizer = CourtYardOptimizer(params)
            self.optimizer.figure = self.figure # Pass figure/canvas if needed by optimizer
            self.optimizer.canvas = self.canvas
            self.optimizer.set_progress_callback(self.update_progress)
            self.optimizer.set_log_callback(self.update_log)
        except Exception as e:
            QMessageBox.critical(self, "初始化错误", f"创建优化器时出错：{str(e)}")
            self.run_btn.setEnabled(True) # Re-enable button on error
            return

        # --- 4. Load Data ---
        self.update_log("开始加载数据文件...")
        df_info, adjacency_dict = None, None # Initialize
        try:
            df_info, adjacency_dict = self.optimizer.load_data(file1, file2)
            if df_info is None or adjacency_dict is None:
                # Check if the optimizer logged the error
                if not self.log_text.toPlainText().strip().endswith("失败！"):
                    self.update_log("错误：优化器未能成功加载数据，请检查日志。")
                QMessageBox.critical(self, "数据加载失败", "未能加载输入数据，请检查文件格式和内容，并查看日志获取详细信息。")
                self.run_btn.setEnabled(True)
                return
            self.update_log("数据加载成功。")
        except Exception as e:
            self.update_log(f"数据加载过程中发生严重错误: {str(e)}")
            QMessageBox.critical(self, "数据加载错误", f"加载数据时发生意外错误：{str(e)}\n请检查文件是否正确以及程序是否有权限读取。")
            self.run_btn.setEnabled(True)
            return


        # --- 5. Run Optimization ---
        self.update_progress(0.2, "正在运行优化算法...") # Example progress update
        QApplication.processEvents()

        try:
            results = self.optimizer.run_optimization()

            # --- 6. Process Results ---
            self.update_progress(0.9, "正在处理和显示结果...")
            self.result_text.setText(self.optimizer.format_results(results))

            # Update plot only if successful
            if results and results.get('status') == 'success':
                # Create output directory (optional, if optimizer doesn't do it)
                output_dir = "optimization_results"
                os.makedirs(output_dir, exist_ok=True)
                self.update_log(f"正在生成图形结果到目录: {output_dir}")

                # Generate plots (ensure optimizer uses self.figure)
                self.optimizer.generate_plots(results, output_dir)
                self.canvas.draw() # Redraw the canvas with the new plot
                self.update_log("图形结果已生成并显示。")
                self.update_progress(1.0, "优化完成！")
                QMessageBox.information(self, "完成", "优化过程已成功完成！")
            else:
                status_msg = results.get('message', '未知状态') if results else '无结果返回'
                self.update_progress(1.0, f"优化未成功: {status_msg}")
                self.update_log(f"优化未成功或未产生有效结果。状态: {status_msg}")
                QMessageBox.warning(self, "优化问题", f"优化过程已结束，但未标记为成功。\n状态：{status_msg}\n请检查日志和结果。")


        except Exception as e:
            error_msg = f"优化计算过程中发生严重错误：{str(e)}"
            self.update_log(error_msg)
            # Add traceback for debugging if needed
            # import traceback
            # self.update_log(traceback.format_exc())
            QMessageBox.critical(self, "优化错误", f"{error_msg}\n请查看日志获取详细信息。")
            self.update_progress(1.0, "优化失败") # Mark progress as finished (failed)

        finally:
            # --- 7. Cleanup ---
            self.run_btn.setEnabled(True) # ALWAYS re-enable the button


    def update_progress(self, value: float, message: str = None):
        """Updates the progress bar and status label."""
        progress_val = max(0, min(100, int(value * 100))) # Clamp value
        self.progress_bar.setValue(progress_val)
        if message:
            self.progress_label.setText(message)
        QApplication.processEvents() # Keep UI responsive


    def update_log(self, message: str):
        """Appends a message to the log text area."""
        self.log_text.append(message)
        # Auto-scroll to the bottom
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        QApplication.processEvents()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # You can set a global font here if desired
    # font = QFont("Segoe UI", 10) # Example using Segoe UI font
    # app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())