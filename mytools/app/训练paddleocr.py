import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from PyQt6.QtWidgets import QApplication
def main():
    from ui.main_window import MainWindow
    from ui_logic.controller import AppController

    app = QApplication(sys.argv)
    window = MainWindow()
    window.controller = AppController(window)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
