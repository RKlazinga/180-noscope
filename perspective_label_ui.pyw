from PyQt5.QtWidgets import QApplication

from perspective_shift.label_ui.ui import UI

if __name__ == '__main__':
    app = QApplication([])
    ui = UI()
    app.exec_()
