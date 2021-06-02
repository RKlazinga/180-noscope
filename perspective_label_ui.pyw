from PyQt5.QtWidgets import QApplication

from data_collection.perspective_shift.label_ui import UI

if __name__ == '__main__':
    app = QApplication([])
    ui = UI()
    app.exec_()
