import sys

from PyQt5.QtWidgets import QApplication

from app.emu_window import EmuWindow

__author__ = ["Paweł Kopeć", "Michał Górecki"]


def main():
    app = QApplication(sys.argv)

    window = EmuWindow()
    window.show()

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
