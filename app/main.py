"""App PySide6: sube ortofoto, cuenta árboles, muestra resultado.

Uso:
    python -m app.main

Modelo: YOLO11n sobre HARV v2 (mAP50=0.587 en test).
Umbral de confianza ajustable (10%–90%, default 30%).
"""
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSpinBox, QMessageBox,
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

from app.inference import TreeDetector


def pil_to_qpixmap(pil_img):
    """Convierte PIL.Image a QPixmap de forma segura."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height,
                  pil_img.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contador de árboles — YOLO11n / HARV v2")
        self.resize(1100, 750)

        self.detector = TreeDetector()
        self.current_image = None

        central = QWidget()
        layout = QVBoxLayout(central)

        # --- Barra superior: botón + umbral ---
        top = QHBoxLayout()
        self.btn_open = QPushButton("Subir ortofoto…")
        self.btn_open.clicked.connect(self.open_image)
        top.addWidget(self.btn_open)

        top.addWidget(QLabel("Confianza mínima:"))
        self.spin_conf = QSpinBox()
        self.spin_conf.setRange(10, 90)   # 10%–90%
        self.spin_conf.setValue(30)       # default 30%
        self.spin_conf.setSuffix(" %")
        self.spin_conf.valueChanged.connect(self.update_conf)
        top.addWidget(self.spin_conf)
        top.addStretch()
        layout.addLayout(top)

        # --- Preview de imagen ---
        self.label_img = QLabel("Sube una ortofoto para empezar")
        self.label_img.setAlignment(Qt.AlignCenter)
        self.label_img.setMinimumSize(900, 550)
        self.label_img.setStyleSheet("border: 1px solid #888; background: #f0f0f0;")
        layout.addWidget(self.label_img)

        # --- Resultado (conteo) ---
        self.label_result = QLabel("Árboles detectados: —")
        self.label_result.setStyleSheet("font-size: 22px; font-weight: bold;")
        self.label_result.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_result)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------
    def update_conf(self):
        self.detector.conf = self.spin_conf.value() / 100.0
        print(f"[conf] {self.detector.conf:.2f}")

    # ------------------------------------------------------------------
    def open_image(self):
        print("[1] open_image")
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecciona ortofoto", "",
            "Imágenes (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        print(f"[2] path={path}")
        if not path:
            return
        try:
            self.current_image = path
            print("[3] prediciendo…")
            annotated, n, boxes = self.detector.predict(path)
            print(f"[4] {n} detecciones")
            pix = pil_to_qpixmap(annotated)
            print(f"[5] QPixmap {pix.width()}x{pix.height()}")
            self.label_img.setPixmap(pix.scaled(
                self.label_img.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))
            self.label_result.setText(f"Árboles detectados: {n}")
            print("[6] UI actualizada")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback; traceback.print_exc()
            QMessageBox.critical(self, "Error", f"No se pudo procesar:\n{e}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()