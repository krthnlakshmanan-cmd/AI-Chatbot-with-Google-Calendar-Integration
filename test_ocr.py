import pytesseract
import os

path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print("File exists:", os.path.isfile(path))

pytesseract.pytesseract.tesseract_cmd = path

print("Tesseract version:", pytesseract.get_tesseract_version())