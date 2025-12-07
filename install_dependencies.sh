#!/usr/bin/env bash
set -e

echo "▶ Определяю интерпретатор Python…"

if command -v python3 &>/dev/null; then
  PYTHON_BIN="python3"
elif command -v python &>/dev/null; then
  PYTHON_BIN="python"
else
  echo "❌ Python не найден. Установи его (например, через brew: brew install python)"
  exit 1
fi

echo "✅ Использую интерпретатор: $PYTHON_BIN"

VENV_DIR=".venv"

echo "▶ Создаю виртуальную среду в $VENV_DIR…"
$PYTHON_BIN -m venv "$VENV_DIR"

echo "▶ Активирую виртуальную среду…"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "▶ Обновляю pip…"
pip install --upgrade pip

echo "▶ Устанавливаю зависимости (pytesseract, pillow)…"
pip install pytesseract pillow

echo "▶ Устанавливаю зависимости (entence-transformers, numpy)…"
pip install sentence-transformers numpy faiss-cpu

echo
echo "✅ Готово!"
echo "Чтобы в следующий раз активировать среду, используй:"
echo "  source $VENV_DIR/bin/activate"
