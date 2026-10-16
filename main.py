"""
Головний скрипт для запуску Завдання 1 та Завдання 2.
"""

import sys
import subprocess

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    print("=" * 70)
    print("   IT STEP ACADEMY — ML: LINEAR REGRESSION (ELECTRICITY USAGE)")
    print("=" * 70)
    print("\n[+] Запуск Завдання 1...")
    import task1
    task1.main()

    print("\n" + "#" * 70 + "\n")

    print("[+] Запуск Завдання 2...")
    import task2
    task2.main()

    print("\n" + "=" * 70)
    print("Всі завдання успішно виконані!")
    print("Створені файли графіків:")
    print("  1. task1_actual_vs_predicted.png")
    print("  2. task2_actual_vs_predicted.png")
    print("  3. task2_feature_coefficients.png")
    print("  4. solution.ipynb (Jupyter Notebook)")
    print("=" * 70)


if __name__ == "__main__":
    main()
