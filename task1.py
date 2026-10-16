"""
Завдання 1: Лінійна регресія для прогнозування споживання електроенергії.
Датасет: energy_usage.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

# Забезпечення коректного UTF-8 виведення у Windows-консолі
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Налаштування стилю графіків
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def main():
    print("=" * 60)
    print("ЗАВДАННЯ 1: Базова модель лінійної регресії")
    print("=" * 60)

    # 1. Завантаження даних
    file_path = "energy_usage.csv"
    print(f"\n[1] Завантаження даних з файлу '{file_path}'...")
    df = pd.read_csv(file_path)

    print("\nПерші 5 рядків датасету:")
    print(df.head())

    print("\nЗагальна інформація про датасет:")
    df.info()

    print("\nСтатистичний опис числових ознак:")
    print(df.describe())

    # 2. Розділення на ознаки (X) та цільову змінну (y)
    features = ["temperature", "humidity", "hour", "is_weekend"]
    target = "consumption"

    X = df[features]
    y = df[target]

    print(f"\n[2] Ознаки: {features}")
    print(f"Цільова змінна: {target}")

    # 3. Розбиття на навчальну та тестову вибірки (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[3] Розмір тренувальної вибірки: {X_train.shape[0]} записів")
    print(f"Розмір тестової вибірки: {X_test.shape[0]} записів")

    # 4. Навчання моделі лінійної регресії
    print("\n[4] Навчання моделі LinearRegression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Виведення коефіцієнтів
    print("\nКоефіцієнти лінійної регресії:")
    for col, coef in zip(features, model.coef_):
        print(f"  • {col:12s}: {coef:+.4f}")
    print(f"  • Інтерсепт (b0): {model.intercept_:+.4f}")

    # 5. Прогнозування та оцінка якості моделі
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("МЕТРИКИ ЯКОСТІ МОДЕЛІ (ТЕСТОВА ВИБІРКА):")
    print(f"  • Середня абсолютна помилка (MAE):        {mae:.2f} кВт·год")
    print(f"  • Середньоквадратична помилка (MSE):       {mse:.2f}")
    print(f"  • Корінь з MSE (RMSE):                    {rmse:.2f} кВт·год")
    print(f"  • Відсоток помилки (MAPE):                {mape:.2f}%")
    print(f"  • Коефіцієнт детермінації (R² score):     {r2:.4f}")
    print("=" * 60)

    # 6. Побудова графіків
    print("\n[5] Побудова графіків «Справжня vs Прогнозована ціна/споживання»...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Графік 1: Scatter plot True vs Predicted
    min_val = min(y_test.min(), y_pred.min()) - 20
    max_val = max(y_test.max(), y_pred.max()) + 20

    sns.scatterplot(
        x=y_test,
        y=y_pred,
        alpha=0.6,
        color="#1f77b4",
        edgecolor="w",
        s=60,
        ax=axes[0],
    )
    axes[0].plot(
        [min_val, max_val],
        [min_val, max_val],
        color="crimson",
        linestyle="--",
        linewidth=2,
        label="Ідеальний прогноз (y = x)",
    )
    axes[0].set_title(
        f"Завдання 1: Справжнє vs Прогнозоване споживання\n(R² = {r2:.4f}, Помилка MAPE = {mape:.2f}%)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Справжнє споживання (кВт·год)", fontweight="bold")
    axes[0].set_ylabel("Прогнозоване споживання (кВт·год)", fontweight="bold")
    axes[0].legend()
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)

    # Графік 2: Порівняння перших 50 точок тестової вибірки
    sample_df = pd.DataFrame(
        {
            "Справжнє": y_test.values[:60],
            "Прогнозоване": y_pred[:60],
        }
    ).reset_index(drop=True)

    axes[1].plot(
        sample_df["Справжнє"],
        label="Справжнє значення",
        marker="o",
        color="#2ca02c",
        linewidth=1.8,
        markersize=4,
    )
    axes[1].plot(
        sample_df["Прогнозоване"],
        label="Прогноз моделі",
        marker="x",
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        markersize=5,
    )
    axes[1].set_title(
        "Порівняння значень на 60 тестових прикладах",
        fontweight="bold",
    )
    axes[1].set_xlabel("Номер зразка", fontweight="bold")
    axes[1].set_ylabel("Споживання (кВт·год)", fontweight="bold")
    axes[1].legend()

    plot_file = "task1_actual_vs_predicted.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Графік збережено у файл: '{plot_file}'")


if __name__ == "__main__":
    main()
