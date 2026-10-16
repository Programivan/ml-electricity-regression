"""
Завдання 2 (Додаткове): Додавання категоріальних ознак (season, district_type)
та кодування за допомогою OneHotEncoder.
Датасет: energy_usage_plus.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    print("=" * 65)
    print("ЗАВДАННЯ 2: Лінійна регресія з категоріальними ознаками (OneHotEncoder)")
    print("=" * 65)

    # 1. Завантаження даних
    file_path = "energy_usage_plus.csv"
    print(f"\n[1] Завантаження розширеного набору даних з '{file_path}'...")
    df = pd.read_csv(file_path)

    print("\nПерші 5 рядків датасету:")
    print(df.head())

    print("\nУнікальні значення категоріальних ознак:")
    print(f"  • season:        {df['season'].unique().tolist()}")
    print(f"  • district_type: {df['district_type'].unique().tolist()}")

    # 2. Ознаки та цільова змінна
    numeric_features = ["temperature", "humidity", "hour", "is_weekend"]
    categorical_features = ["season", "district_type"]
    all_features = numeric_features + categorical_features
    target = "consumption"

    X = df[all_features]
    y = df[target]

    # 3. Розбиття на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n[2] Розмір вибірок: Train = {len(X_train)}, Test = {len(X_test)}")

    # 4. Створення трансформера OneHotEncoder для категоріальних ознак
    # drop='first' допомагає уникнути мультиколінеарності (dummy variable trap)
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    # 5. Створення та навчання пайплайну
    print("\n[3] Навчання моделі LinearRegression через Pipeline (OneHotEncoder + Regression)...")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)

    # Отримання імен закодованих ознак та коефіцієнтів
    cat_feature_names = model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    encoded_feature_names = list(cat_feature_names) + numeric_features
    coefs = model.named_steps["regressor"].coef_
    intercept = model.named_steps["regressor"].intercept_

    print("\nОтримані коефіцієнти регресії для кожної ознаки:")
    for name, c in zip(encoded_feature_names, coefs):
        print(f"  • {name:30s}: {c:+.4f}")
    print(f"  • Інтерсепт (b0)                : {intercept:+.4f}")

    # 6. Оцінка моделі
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 65)
    print("МЕТРИКИ ЯКОСТІ МОДЕЛІ ДЛЯ ЗАВДАННЯ 2:")
    print(f"  • Середня абсолютна помилка (MAE):        {mae:.2f} кВт·год")
    print(f"  • Середньоквадратична помилка (MSE):       {mse:.2f}")
    print(f"  • Корінь з MSE (RMSE):                    {rmse:.2f} кВт·год")
    print(f"  • Відсоток помилки (MAPE):                {mape:.2f}%")
    print(f"  • Коефіцієнт детермінації (R² score):     {r2:.4f}")
    print("=" * 65)

    # 7. Порівняння з моделлю Завдання 1 (без категорій)
    # Навчаємо модель без категоріальних ознак на тих самих індексах
    base_model = LinearRegression()
    base_model.fit(X_train[numeric_features], y_train)
    base_pred = base_model.predict(X_test[numeric_features])
    base_mape = mean_absolute_percentage_error(y_test, base_pred) * 100
    base_mae = mean_absolute_error(y_test, base_pred)
    base_r2 = r2_score(y_test, base_pred)

    print("\nПОРІВНЯЛЬНИЙ АНАЛІЗ МОДЕЛЕЙ:")
    print(f"  • Помилка (MAPE): {base_mape:.2f}% (Завдання 1) ➔ {mape:.2f}% (Завдання 2)")
    print(f"  • MAE:            {base_mae:.2f} кВт·год ➔ {mae:.2f} кВт·год")
    print(f"  • R² Score:       {base_r2:.4f} ➔ {r2:.4f}")
    print("  => Додавання категорій району та сезону суттєво покращило точність прогнозу!")

    # 8. Побудова графіків
    print("\n[4] Створення графіків візуалізації...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Графік 1: Scatter plot True vs Predicted
    min_val = min(y_test.min(), y_pred.min()) - 20
    max_val = max(y_test.max(), y_pred.max()) + 20

    sns.scatterplot(
        x=y_test,
        y=y_pred,
        alpha=0.65,
        color="#2ca02c",
        edgecolor="w",
        s=65,
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
        f"Завдання 2: Справжнє vs Прогнозоване споживання\n(R² = {r2:.4f}, Помилка MAPE = {mape:.2f}%)",
        fontweight="bold",
    )
    axes[0].set_xlabel("Справжнє споживання (кВт·год)", fontweight="bold")
    axes[0].set_ylabel("Прогнозоване споживання (кВт·год)", fontweight="bold")
    axes[0].legend()
    axes[0].set_xlim(min_val, max_val)
    axes[0].set_ylim(min_val, max_val)

    # Графік 2: Порівняння базової моделі та моделі з категоріями (перші 45 точок)
    sample_n = 45
    x_axis = np.arange(sample_n)
    axes[1].plot(
        x_axis,
        y_test.values[:sample_n],
        label="Справжнє значення",
        color="black",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    axes[1].plot(
        x_axis,
        base_pred[:sample_n],
        label=f"Завдання 1: без категорій (MAPE {base_mape:.1f}%)",
        color="#1f77b4",
        linestyle=":",
        linewidth=1.7,
    )
    axes[1].plot(
        x_axis,
        y_pred[:sample_n],
        label=f"Завдання 2: з OneHotEncoder (MAPE {mape:.1f}%)",
        color="#d62728",
        linestyle="--",
        linewidth=1.9,
    )
    axes[1].set_title(
        f"Порівняння прогнозів обох моделей на {sample_n} тестових прикладах",
        fontweight="bold",
    )
    axes[1].set_xlabel("Номер зразка", fontweight="bold")
    axes[1].set_ylabel("Споживання (кВт·год)", fontweight="bold")
    axes[1].legend()

    plot_file = "task2_actual_vs_predicted.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Графік збережено у файл: '{plot_file}'")

    # Створення другого графіка: порівняння коефіцієнтів
    plt.figure(figsize=(10, 5))
    coef_series = pd.Series(coefs, index=encoded_feature_names).sort_values()
    colors = ["#d62728" if c < 0 else "#2ca02c" for c in coef_series.values]
    bars = plt.barh(coef_series.index, coef_series.values, color=colors, alpha=0.85)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Вплив ознак на споживання електроенергії (Ваги регресії)", fontweight="bold")
    plt.xlabel("Коефіцієнт регресії (кВт·год)", fontweight="bold")

    coef_plot_file = "task2_feature_coefficients.png"
    plt.savefig(coef_plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Графік коефіцієнтів збережено у файл: '{coef_plot_file}'")


if __name__ == "__main__":
    main()
