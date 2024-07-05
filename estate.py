import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import simpledialog
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import numpy as np
from PIL import Image, ImageTk
import torch 
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import pickle

class LinearModel(nn.Module):
    def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(562, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 32),
                nn.LeakyReLU(),
                nn.Linear(32, 1)
        )
    def forward(self, X):
        return self.network(X)

# Глобальные переменные для DataFrame и Treeview
df = None
tree = None

#Главная форма
def open_main_application():
    loading_screen.destroy()
    root = tk.Tk()
    root.title("EstateOracle")
    root.geometry("1280x720")
    canvas = tk.Canvas(root, width=1280, height=720)
    canvas.pack()
    background_image = tk.PhotoImage(file="C:/Users/gorox/Desktop/EstateOracle/background.png")
    canvas.create_image(0, 0, anchor=tk.NW, image=background_image)
        # Установка размеров изображения на размеры Canvas
    canvas.image = background_image
    canvas.config(width=canvas.image.width(), height=canvas.image.height())

    welcome_label = tk.Label(root, text="Добро пожаловать", font=("Arial", 64), bg="#f0cbbb", fg="white")
    welcome_label.place(relx=0.5, rely=0.1, anchor="center")

    activity_label = tk.Label(root, text="Чем займёмся сегодня", font=("Arial", 40), bg="#f0cbbb", fg="white")
    activity_label.place(relx=0.5, rely=0.2, anchor="center")

    data_button = tk.Button(root, text="Просмотр данных", font=("Arial", 40), bg="#8CD19D", fg="white", command=view_data)
    data_button.place(relx=0.5, rely=0.35, anchor="center")

    analytics_button = tk.Button(root, text="Инфографика", font=("Arial", 40), bg="#6FA0DD", fg="white", command=open_analytics)
    analytics_button.place(relx=0.5, rely=0.55, anchor="center")

    forecast_button = tk.Button(root, text="Прогнозирование", font=("Arial", 40), bg="#FFA500", fg="white", command=open_forecasting_window)
    forecast_button.place(relx=0.5, rely=0.75, anchor="center")

    logout_button = tk.Button(root, text="Выйти", font=("Arial", 20), bg="red", fg="white", command=root.destroy)
    logout_button.place(relx=0.05, rely=0.95, anchor="sw")
    root.mainloop()

# Форма просмотра данных
def view_data():
    
    global df, tree
    # Создаем новое окно для отображения данных
    data_window = tk.Toplevel()
    data_window.title("Просмотр данных")
    data_window.geometry("1280x720")

    # Читаем данные из файла CSV с помощью pandas
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл с данными не найден, поместите файл с названием data в каталог программы")
        return

    # Создаем таблицу для отображения данных
    tree = ttk.Treeview(data_window, columns=list(df.columns), show="headings")

    # Добавляем заголовки столбцов
    for col in df.columns:
        tree.heading(col, text=col, command=lambda c=col: sort_column(tree, c))

    # Добавляем данные в таблицу
    for index, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Размещаем таблицу на окне
    tree.pack(expand=True, fill="both")

    def edit_entry(event):
        item = tree.selection()[0]
        # Получаем текущие данные из выделенной строки
        values = tree.item(item, "values")
        # Предлагаем пользователю внести изменения
        new_values = simpledialog.askstring("Редактирование записи", "Введите новые значения через запятую", initialvalue=",".join(values))
        if new_values:
            # Разделяем введенные пользователем значения по запятой и обновляем данные в таблице
            new_values = new_values.split(",")
            tree.item(item, values=new_values)

    # Привязываем обработчик события (выделение строки) к функции изменения данных
    tree.bind("<Double-1>", edit_entry)

    # Создаем кнопку для добавления записи
    add_button = ttk.Button(data_window, text="Добавить запись", command=add_entry)
    add_button.pack(side="right", padx=10, pady=10)  # кнопка будет слева с отступом по горизонтали 10 и вертикали 10

    # Создаем кнопку для удаления записи
    delete_button = ttk.Button(data_window, text="Удалить запись", command=delete_entry)
    delete_button.pack(side="left", padx=10, pady=10)  # кнопка будет справа с отступом по горизонтали 10 и вертикали 10
    
    # Добавляем фильтры и кнопку для поиска
    filter_frame = ttk.Frame(data_window)
    filter_frame.pack(pady=15)

    # Создаем виджеты для ввода значений параметров
    
    # Создаем виджеты для ввода диапазона цен
    min_price_label = ttk.Label(filter_frame, text="Мин. цена:")
    min_price_label.grid(row=0, column=0, padx=5, pady=5)
    min_price_entry = ttk.Entry(filter_frame)
    min_price_entry.grid(row=0, column=1, padx=5, pady=5)

    max_price_label = ttk.Label(filter_frame, text="Макс. цена:")
    max_price_label.grid(row=0, column=2, padx=5, pady=5)
    max_price_entry = ttk.Entry(filter_frame)
    max_price_entry.grid(row=0, column=3, padx=5, pady=5)

    rooms_label = ttk.Label(filter_frame, text="Количество комнат:")
    rooms_label.grid(row=1, column=0, padx=5, pady=5)
    rooms_entry = ttk.Entry(filter_frame)
    rooms_entry.grid(row=1, column=1, padx=5, pady=5)

    area_label = ttk.Label(filter_frame, text="Площадь:")
    area_label.grid(row=1, column=2, padx=5, pady=5)
    area_entry = ttk.Entry(filter_frame)
    area_entry.grid(row=1, column=3, padx=5, pady=5)

    type_label = ttk.Label(filter_frame, text="Тип апартаментов:")
    type_label.grid(row=2, column=0, padx=5, pady=5)
    type_values = df['Apartment type'].unique().tolist()  # Получаем уникальные значения столбца Apartment type
    type_combobox = ttk.Combobox(filter_frame, values=type_values)
    type_combobox.grid(row=2, column=1, padx=5, pady=5)

    renovation_label = ttk.Label(filter_frame, text="Ремонт:")
    renovation_label.grid(row=2, column=2, padx=5, pady=5)
    renovation_values = df['Renovation'].unique().tolist()  # Получаем уникальные значения столбца Renovation
    renovation_combobox = ttk.Combobox(filter_frame, values=renovation_values)
    renovation_combobox.grid(row=2, column=3, padx=5, pady=5)

    floor_label = ttk.Label(filter_frame, text="Этаж:")
    floor_label.grid(row=3, column=0, padx=5, pady=5)
    floor_entry = ttk.Entry(filter_frame)
    floor_entry.grid(row=3, column=1, padx=5, pady=5)

    metro_label = ttk.Label(filter_frame, text="Станция метро:")
    metro_label.grid(row=3, column=2, padx=5, pady=5)
    metro_values = sorted(df['Metro station'].unique().tolist())
    metro_combobox = ttk.Combobox(filter_frame, values = metro_values)
    metro_combobox.grid(row=3, column=3, padx=5, pady=5)

    

    # Функция для выполнения поиска
    def search():
        # Получаем значения параметров из виджетов
        min_price = min_price_entry.get()
        max_price = max_price_entry.get()
        type = type_combobox.get()
        rooms = rooms_entry.get()
        area = area_entry.get()
        floor = floor_entry.get()
        metro = metro_combobox.get()
        renovation = renovation_combobox.get()

        # Фильтруем DataFrame в соответствии с введенными значениями
        filtered_df = df
        if min_price and max_price:
            filtered_df = filtered_df[(filtered_df['Price'] >= float(min_price)) & (filtered_df['Price'] <= float(max_price))]
        if type:
            filtered_df = filtered_df[filtered_df['Apartment type'] == type]
        if rooms:
            filtered_df = filtered_df[filtered_df['Number of rooms'] == int(rooms)]
        if area:
            filtered_df = filtered_df[filtered_df['Area'] == float(area)]
        if floor:
            filtered_df = filtered_df[filtered_df['Floor'] == int(floor)]
        if metro:
            filtered_df = filtered_df[filtered_df['Metro station'] == metro]
        if renovation:
            filtered_df = filtered_df[filtered_df['Renovation'] == renovation]

        # Очищаем текущее отображение таблицы
        for row in tree.get_children():
            tree.delete(row)
        
        # Добавляем отфильтрованные данные в таблицу
        for index, row in filtered_df.iterrows():
            tree.insert("", "end", values=list(row))

    # Функция для сброса фильтров
    def reset_filters():
        # Очищаем значения виджетов фильтров
        min_price_entry.delete(0, tk.END)
        max_price_entry.delete(0, tk.END)
        type_combobox.set('')
        rooms_entry.delete(0, tk.END)
        area_entry.delete(0, tk.END)
        floor_entry.delete(0, tk.END)
        metro_combobox.set('')
        renovation_combobox.set('')
        # Повторно запускаем поиск без фильтров
        search()

    # Кнопка для выполнения поиска
    search_button = ttk.Button(filter_frame, text="Поиск", command=search)
    search_button.grid(row=4, column=1, padx=5, pady=5)

    # Кнопка для сброса фильтров
    reset_button = ttk.Button(filter_frame, text="Сброс", command=reset_filters)
    reset_button.grid(row=4, column=2, padx=5, pady=5)


# Сортировка стобцов в таблице
def sort_column(tree, col):
    """Сортировка данных в таблице по указанному столбцу."""
    # Получаем текущий порядок сортировки для столбца
    current_order = tree.heading(col)["text"]

    # Определяем, какой порядок сортировки должен быть следующим
    if current_order.endswith(" ↓"):
        reverse = False  # Сортируем по возрастанию
    else:
        reverse = True  # Сортируем по убыванию

    # Функция для преобразования значений в числовой тип
    def convert_to_float(value):
        try:
            return float(value)
        except ValueError:
            return value

    # Сортируем данные по указанному столбцу, преобразуя их к числовому типу
    data = [(convert_to_float(tree.set(child, col)), child) for child in tree.get_children("")]
    data.sort(reverse=reverse)
    for index, (val, child) in enumerate(data):
        tree.move(child, "", index)

    # Обновляем заголовок столбца с учетом текущего порядка сортировки
    new_order = " ↓" if reverse else " ↑"
    tree.heading(col, text=col + new_order)

def add_entry():
    global df, tree
    # Функция для добавления новой записи в таблицу
    def add():
        # Получаем значения из полей ввода
        new_values = [
            price_entry.get(),
            type_combobox.get(),
            metro_entry.get(),
            minutes_entry.get(),
            region_entry.get(),
            rooms_entry.get(),
            area_entry.get(),
            living_area_entry.get(),
            kitchen_area_entry.get(),
            floor_entry.get(),
            num_floors_entry.get(),
            renovation_combobox.get()
        ]
        # Проверяем, что все поля заполнены
        if all(new_values):
            # Добавляем новую строку в DataFrame
            df.loc[len(df)] = new_values
            # Добавляем новую строку в таблицу
            tree.insert("", "end", values=new_values)
            # Закрываем диалоговое окно
            add_window.destroy()
        else:
            messagebox.showerror("Ошибка", "Пожалуйста, заполните все поля.")

    # Создаем диалоговое окно для добавления записи
    add_window = tk.Toplevel()
    add_window.title("Добавление записи")
    add_window.configure(bg="#f0cbbb")

    # Создаем виджеты для ввода данных
    price_label = ttk.Label(add_window, text="Цена:")
    price_label.grid(row=0, column=0, padx=5, pady=5)
    price_entry = ttk.Entry(add_window)
    price_entry.grid(row=0, column=1, padx=5, pady=5)

    type_label = ttk.Label(add_window, text="Тип апартаментов:")
    type_label.grid(row=1, column=0, padx=5, pady=5)
    type_values = df['Apartment type'].unique().tolist()
    type_combobox = ttk.Combobox(add_window, values=type_values)
    type_combobox.grid(row=1, column=1, padx=5, pady=5)

    metro_label = ttk.Label(add_window, text="Метро:")
    metro_label.grid(row=2, column=0, padx=5, pady=5)
    metro_entry = ttk.Entry(add_window)
    metro_entry.grid(row=2, column=1, padx=5, pady=5)

    minutes_label = ttk.Label(add_window, text="Минут до метро:")
    minutes_label.grid(row=3, column=0, padx=5, pady=5)
    minutes_entry = ttk.Entry(add_window)
    minutes_entry.grid(row=3, column=1, padx=5, pady=5)

    region_label = ttk.Label(add_window, text="Район:")
    region_label.grid(row=4, column=0, padx=5, pady=5)
    region_entry = ttk.Entry(add_window)
    region_entry.grid(row=4, column=1, padx=5, pady=5)

    rooms_label = ttk.Label(add_window, text="Количество комнат:")
    rooms_label.grid(row=5, column=0, padx=5, pady=5)
    rooms_entry = ttk.Entry(add_window)
    rooms_entry.grid(row=5, column=1, padx=5, pady=5)

    area_label = ttk.Label(add_window, text="Площадь:")
    area_label.grid(row=6, column=0, padx=5, pady=5)
    area_entry = ttk.Entry(add_window)
    area_entry.grid(row=6, column=1, padx=5, pady=5)

    living_area_label = ttk.Label(add_window, text="Жилая площадь:")
    living_area_label.grid(row=7, column=0, padx=5, pady=5)
    living_area_entry = ttk.Entry(add_window)
    living_area_entry.grid(row=7, column=1, padx=5, pady=5)

    kitchen_area_label = ttk.Label(add_window, text="Площадь кухни:")
    kitchen_area_label.grid(row=8, column=0, padx=5, pady=5)
    kitchen_area_entry = ttk.Entry(add_window)
    kitchen_area_entry.grid(row=8, column=1, padx=5, pady=5)

    floor_label = ttk.Label(add_window, text="Этаж:")
    floor_label.grid(row=9, column=0, padx=5, pady=5)
    floor_entry = ttk.Entry(add_window)
    floor_entry.grid(row=9, column=1, padx=5, pady=5)

    num_floors_label = ttk.Label(add_window, text="Этажность:")
    num_floors_label.grid(row=10, column=0, padx=5, pady=5)
    num_floors_entry = ttk.Entry(add_window)
    num_floors_entry.grid(row=10, column=1, padx=5, pady=5)

    renovation_label = ttk.Label(add_window, text="Ремонт:")
    renovation_label.grid(row=11, column=0, padx=5, pady=5)
    renovation_values = df['Renovation'].unique().tolist()
    renovation_combobox = ttk.Combobox(add_window, values=renovation_values)
    renovation_combobox.grid(row=11, column=1, padx=5, pady=5)

    # Создаем кнопку для добавления записи
    add_button = ttk.Button(add_window, text="Добавить", command=add)
    add_button.grid(row=12, columnspan=2, padx=5, pady=10)

def delete_entry():
    global df, tree
    # Функция для удаления выделенной записи из таблицы и DataFrame
    def delete():
        # Получаем выделенную запись
        selected_item = tree.selection()
        if selected_item:
            # Удаляем запись из DataFrame и таблицы только если пользователь подтвердил удаление
            if messagebox.askyesno("Подтверждение удаления", "Вы уверены, что хотите удалить выбранную запись?"):
                # Удаляем запись из DataFrame
                index = tree.index(selected_item)
                df.drop(df.index[index], inplace=True)
                # Удаляем запись из таблицы
                tree.delete(selected_item)

    # Создаем диалоговое окно для удаления записи
    delete_window = tk.Toplevel()
    delete_window.title("Удаление записи")

    # Создаем кнопку для удаления записи
    delete_button = ttk.Button(delete_window, text="Удалить", command=delete)
    delete_button.pack(padx=10, pady=5)

# Форма "Инфографика"
def open_analytics():
    analytics_window = tk.Toplevel()
    analytics_window.title("Инфографика")
    analytics_window.geometry("1280x720")

    canvas = tk.Canvas(analytics_window, width=1280, height=720)
    canvas.pack()

    background_image = tk.PhotoImage(file="C:/Users/gorox/Desktop/EstateOracle/background.png")
    canvas.create_image(0, 0, anchor=tk.NW, image=background_image)

    analytics_label = tk.Label(analytics_window, text="Инфографика", font=("Arial", 64), bg="#f0cbbb", fg="white")
    analytics_label.place(relx=0.5, rely=0.1, anchor="center")

    # Установка размеров изображения на размеры Canvas
    canvas.image = background_image
    canvas.config(width=canvas.image.width(), height=canvas.image.height())

    # Размещение кнопок "столбиков"
    button1 = tk.Button(analytics_window, text="Распределение цены", font=("Arial", 30), bg="#f0cbbb", fg="white", command=show_3d_bar)
    button1.place(relx=0.3, rely=0.3, anchor="center")

    button2 = tk.Button(analytics_window, text="Процентное соотношение \n квартир по количеству комнат", font=("Arial", 30), bg="#f0cbbb", fg="white", command=plot_room_distribution)
    button2.place(relx=0.3, rely=0.5, anchor="center")

    button_3 = tk.Button(analytics_window, text="Взаимосвязь цены \nи площади", font=("Arial", 30), bg="#f0cbbb", fg="white", command=show_scatter_plot)
    button_3.place(relx=0.3, rely=0.7, anchor="center")

    button4 = tk.Button(analytics_window, text="Распределение площади", font=("Arial", 30), bg="#f0cbbb", fg="white", command=plot_area_distribution)
    button4.place(relx=0.7, rely=0.3, anchor="center")

    button5 = tk.Button(analytics_window, text="Взаимосвязь цены \nи кол-ва комнат", font=("Arial", 30), bg="#f0cbbb", fg="white", command=plot_time_to_metro)
    button5.place(relx=0.7, rely=0.5, anchor="center")

    button6 = tk.Button(analytics_window, text="Процентное соотношение \n квартир по качеству ремонта", font=("Arial", 30), bg="#f0cbbb", fg="white", command=plot_renovation_distribution)
    button6.place(relx=0.7, rely=0.7, anchor="center")

def show_3d_bar():
    plt.figure("1280x720")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл с данными не найден")
        return

    # Агрегируем данные по количеству квартир для каждой комбинации количества комнат и цены
    aggregated_data = df.groupby(['Number of rooms', 'Price']).size().reset_index(name='Count')

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = aggregated_data['Number of rooms']
    y = aggregated_data['Price']
    z = np.zeros(len(aggregated_data))  # Для каждой комбинации создаем начальную высоту столбца

    dx = dy = 0.5  # Ширина и высота столбцов

    ax.bar3d(x, y, z, dx, dy, aggregated_data['Count'], color='skyblue', alpha=0.8)
    # Устанавливаем шкалу для оси X с каждой единицей
    ax.set_xticks(range(int(df['Number of rooms'].min()), int(df['Number of rooms'].max()) + 1))

    ax.set_xlabel('Количество комнат')
    ax.set_ylabel('Цена')
    ax.set_zlabel('Количество квартир')

    plt.title('Оптимизированная трехмерная столбчатая диаграмма')
    plt.show()



def plot_room_distribution():
    plt.figure("1280x720")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл с данными не найден")
        return
    
    # Преобразуем значения комнат в числовой тип данных
    df['Number of rooms'] = pd.to_numeric(df['Number of rooms'], errors='coerce')
    # Заменяем значения больше 6 на "7+"
    df['Number of rooms'] = df['Number of rooms'].apply(lambda x: f"{int(x)} комнатные квартиры" if isinstance(x, float) and x >= 1 and x <= 6 else '7+ комнатные квартиры' if isinstance(x, float) and x > 6 else 'Студии')
    # Подсчитываем количество квартир для каждого количества комнат
    room_counts = df['Number of rooms'].value_counts() 
    # Создаем круговую диаграмму
    plt.figure(figsize=(6, 6))
    plt.title('Процентное соотношение квартир по количеству комнат', pad=50)
    labels = [f"{label}\n({count} квартир)" for label, count in zip(room_counts.index, room_counts)]
    explode = (0.1, 0.1, 0.1, 0.1,0,0,0,0)
    plt.pie(room_counts, labels=room_counts, autopct='%1.1f%%', startangle=140, labeldistance=1.1, explode=explode)
    plt.legend(room_counts.index, title="Количество комнат", loc="center right", bbox_to_anchor=(1, 0.5))
    plt.axis('equal')  # Сделаем круговую диаграмму круглой
    plt.show()

# График Цена ~ площадь
def show_scatter_plot():
    plt.figure("1280x720")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл с данными не найден")
        return
    # Создание объекта фигуры Matplotlib
    fig, ax = plt.subplots(figsize=(16, 9))
    # Построение точечной диаграммы
    sns.scatterplot(data=df, x='Area', y='Price', ax=ax, color='g')
    # Расчет и вывод корреляции между ценой и площадью
    correlation = round(df[['Area', 'Price']].corr()['Price'].loc['Area'], 2)
    ax.set_title(f'Взаимосвязь цены и площади: {correlation}')
    ax.set_xlabel('Площадь (м^2)')
    ax.set_ylabel('Цена (млрд)')
    # Показать график
    plt.show()

# График Распределения площади
def plot_area_distribution():
    plt.figure("1280x720")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл с данными не найден")
        return
    # Создание графика
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.histplot(data=df, x='Area', bins=50, ax=ax, color='g')
    ax.set_title('Распределение площади')
    ax.set_ylabel('Количество квартир')
    ax.set_xlabel('Площадь (м^2)')
    ax.set_xticks(range(0, 1001, 100))
    # Отображение графика
    plt.show()    

# График Время до метро
def plot_time_to_metro():
    plt.figure("1280x720")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        print("Файл с данными не найден")
        return
    
    # Создаем список интервалов времени
    time_intervals = ['Меньше 5 минут', '5-10 минут', '10-20 минут', '20-30 минут', 'Больше 30 минут']
    
    # Создаем столбец с интервалами времени до метро
    df['Time to metro interval'] = pd.cut(df['Minutes to metro'], bins=[0, 5, 10, 20, 30, float('inf')], labels=time_intervals)
    
    # Подсчитываем количество квартир в каждом интервале
    time_counts = df['Time to metro interval'].value_counts()
    
    # Создаем круговую диаграмму
    plt.figure(figsize=(6, 6))
    plt.title('Процентное соотношение времени до метро', pad=50)
    labels = [f"{label}\n({count} квартир)" for label, count in zip(time_counts.index, time_counts)]
    explode = (0.1, 0.1, 0.1, 0,0)
    patches, texts, autotexts = plt.pie(time_counts, labels=labels, autopct='%1.1f%%', startangle=140, explode=explode)
    plt.axis('equal')
    
    # Добавляем подписи с процентным соотношением для каждого сегмента
    for autotext in autotexts:
        autotext.set_horizontalalignment('center')
        autotext.set_verticalalignment('center')
    
    plt.legend(time_counts.index, title="Время до метро", loc="right", bbox_to_anchor=(1, 0.5))
    plt.show()

# График Ремонт
def plot_renovation_distribution():
    plt.figure("1280x720")
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        messagebox.showerror("Ошибка", "Файл с данными не найден")
        return
    
    # Подсчитываем количество квартир для каждого типа ремонта
    renovation_counts = df['Renovation'].value_counts() 
    
    # Создаем круговую диаграмму
    plt.figure(figsize=(6, 6))
    plt.title('Процентное соотношение качества ремонта в квартирах', pad=50)
    labels = [f"{label.replace('European-style renovation', 'Евроремонт').replace('Cosmetic', 'Косметический ремонт').replace('Without renovation', 'Без ремонта').replace('Designer', 'Дизайнерский ремонт')}\n({count} квартир)" for label, count in zip(renovation_counts.index, renovation_counts)]
    plt.pie(renovation_counts, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Сделаем круговую диаграмму круглой
    plt.legend(['Косметический ремонт', 'Евроремонт', 'Без ремонта', 'Дизайнерский ремонт'], title="Тип ремонта", loc="right", bbox_to_anchor=(1, 0.5))
    plt.show()


def open_forecasting_window():
    # Создание нового окна для ввода параметров недвижимости
    forecasting_window = tk.Toplevel()
    forecasting_window.title("Прогнозирование")
    forecasting_window.configure(bg="#f0cbbb")
    

    # Функция для загрузки данных из файла data.csv
    def load_data(column_name):
        with open('data.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return sorted(set(row[column_name] for row in reader))

    # Получение списков для выпадающих списков
    raw_apartment_types = load_data('Apartment type')
    raw_regions = load_data('Region')
    raw_renovations = load_data('Renovation')
    metro_stations = load_data('Metro station')

    # Отображение значений для параметров
    apartment_type_mapping = {
        "New building": "Новая",
        "Secondary": "Вторичная"
    }
    region_mapping = {
        "Moscow": "Москва",
        "Moscow region": "Московская область"
    }
    renovation_mapping = {
        "Cosmetic": "Косметический",
        "European-style renovation": "Евроремонт",
        "Designer": "Дизайнерский",
        "Without renovation": "Без ремонта"
    }

    apartment_types = [apartment_type_mapping.get(item, item) for item in raw_apartment_types]
    regions = [region_mapping.get(item, item) for item in raw_regions]
    renovations = [renovation_mapping.get(item, item) for item in raw_renovations]

  

    # Создание и размещение элементов формы для ввода параметров
    tk.Label(forecasting_window, text="Тип квартиры:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=1, column=0)
    apartment_type_entry = ttk.Combobox(forecasting_window, values=apartment_types)
    apartment_type_entry.grid(row=1, column=1)

    tk.Label(forecasting_window, text="Станция метро:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=2, column=0)
    metro_station_entry = ttk.Combobox(forecasting_window, values=metro_stations)
    metro_station_entry.grid(row=2, column=1)

    tk.Label(forecasting_window, text="Время до метро (минуты):", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=3, column=0)
    minutes_to_metro_entry = tk.Entry(forecasting_window)
    minutes_to_metro_entry.grid(row=3, column=1)

    tk.Label(forecasting_window, text="Регион:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=4, column=0)
    region_entry = ttk.Combobox(forecasting_window, values=regions)
    region_entry.grid(row=4, column=1)

    tk.Label(forecasting_window, text="Количество комнат:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=5, column=0)
    number_of_rooms_entry = tk.Entry(forecasting_window)
    number_of_rooms_entry.grid(row=5, column=1)

    tk.Label(forecasting_window, text="Общая площадь:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=6, column=0)
    area_entry = tk.Entry(forecasting_window)
    area_entry.grid(row=6, column=1)

    tk.Label(forecasting_window, text="Жилая площадь:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=7, column=0)
    living_area_entry = tk.Entry(forecasting_window)
    living_area_entry.grid(row=7, column=1)

    tk.Label(forecasting_window, text="Площадь кухни:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=8, column=0)
    kitchen_area_entry = tk.Entry(forecasting_window)
    kitchen_area_entry.grid(row=8, column=1)

    tk.Label(forecasting_window, text="Этаж:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=9, column=0)
    floor_entry = tk.Entry(forecasting_window)
    floor_entry.grid(row=9, column=1)

    tk.Label(forecasting_window, text="Этажность:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=10, column=0)
    number_of_floors_entry = tk.Entry(forecasting_window)
    number_of_floors_entry.grid(row=10, column=1)

    tk.Label(forecasting_window, text="Ремонт:", bg="#f0cbbb", fg="white", font=("Arial Black", 14)).grid(row=11, column=0)
    renovation_entry = ttk.Combobox(forecasting_window, values=renovations)
    renovation_entry.grid(row=11, column=1)

    def prognoz(device='cuda'):
        def vvod_dannyx(houses):
            aparts = pd.get_dummies(houses['Apartment type']) 
            with_aparts = houses.join(aparts)
            hs1 = with_aparts.drop(['Apartment type'], axis=1)

            metro = pd.get_dummies(hs1['Metro station']) 
            with_metro = hs1.join(metro)
            hs2 = with_metro.drop(['Metro station'], axis=1)

            reg = pd.get_dummies(hs2['Region']) 
            with_reg = hs2.join(reg)
            hs3 = with_reg.drop(['Region'], axis=1)

            ren = pd.get_dummies(hs3['Renovation']) 
            with_ren = hs3.join(ren)
            hs = with_ren.drop(['Renovation'], axis=1)
            return hs.dropna(how='any')

        # Получение введенных пользователем параметров
        data = pd.read_csv('data.csv')
        reverse_apartment_type_mapping = {v: k for k, v in apartment_type_mapping.items()}
        reverse_region_mapping = {v: k for k, v in region_mapping.items()}
        reverse_renovation_mapping = {v: k for k, v in renovation_mapping.items()}

        dict = {
            'Apartment type': [reverse_apartment_type_mapping.get(apartment_type_entry.get(), apartment_type_entry.get())],
            'Metro station': [metro_station_entry.get()],
            'Minutes to metro': [float(minutes_to_metro_entry.get())],
            'Region': [reverse_region_mapping.get(region_entry.get(), region_entry.get())],
            'Number of rooms': [float(number_of_rooms_entry.get())],
            'Area': [float(area_entry.get())],
            'Living area': [float(living_area_entry.get())],
            'Kitchen area': [float(kitchen_area_entry.get())],
            'Floor': [float(floor_entry.get())],
            'Number of floors': [float(number_of_floors_entry.get())],
            'Renovation': [reverse_renovation_mapping.get(renovation_entry.get(), renovation_entry.get())]
        }
        
        
        data_dict = pd.DataFrame(dict)
        
        # scaler_y = StandardScaler()
        # scaler_x = StandardScaler()
        
        data_exp = pd.concat([data.drop(['Price'], axis = 1), data_dict],axis = 0,  ignore_index=True)
        data_trans = vvod_dannyx(data_exp)
        
        with open('pickle_model.pkl', 'rb') as file: 
            pickle_model = pickle.load(file)
        
        # scaler_x.fit_transform(data_trans)
        # scaler_y.fit_transform(np.reshape(data['Price'], (-1,1)))
        
        # mean_y, scale_y = scaler_y.mean_, scaler_y.scale_
        # print(mean_y, scale_y)
        # mean_x, var_x = scaler_x.mean_, scaler_x.var_    
        
        # Вызов вашей функции прогнозирования с полученными параметрами
        # model = torch.load('model_igor.pth')
        # model.eval()
        # torch_dat = torch.Tensor(data_trans.iloc[-1]).float()
        # print(torch_dat.shape)
        pred = pickle_model.predict(data_trans)
        # print(pred.shape)
        
        print(pred[-1], data_trans.iloc[-1], data_trans.iloc[1])
        
        # Округление числа и форматирование с разделением разрядов пробелами
        formatted_number = '{:,.0f}'.format(round(float(pred[-1]))).replace(',', ' ')
        # Вывод отформатированного числа
        result_label.config(text=f"Прогнозируемая цена: {formatted_number} ₽", bg="#f0cbbb", fg="green")
        # Вывод результата на форме или в новом окне

    # Кнопка для запуска прогнозирования
    predict_button = tk.Button(forecasting_window, text="Спрогнозировать цену", command=prognoz)
    predict_button.grid(row=12, columnspan=2)

    # Метка для отображения результата прогнозирования
    result_label = tk.Label(forecasting_window, text="", font=("Arial Black", 14), bg="#f0cbbb")
    result_label.grid(row=13, columnspan=2)


# Создание экрана загрузки
loading_screen = tk.Tk()
loading_screen.title("Загрузка...")
loading_screen.geometry("300x300")
loading_screen.configure(bg="#f0cbbb")  # Установка цвета фона

# Загрузка GIF-изображения анимации
loading_gif = Image.open("C:/Users/gorox/Desktop/EstateOracle/HdWy.gif")
loading_gif = loading_gif.resize((200, 150), Image.BICUBIC)  # Исправлено на BILINEAR
loading_gif = ImageTk.PhotoImage(loading_gif)

# Отображение анимации загрузки
loading_label = tk.Label(loading_screen, image=loading_gif)
loading_label.image = loading_gif
loading_label.pack(pady=20)

# Добавление текста о загрузке

label1 = tk.Label(loading_screen, text="Estate Oracle", font=("Arial", 24, "bold"))
label1.pack()

label2 = tk.Label(loading_screen, text="Пожалуйста, подождите, идет загрузка...", font=("Arial", 12))
label2.pack()

# Отображение экрана загрузки
loading_screen.after(1000, open_main_application)  # Закрытие экрана загрузки через 1 секунды

# Запуск главного цикла программы
loading_screen.mainloop()

