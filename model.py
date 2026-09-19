import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg') 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import glob
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import os
import threading
import warnings
import sys

# --- KONFIGURASI ---
STATIONS = ['cken', 'ckup', 'cgrt', 'cbjm', 'cbik', 'cbkt']
warnings.simplefilter(action='ignore', category=FutureWarning)
color_pal = sns.color_palette()

# BARU: Menentukan path absolut dari direktori tempat skrip ini berada
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# --- BAGIAN 1: FUNGSI LOGIKA ---

def load_all_data(base_path='.'):
    # BARU: Menggunakan SCRIPT_DIR sebagai dasar untuk semua path
    full_station_path = os.path.join(SCRIPT_DIR, base_path)
    search_path = os.path.join(full_station_path, '**/final_data_*.csv')
    all_files = glob.glob(search_path, recursive=True)
    
    if not all_files: return None
    df_list = [pd.read_csv(f, index_col='datetime', parse_dates=True) for f in all_files]
    if not df_list: return None
    full_df = pd.concat(df_list).sort_index()
    return full_df

# ... (fungsi create_features dan add_seasonal_lags tidak berubah) ...
def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['year'] = df.index.year
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def add_seasonal_lags(df, target_map):
    df = df.copy()
    df['lag_1_year'] = (df.index - pd.DateOffset(years=1)).map(target_map)
    df['lag_2_year'] = (df.index - pd.DateOffset(years=2)).map(target_map)
    return df
    
def run_forecast_logic(station_name, month_choice, status_callback):
    try:
        status_callback(f"Mencari data untuk stasiun: {station_name}...")
        df_full = load_all_data(base_path=station_name)
        if df_full is None: return {"error": f"Tidak ada file 'final_data_*.csv' ditemukan di folder '{station_name}'."}

        status_callback("Membuat fitur time series...")
        df_full = create_features(df_full)
        target_map_full = df_full['Average_Value'].to_dict()
        df_full = add_seasonal_lags(df_full, target_map_full)
        
        df_month = df_full[df_full.index.month == month_choice].copy()
        if df_month.empty: return {"error": f"Error: Tidak ada data untuk bulan {month_choice} di stasiun {station_name}."}

        latest_year = df_month.index.year.max()
        df_target_year = df_month[df_month.index.year == latest_year].copy()
        df_target_year.dropna(subset=['lag_1_year', 'lag_2_year', 'average_fluxadjflux', 'Dst'], inplace=True)

        if len(df_target_year) < 10: return {"error": f"Error: Tidak cukup data untuk bulan {month_choice} pada tahun {latest_year}."}

        split_point = int(len(df_target_year) * 0.7)
        train = df_target_year.iloc[:split_point].copy()
        test = df_target_year.iloc[split_point:].copy()

        FEATURES = ['hour', 'dayofweek', 'dayofmonth', 'weekofyear', 'year', 'average_fluxadjflux', 'Dst',
                    'lag_1_year', 'lag_2_year', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
        TARGET = 'Average_Value'
        
        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        status_callback(f"Melatih model XGBoost untuk stasiun {station_name}, bulan {month_choice}...")
        reg = xgb.XGBRegressor(n_estimators=1000,
                           learning_rate=0.01,
                           max_depth=2,
                           subsample=0.8,
                           colsample_bytree=0.8,    # 3. Gunakan 80% fitur untuk setiap pohon
                           reg_lambda=1,
                           early_stopping_rounds=50,
                           objective='reg:squarederror')
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        
        test['prediction'] = reg.predict(X_test)
        score = np.sqrt(mean_squared_error(test['Average_Value'], test['prediction']))
        
        status_callback("Membuat prediksi masa depan...")
        future_year = latest_year + 1
        future_start_date = f"{future_year}-{month_choice:02d}-01"
        future_end_date = f"{future_year}-{month_choice:02d}-07"
        future = pd.date_range(future_start_date, future_end_date, freq="1h")
        future_df = pd.DataFrame(index=future)
        future_df["isFuture"] = True
        df_and_future = pd.concat([df_full, future_df])
        df_and_future['isFuture'] = df_and_future['isFuture'].fillna(False)
        df_and_future = create_features(df_and_future)
        df_and_future = add_seasonal_lags(df_and_future, target_map_full)
        future_w_features = df_and_future.query('isFuture').copy()
        future_w_features['average_fluxadjflux'] = future_w_features['average_fluxadjflux'].fillna(df_full['average_fluxadjflux'].iloc[-1])
        future_w_features['Dst'] = future_w_features['Dst'].fillna(df_full['Dst'].iloc[-1])
        future_w_features['prediction'] = reg.predict(future_w_features[FEATURES])
        
        status_callback("Selesai. Mempersiapkan hasil...")
        return {
            "rmse": score, "test_df": test, "future_df": future_w_features,
            "station": station_name, "month": month_choice, "year": latest_year
        }
    except Exception as e:
        return {"error": f"Terjadi Error: {str(e)}"}


# --- BAGIAN 2: KELAS APLIKASI GUI ---

class ForecastApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prediksi TEC Multi-Stasiun")
        self.geometry("1100x850")

        main_canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)
        self.scrollable_frame = ttk.Frame(main_canvas)
        main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))

        control_frame = ttk.Frame(self.scrollable_frame, padding="10")
        control_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(control_frame, text="Pilih Stasiun:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.station_var = tk.StringVar(value=STATIONS[0])
        station_menu = ttk.OptionMenu(control_frame, self.station_var, STATIONS[0], *STATIONS)
        station_menu.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Pilih Bulan:", font=("Arial", 12)).pack(side=tk.LEFT, padx=(20, 5))
        self.month_var = tk.StringVar(value='12')
        months = [str(i) for i in range(1, 13)]
        month_menu = ttk.OptionMenu(control_frame, self.month_var, months[11], *months)
        month_menu.pack(side=tk.LEFT, padx=5)

        self.run_button = ttk.Button(control_frame, text="Jalankan Prediksi", command=self.start_forecast_thread)
        self.run_button.pack(side=tk.LEFT, padx=10)

        self.rmse_label = ttk.Label(control_frame, text="RMSE: -", font=("Arial", 12, "bold"))
        self.rmse_label.pack(side=tk.LEFT, padx=20)
        
        self.status_label = ttk.Label(self.scrollable_frame, text="Selamat datang! Pilih stasiun & bulan, lalu klik jalankan.", padding="5", anchor="w")
        self.status_label.pack(fill=tk.X, padx=10)

        self.fig1 = Figure(figsize=(10, 4), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.scrollable_frame)
        self.canvas1.get_tk_widget().pack(pady=10)

        self.fig2 = Figure(figsize=(10, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.scrollable_frame)
        self.canvas2.get_tk_widget().pack(pady=10)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.quit()
        self.destroy()

    def start_forecast_thread(self):
        self.run_button.config(state="disabled")
        self.status_label.config(text="Memulai proses, harap tunggu...")
        station = self.station_var.get()
        month = int(self.month_var.get())
        thread = threading.Thread(target=self.run_forecast, args=(station, month))
        thread.daemon = True
        thread.start()

    def run_forecast(self, station, month):
        results = run_forecast_logic(station, month, self.update_status)
        self.after(0, self.update_gui_with_results, results)

    def update_gui_with_results(self, results):
        if "error" in results:
            messagebox.showerror("Error", results["error"])
            self.status_label.config(text="Proses gagal. Silakan coba lagi.")
        else:
            # Plot 1: Evaluasi
            self.ax1.clear() 
            test_df = results['test_df']
            test_df['Average_Value'].plot(ax=self.ax1, label='Nilai Aktual (Test)', style='-')
            test_df['prediction'].plot(ax=self.ax1, label='Prediksi XGBoost', style='--')
            self.ax1.set_title(f"Prediksi vs Aktual TEC ({results['station']} - Bulan {results['month']} - {results['year']})")
            self.ax1.legend()
            self.ax1.grid(True)
            self.canvas1.draw() 

            # Plot 2: Prediksi Masa Depan
            self.ax2.clear()
            future_df = results['future_df']
            future_start = future_df.index.min().strftime('%Y-%m-%d')
            future_end = future_df.index.max().strftime('%Y-%m-%d')
            future_df['prediction'].plot(ax=self.ax2, color=color_pal[4])
            self.ax2.set_title(f"Prediksi TEC ({results['station']}) - {future_start} s/d {future_end}")
            self.ax2.grid(True)
            self.canvas2.draw()
            
            self.rmse_label.config(text=f"RMSE ({results['station']}): {results['rmse']:.2f}")
            self.status_label.config(text="Selesai. Hasil ditampilkan.")

        self.run_button.config(state="normal")

    def update_status(self, message):
        self.after(0, self.status_label.config, {'text': message})


if __name__ == "__main__":
    app = ForecastApp()
    app.mainloop()
