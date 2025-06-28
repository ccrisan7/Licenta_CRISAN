import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import serial
import time
import threading
from sklearn.metrics import r2_score
from matplotlib import rcParams
from scipy.interpolate import UnivariateSpline
from fpdf import FPDF
from datetime import datetime, timedelta
import os

# Setari initiale
ser = serial.Serial('COM3', 9600)  # Portul serial
max_time = 90  # Maxim 90 de secunde per etapa calibrare
start_time = 0
current_stage = 0  # 0 = nu s-a inceput, 1 = calibrare 1, 2 = calibrare 2, 3 = calibrare 3, 4 = titrare
timer_running = False
medie = 0  # Valoarea medie
i = 0  # Contor pentru numarul de citiri
medii_finale = []  # Lista pentru medii finale
pHfinal = []
volumfinal = []
timer_label = None
running = False  # Controlam daca timerul ruleaza sau nu
pHuri = [6.86, 4.01, 9.21]
canvas = None 
string1_val = "Substanta necunoscuta"
string2_val = "Substanta necunoscuta"
value_val = 30
pKavalue = None

global m, b, pkafinal

def gaseste_dreapta_si_plot_tk(medii_finale, frame):
    global canvas
    if len(medii_finale) < 3:
        print("Vectorul medii_finale trebuie sa contina cel putin 3 valori pentru a calcula dreapta.")
        return

    tensiuni = np.array(medii_finale[:3])
    pH_valori = np.array([6.86, 4.01, 9.21])

    # Calculul coeficientilor dreptei de regresie liniara
    A = np.vstack([tensiuni, np.ones(len(tensiuni))]).T
    m, b = np.linalg.lstsq(A, pH_valori, rcond=None)[0]

    if np.isclose(m, 0):
        print("Panta dreptei este aproape de zero, nu se poate calcula tensiunea in functie de pH.")
        return

    # Valorile prezise pe baza tensiunilor
    pH_pred = m * tensiuni + b
    r2 = r2_score(pH_valori, pH_pred)

    # Generam date pentru plot
    pH_interval = np.linspace(0, 14, 100)
    tensiuni_dreapta = (pH_interval - b) / m

    # Cream figura pentru Tkinter
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(tensiuni, pH_valori, color='blue', label='Puncte de calibrare')
    ax.plot(tensiuni_dreapta, pH_interval, color='red', linestyle='-',
            label=f'Dreapta de etalonare: pH = {m:.4f}T + {b:.4f} \n(R² = {r2:.4f})')
    ax.set_xlabel("Tensiunea electrica (V)")
    ax.set_ylabel("pH")
    ax.set_title("Dreapta de etalonare a pH-metrului")
    ax.set_xlim(min(tensiuni_dreapta) - 0.5, max(tensiuni_dreapta) + 0.5)
    ax.set_ylim(0, 14)
    ax.legend()
    ax.grid(True)

    # Afisam plot-ul in interfata Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    canvas.print_figure('calibrare.png', dpi=300)

    return m, b


def find_single_inflection(ph_values, volume_values, frame, smoothing_factor=20):
    global pKavalue
    for widget in frame.winfo_children():
        widget.destroy()

    vol = np.array(volume_values)
    pH = np.array(ph_values)

    # Spline cu factor de netezire
    spline = UnivariateSpline(vol, pH, s=smoothing_factor)

    # Puncte dense pentru evaluare
    x_dense = np.linspace(min(vol), max(vol), 1000)
    y_dense = spline(x_dense)
    dy = spline.derivative(n=1)(x_dense)
    d2y = spline.derivative(n=2)(x_dense)

    # Gasim unde derivata a doua isi schimba semnul
    sign_change_indices = np.where(np.diff(np.sign(d2y)))[0]

    if len(sign_change_indices) == 0:
        print("Nu s-a gasit niciun punct de inflexiune.")
        return

    # Alegem cea mai semnificativa inflexiune
    best_index = sign_change_indices[np.argmax(np.abs(dy[sign_change_indices]))]
    inflection_x = x_dense[best_index]
    inflection_y = y_dense[best_index]

    # Plot
    fig = plt.Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(vol, pH, 'o', label='Date experimentale', markersize=4)
    ax.plot(x_dense, y_dense, '-', label='Curba caracteristica')
    ax.axvline(inflection_x, color='red', linestyle='--',
               label=f'Inflectiune ≈ {inflection_x:.2f} mL, pH = {inflection_y:.2f}')
    ax.scatter(inflection_x, inflection_y, color='red', zorder=5)
    ax.set_title('Curba de titrare')
    ax.set_xlabel('Volum titrant (mL)')
    ax.set_ylabel('pH')
    ax.legend()
    ax.grid(True)

    # Integrare in interfata Tkinter
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Salvare imagine
    fig.savefig('titrare.png', dpi=300)
    pKavalue = spline(inflection_x / 2)

    # Returnam coordonatele punctului de inflexiune
    return pKavalue

# Functie pentru a trimite comenzi catre Arduino
def send_command(command):
    ser.write((command + "\n").encode())

# Functie care citeste valorile de la Arduino
def read_from_arduino():
    global medie, i, timer_running
    while timer_running:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            try:
                valoare_analog = int(data)  # Incercam sa convertim direct in int
                medie += valoare_analog
                i += 1
                print(f"Citire Arduino: {valoare_analog}, Tensiune electica masurata: {(valoare_analog * 5 / 1023):.2f} V")
                if i >= 10:  # Daca am facut 10 citiri
                    medie /= 10
                    medii_finale.append(medie * 5 / 1023)  # Salvam media in lista de medii finale
                    update_result()  # Actualizam rezultatul
                    break
            except ValueError:
                # Daca nu este un numar valid, ignoram linia
                continue
        time.sleep(1)
    next_button.pack(pady=10)  # il readaugi in interfata

# Functie pentru a incepe cronometru si citirea de la Arduino
def start_stage(stage):
    global start_time, current_stage, timer_running, medie, i, start_button, next_button
    start_button.pack_forget()
    next_button.pack_forget()
    start_time = time.time()  # Timpul curent pentru cronometru
    medie = 0  # Resetam valoarea medie
    i = 0  # Resetam contorul citirilor
    current_stage = stage
    timer_running = True
    result_label.config(text=f"Calibrare {stage} (pH = {pHuri[stage - 1]})\nCitire in curs...")
    send_command(f"Calibrare{stage}")  # Trimitem comanda corespunzatoare etapei
    # incepem sa citim datele de la Arduino intr-un thread separat
    threading.Thread(target=read_from_arduino, daemon=True).start()

fig, ax = plt.subplots(figsize=(8, 6))

def plot_ph_vs_volume(pHfinal, volumfinal, frame):
    # Optional: seteaza fonturi safe
    rcParams.update({'font.family': 'Arial'}) 

    # Goleste frame-ul inainte sa redesenezi
    for widget in frame.winfo_children():
        widget.destroy()

    # Creeaza graficul
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(volumfinal, pHfinal, marker='o', color='blue', linestyle='-')

    ax.set_title("Curba de titrare", fontsize=12)
    ax.set_xlabel("Volum titrant (mL)", fontsize=10)
    ax.set_ylabel("pH", fontsize=10)
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack()
    # inchide figura pentru a elibera memoria
    plt.close(fig)


def read_titrare_from_arduino(frame, m, b):
    global timer_running 
    timer_running = True
    i = 0
    print(f"Inceput titrare:{timer_running}")
    while timer_running:
        if ser.in_waiting > 0:
            data = ser.readline().decode().strip()
            try:
                valoare_analog = int(data)
                tensiune = valoare_analog * 5 / 1023
                ph = tensiune * m + b
                print(f"Citire Arduino: {valoare_analog}, Tensiune electrica: {tensiune:.2f} V -> pH = {ph}")
                pHfinal.append(ph)
                volumfinal.append(i / 4)  
                frame.after(0, lambda: plot_ph_vs_volume(pHfinal, volumfinal, frame))
                i += 1
                if i >= 120:
                    next_button.config(state=tk.NORMAL)
                    timer_running = False
                    break
            except ValueError:
                continue
        time.sleep(1)
    next_button.pack(pady=10)  # il readaugi in interfata

def update_timer(start_time, frame, volum=0):
    global running
    while running:
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if seconds % 30 == 0:
            volum = int(elapsed) / 120 + 0.25
        if minutes >= 60:
            timer_label.config(text="Titrarea este finalizata.")
            running = False
            break
        timer_label.config(text=f"Cronometru: {minutes:02d}:{seconds:02d}\nProcesul de titrare automatizata dureaza 60 de minute.\nVolum de titrant: {volum:.2f} mL")
        time.sleep(1)

def start_titrare(frame, m, b):
    global running, timer_label

    result_label.config(text="Titrare in curs...")
    send_command(f"Titrare")  # Trimitem comanda corespunzatoare etapei

    # Afisam cronometru pe UI
    if timer_label is None:
        timer_label = tk.Label(frame, text="00:00", font=("Arial", 16))
        timer_label.pack(pady=10)

    running = True
    start_time = time.time()    

    print("Inceput thread 1")

    # Thread pentru cronometru
    threading.Thread(target=update_timer, args=(start_time, frame), daemon=True).start()

    print("Inceput thread 2")

    # Thread pentru citire date Arduino
    threading.Thread(target=read_titrare_from_arduino, args=(frame, m, b), daemon=True).start()

# Functie pentru a opri cronometru si a opri citirea
def stop_timer():
    global timer_running
    timer_running = False
    elapsed_time = int(time.time() - start_time)
    if elapsed_time >= max_time:
        result_label.config(text="Timpul a expirat. Poti trece la urmatoarea etapa.")
    else:
        result_label.config(text="Etapa finalizata cu succes!")
    next_button.config(state=tk.NORMAL)  # Permitem trecerea la urmatoarea etapa

def generatePdf():
    global medii_finale, pHfinal, volumfinal, string1_val, string2_val, value_val, m, b, pHuri

    # Data si ora
    begin = datetime.now() - timedelta(minutes=60)  # scade 60 de minute
    timestamp = begin.strftime("%Y-%m-%d_%H-%M-%S")
    begin_display = begin.strftime("%d.%m.%Y, %H:%M:%S")

    filename = f"Titrare_{timestamp}.pdf"

    # PDF setup
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Titlu
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Titrare automatizata - Date experimentale si rezultate", ln=True, align='C')
    pdf.ln(10)

    # Informatii generale
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f"Data si ora inceperii experimentului: {begin_display}", ln=True)
    pdf.cell(0, 10, f"Substanta titranta: {string1_val}", ln=True)
    pdf.cell(0, 10, f"Substanta titrata: {string2_val}", ln=True)
    pdf.cell(0, 10, f"Volum titrant utilizat: {value_val} mL", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Valori medii de etalonare (V):", ln=True)
    pdf.set_font("Arial", '', 12)
    for idx, val in enumerate(medii_finale[:3]):
        pdf.cell(0, 10, f"Etalon {idx+1} (pH = {pHuri[idx]}): {val:.4f} V", ln=True)

    pdf.ln(10)
    if os.path.exists("calibrare.png"):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Grafic de etalonare:", ln=True)
        pdf.image("calibrare.png", x=30, w=150)
        pdf.ln(10)

    # Tabel titrare
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Valori in timpul titrarii:", ln=True)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(60, 8, "Volum (mL)", 1)
    pdf.cell(60, 8, "Tensiune (V)", 1)
    pdf.cell(60, 8, "pH", 1, ln=True)
    pdf.set_font("Arial", '', 10)

    for i in range(len(pHfinal)):
        tensiune = (pHfinal[i] - b) / m
        pdf.cell(60, 8, f"{volumfinal[i]:.2f}", 1)
        pdf.cell(60, 8, f"{tensiune:.4f}", 1)
        pdf.cell(60, 8, f"{pHfinal[i]:.2f}", 1, ln=True)

    pdf.ln(10)

    pdf.add_page()
    if os.path.exists("titrare.png"):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Curba de titrare:", ln=True)
        pdf.image("titrare.png", x=20, w=170)

    pdf.set_font("Arial", '', 14)
    pdf.cell(0, 10, f"Valoarea pKa calculata pentru {string2_val} este: {pKavalue:.2f}.", ln=True)

    pdf.output(filename)
    print(f"PDF salvat ca: {filename}")

# Functie pentru a actualiza rezultatul in interfata
def update_result():
    global medie
    if current_stage == 1:
        result_label.config(text=f"Tensiunea electica medie la Calibrarea 1 (pH 6.86): {(medie * 5 / 1023):.2f} V")
    elif current_stage == 2:
        result_label.config(text=f"Tensiunea electica medie la Calibrarea 2 (pH 4.01): {(medie * 5 / 1023):.2f} V")
    elif current_stage == 3:
        result_label.config(text=f"Tensiunea electica medie la Calibrarea 3 (pH 9.21): {(medie * 5 / 1023):.2f} V")
    elif current_stage == 4:
        result_label.config(text=f"Tensiunea electica medie la Calibrarea 3 (pH 9.21): {(medie * 5 / 1023):.2f} V")
    elif current_stage == 5:
        result_label.config(text=f"Tensiunea electica medie la Calibrarea 3 (pH 9.21): {(medie * 5 / 1023):.2f} V")
    stop_timer()

# Functie pentru a trece la etapa urmatoare
def next_stage():
    global current_stage
    if current_stage == 1:
        start_button.pack_forget()
        next_button.pack_forget()  # ascunde complet butonul
        start_stage(2)
    elif current_stage == 2:
        next_button.pack_forget()  # ascunde complet butonul
        start_stage(3)
    elif current_stage == 3:
        next_button.pack_forget()  # ascunde complet butonul
        global m, b
        # Dupa finalizarea ultimei etape, afisam rezultatele finale
        m, b = gaseste_dreapta_si_plot_tk(medii_finale, plot_frame)
        result_label.config(text=f"Calibrarea a fost finalizata!\nMedii finale:\nCalibrare 1 (pH = 6.87): {medii_finale[0]:.2f} V\nCalibrare 2 (pH = 4.01): {medii_finale[1]:.2f} V\nCalibrare 3 (pH = 9.21): {medii_finale[2]:.2f} V\nEcuatia dreptei de etalonare: pH = {m:.4f} * Tensiune + {b:.4f}")
        print("Final calibrare")
        print(m, b)
        current_stage = 4
        next_button.pack(pady=10)  # il readaugi in interfata
    elif current_stage == 4:
        next_button.pack_forget()  # ascunde complet butonul
        if canvas is not None:
            canvas.get_tk_widget().destroy()  # sterge widget-ul asociat cu canvas-ul
        print("Inceput titrare")
        next_button.pack_forget()
        start_titrare(plot_frame, m, b)
        current_stage = 5
    elif current_stage == 5:
        next_button.pack_forget()
        pkafinal = find_single_inflection(pHfinal, volumfinal, plot_frame)
        result_label.config(text=f"Titrarea a fost finalizata!\nValoarea pKa calculata: {pkafinal:.2f}\nDocumentul PDF cu datele \nexperimentale a fost generat.")
        generatePdf()

# Functie pentru a incepe procesul, ascunzand campurile de input
def start_process():
    global start_button, string1_entry, string2_entry, start_button, string1_label, string2_label, string1_val, string2_val

    # Preluam valorile introduse
    string1_val = string1_entry.get()
    string2_val = string2_entry.get()

    # Afisam un mesaj de confirmare
    result_label.config(text=f"Proces de analizat cu substantele {string1_val} si {string2_val}. Folosim {value_val} mL de {string1_val} la titrare.")

    # Ascundem campurile de input
    string1_label.pack_forget()
    string2_label.pack_forget()
    string1_entry.pack_forget()
    string2_entry.pack_forget()
    start_button.config(state=tk.DISABLED)

    # Continuam cu procesul de calibrare
    start_stage(1)

# Cream interfata grafica cu Tkinter
root = tk.Tk()
root.iconbitmap(r"C:\Users\VIVOBOOK\Desktop\Licenta\Automatizare\sketch_mar31a\chatgpt_image_apr_19__2025__09_17_15_pm_IBB_icon.ico")
root.geometry("1920x1080")  # Seteaza dimensiunea ferestrei la 1920x1080 pixeli

# Permitem redimensionarea ferestrei (poti modifica cu False daca nu vrei sa o redimensionezi)
root.resizable(True, True)
root.title("Determinarea constantei pKa prin titrare automatizata")

# Eticheta pentru a afisa rezultatul
result_label = tk.Label(root, text="La apasarea butonului Start proces\n se va incepe calibrarea 1 (pH = 6.86). ", font=("Arial", 16), width=40, height=4)
result_label.pack(pady=20)

# Campuri pentru introducerea celor doua siruri si valorii finale
string1_label = tk.Label(root, text="Substanta titranta:")
string1_label.pack(pady=5)
string1_entry = tk.Entry(root, font=("Arial", 14))
string1_entry.pack(pady=5)

string2_label = tk.Label(root, text="Substanta titrata:")
string2_label.pack(pady=5)
string2_entry = tk.Entry(root, font=("Arial", 14))
string2_entry.pack(pady=5)

# Butonul pentru a incepe procesul
start_button = tk.Button(root, text="Start proces", font=("Arial", 14), command=start_process, width=20)
start_button.pack(pady=10)

# Butonul pentru a trece la urmatoarea etapa
next_button = tk.Button(root, text="Urmatoarea Etapa", font=("Arial", 14), command=next_stage, width=20, state=tk.DISABLED)
next_button.pack(pady=10)

# Cronometru pentru afisare
timer_label = tk.Label(root, text="Durata masurare: 90 secunde", font=("Arial", 14))
timer_label.pack(pady=20)

# Frame pentru afisarea plot-ului
plot_frame = tk.Frame(root)
plot_frame.pack(pady=20)

# Pornim aplicatia
root.mainloop()
