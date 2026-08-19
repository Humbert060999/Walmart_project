import sys
import os

# Añadir el directorio actual al path para evitar problemas de rutas
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importaciones protegidas (si algún archivo no existe o le falta la función main, no colapsa el menú)
try:
    from walmart_sales_forecast.eda import main as run_eda
except ImportError:
    run_eda = None

try:
    from walmart_sales_forecast.arima import main as train_arima
except ImportError:
    train_arima = None

try:
    from walmart_sales_forecast.r_lineal import main as train_lr
except ImportError:
    train_lr = None

try:
    from walmart_sales_forecast.forest import main as train_rf
except ImportError:
    train_rf = None

def mostrar_menu():
    print("\n" + "="*50)
    print("   SISTEMA DE PRONÓSTICO DE VENTAS WALMART")
    print("="*50)
    print("1. Ejecutar EDA (Análisis Exploratorio de Datos)")
    print("2. Entrenar ARIMA (Serie Temporal Global)")
    print("3. Entrenar Regresión Lineal (Multivariable)")
    print("4. Entrenar Random Forest")
    print("5. Salir")
    print("="*50)

def main():
    while True:
        mostrar_menu()
        opcion = input("Elige una opción (1-5): ")

        if opcion == '1':
            if run_eda:
                run_eda()
            else:
                print("❌ No se encontró el archivo de EDA (ej. eda.py) en walmart_sales_forecast/")
                
        elif opcion == '2':
            if train_arima:
                train_arima()
            else:
                print("❌ No se encontró train_arima.py en walmart_sales_forecast/")
                
        elif opcion == '3':
            if train_lr:
                train_lr()
            else:
                print("❌ No se encontró train_linear_regression.py en walmart_sales_forecast/")
                
        elif opcion == '4':
            if train_rf:
                train_rf()
            else:
                print("❌ No se encontró el archivo de Random Forest (ej. train_random_forest.py) en walmart_sales_forecast/")
                
        elif opcion == '5':
            print("👋 Saliendo del sistema. ¡Hasta luego!")
            break
        else:
            print("❌ Opción no válida. Por favor, elige un número entre 1 y 5.")

if __name__ == "__main__":
    main()