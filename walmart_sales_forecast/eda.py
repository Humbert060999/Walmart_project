
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AnalisisExploratorio:
    

    # 1) DUPLICADOS Y VALIDACIÓN DE FORMATO

    def detectarDuplicados(self, df: pd.DataFrame, columnasClave: list = None) -> pd.DataFrame:
        """
        Busca filas duplicadas. Si le pasas columnasClave (ej.
        ["Store", "Dept", "Date"]), busca duplicados según SOLO esas
        columnas; si no, compara la fila completa.
        Devuelve el subconjunto de filas duplicadas encontradas.
        """
        if columnasClave:
            mascara = df.duplicated(subset=columnasClave, keep=False)
        else:
            mascara = df.duplicated(keep=False)

        duplicados = df[mascara]
        print(f"Filas duplicadas encontradas: {len(duplicados)} de {len(df)} "
              f"({len(duplicados) / len(df) * 100:.2f}%)")
        return duplicados

    def eliminarDuplicados(self, df: pd.DataFrame, columnasClave: list = None) -> pd.DataFrame:
        """
        Elimina duplicados y devuelve un DataFrame limpio (se
        conserva la primera aparición de cada fila repetida).
        """
        filasAntes = len(df)
        if columnasClave:
            dfLimpio = df.drop_duplicates(subset=columnasClave, keep="first")
        else:
            dfLimpio = df.drop_duplicates(keep="first")

        filasEliminadas = filasAntes - len(dfLimpio)
        print(f"Se eliminaron {filasEliminadas} filas duplicadas "
              f"({filasAntes} -> {len(dfLimpio)})")
        return dfLimpio

    def validarFormato(self, df: pd.DataFrame, tiposEsperados: dict) -> bool:
       
        todoCorrecto = True
        for columna, tipoEsperado in tiposEsperados.items():
            if columna not in df.columns:
                print(f"⚠ Columna '{columna}' no existe en el DataFrame")
                todoCorrecto = False
                continue

            tipoReal = str(df[columna].dtype)
            if tipoReal != tipoEsperado:
                print(f"⚠ '{columna}': se esperaba {tipoEsperado}, tiene {tipoReal}")
                todoCorrecto = False

        if todoCorrecto:
            print("✓ Todos los formatos son correctos")
        return todoCorrecto

    # 2) RESÚMENES NUMÉRICOS

    def resumenGeneral(self, df: pd.DataFrame, nombre: str) -> None:
        """Imprime forma, tipos de datos y rango de fechas (si aplica)."""
        print(f"\n{'=' * 60}\nRESUMEN: {nombre}\n{'=' * 60}")
        print(f"Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")
        print(f"\nTipos de datos:\n{df.dtypes}")
        if "Date" in df.columns:
            print(f"\nRango de fechas: {df['Date'].min()} a {df['Date'].max()}")

    def resumenNulos(self, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
        """Tabla de nulos por columna, ordenada de mayor a menor porcentaje."""
        nulos = df.isnull().sum()
        porcentaje = (nulos / len(df) * 100).round(2)
        tabla = pd.DataFrame({"nulos": nulos, "porcentaje_%": porcentaje})
        tabla = tabla[tabla["nulos"] > 0].sort_values("porcentaje_%", ascending=False)

        print(f"\nValores nulos en {nombre}:")
        print(tabla if not tabla.empty else "  Sin valores nulos.")
        return tabla

    def resumenEstadistico(self, df: pd.DataFrame, nombre: str) -> pd.DataFrame:
        """Estadísticas descriptivas (media, min, max, cuartiles, etc.)."""
        print(f"\nEstadísticas descriptivas de {nombre}:")
        descripcion = df.describe(include="all").T
        print(descripcion)
        return descripcion

    def detectarOutliersIQR(self, df: pd.DataFrame, columna: str) -> pd.DataFrame:
        """Detecta outliers con el método de rango intercuartílico (IQR)."""
        q1, q3 = df[columna].quantile([0.25, 0.75])
        iqr = q3 - q1
        limiteInferior = q1 - 1.5 * iqr
        limiteSuperior = q3 + 1.5 * iqr

        outliers = df[(df[columna] < limiteInferior) | (df[columna] > limiteSuperior)]
        print(f"Outliers detectados en {columna}: {len(outliers)} de {len(df)} filas "
              f"({len(outliers) / len(df) * 100:.2f}%)")
        return outliers

    def ejecutarEdaBasico(self, df: pd.DataFrame, nombre: str) -> None:
        """Atajo: corre resumenGeneral + resumenNulos + resumenEstadistico juntos."""
        self.resumenGeneral(df, nombre)
        self.resumenNulos(df, nombre)
        self.resumenEstadistico(df, nombre)

    # 3) GRÁFICOS

    def graficarDistribucionVentas(self, df: pd.DataFrame) -> None:
        """Histograma de Weekly_Sales — para ver el sesgo/outliers."""
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Weekly_Sales"].dropna(), kde=True, bins=60)
        plt.title("Distribución de Weekly_Sales")
        plt.xlabel("Weekly_Sales")
        plt.tight_layout()
        plt.show()

    def graficarVentasPorFeriado(self, df: pd.DataFrame) -> None:
        """Boxplot: Weekly_Sales según IsHoliday (True/False)."""
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x="IsHoliday", y="Weekly_Sales")
        plt.title("Ventas semanales: Feriado vs. No feriado")
        plt.xlabel("¿Es semana feriado?")
        plt.ylabel("Weekly_Sales")
        plt.tight_layout()
        plt.show()

    def graficarVentasPorTipoTienda(self, df: pd.DataFrame) -> None:
        """Boxplot: Weekly_Sales según Type (A/B/C)."""
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x="Type", y="Weekly_Sales", order=["A", "B", "C"])
        plt.title("Ventas semanales por tipo de tienda")
        plt.xlabel("Tipo de tienda")
        plt.ylabel("Weekly_Sales")
        plt.tight_layout()
        plt.show()

    def graficarEstacionalidad(self, df: pd.DataFrame) -> None:
        """Línea de tiempo: ventas totales agrupadas por semana."""
        ventasPorSemana = df.groupby("Date")["Weekly_Sales"].sum().sort_index()
        plt.figure(figsize=(14, 5))
        ventasPorSemana.plot()
        plt.title("Ventas totales por semana (estacionalidad)")
        plt.xlabel("Fecha")
        plt.ylabel("Ventas totales")
        plt.tight_layout()
        plt.show()

    def graficarCorrelacion(self, df: pd.DataFrame) -> None:
        """Heatmap: correlación entre variables numéricas y Weekly_Sales."""
        columnasNumericas = df.select_dtypes(include="number").columns
        matrizCorrelacion = df[columnasNumericas].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(matrizCorrelacion, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Matriz de correlación (dataset combinado)")
        plt.tight_layout()
        plt.show()
