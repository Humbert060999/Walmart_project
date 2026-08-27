import pandas as pd
import pytest

from walmart_sales_forecast.features import Preprocesador


@pytest.fixture
def preprocesador():
    return Preprocesador()


@pytest.fixture
def dfNumerico():
    return pd.DataFrame({
        "Temperature": [10.0, 20.0, 30.0],
        "Fuel_Price": [2.0, 2.0, 2.0],
        "CPI": [200.0, 200.0, 200.0],
        "Unemployment": [7.0, 7.0, 7.0],
        "Size": [100000, 100000, 100000],
    })


def test_ajustarScalerUsaSoloLosDatosDeTrain(preprocesador, dfNumerico):
    preprocesador.ajustar_scaler(dfNumerico)

    assert preprocesador.scaler.mean_[0] == pytest.approx(20.0)


def test_transformarScalerNoReajustaConDatosNuevos(preprocesador, dfNumerico):
    validation = pd.DataFrame({
        "Temperature": [1000.0],
        "Fuel_Price": [2.0],
        "CPI": [200.0],
        "Unemployment": [7.0],
        "Size": [100000],
    })

    preprocesador.ajustar_scaler(dfNumerico)
    mediaAntes = preprocesador.scaler.mean_[0]
    preprocesador.transformar_scaler(validation)

    assert preprocesador.scaler.mean_[0] == mediaAntes


def test_tratarOutliersRecortaConLimitesDados(preprocesador):
    df = pd.DataFrame({"Weekly_Sales": [-100.0, 50.0, 100000.0]})

    resultado = preprocesador.tratar_outliers(df, columna="Weekly_Sales", limites=(0.0, 1000.0))

    assert resultado["Weekly_Sales"].tolist() == [0.0, 50.0, 1000.0]


def test_codificarCategoricasGeneraColumnasEsperadas(preprocesador):
    df = pd.DataFrame({"Type": ["A", "B"], "IsHoliday": [True, False]})

    resultado = preprocesador.codificar_categoricas(df)

    assert {"TipoTienda_A", "TipoTienda_B"}.issubset(resultado.columns)
    assert resultado["IsHoliday"].tolist() == [1, 0]


def test_crearVariablesLagPrimeraSemanaEsNula(preprocesador):
    df = pd.DataFrame({
        "Store": [1, 1],
        "Dept": [1, 1],
        "Date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
        "Weekly_Sales": [1000.0, 1200.0],
    })

    resultado = preprocesador.crear_variables_lag(df)

    assert pd.isna(resultado.iloc[0]["Ventas_Lag_1"])
    assert resultado.iloc[1]["Ventas_Lag_1"] == 1000.0


def test_dividirTrainValRespetaOrdenTemporal(preprocesador):
    fechas = pd.date_range("2024-01-01", periods=10, freq="W")
    df = pd.DataFrame({"Date": fechas, "Weekly_Sales": range(10)})

    train, validation = preprocesador.dividir_train_val(df, pct_validacion=0.2)

    assert train["Date"].max() < validation["Date"].min()
    assert len(validation) == 2
