import pandas as pd
import pytest

from walmart_sales_forecast.analisis_eda import AnalisisExploratorio


@pytest.fixture
def analisis():
    return AnalisisExploratorio()


def test_detectarDuplicadosEncuentraFilasRepetidas(analisis):
    df = pd.DataFrame({
        "Store": [1, 1, 2],
        "Dept": [1, 1, 1],
        "Date": ["2024-01-01", "2024-01-01", "2024-01-08"],
    })

    duplicados = analisis.detectarDuplicados(df, columnasClave=["Store", "Dept", "Date"])

    assert len(duplicados) == 2


def test_eliminarDuplicadosReduceFilas(analisis):
    df = pd.DataFrame({"Store": [1, 1, 2]})

    resultado = analisis.eliminarDuplicados(df)

    assert len(resultado) == 2


def test_validarFormatoDetectaTipoIncorrecto(analisis):
    df = pd.DataFrame({"Store": ["1", "2"]})

    esCorrecto = analisis.validarFormato(df, {"Store": "int64"})

    assert esCorrecto is False


def test_validarFormatoConfirmaTipoCorrecto(analisis):
    df = pd.DataFrame({"Store": [1, 2]})

    esCorrecto = analisis.validarFormato(df, {"Store": "int64"})

    assert esCorrecto is True


def test_resumenNulosSoloListaColumnasConNulos(analisis):
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})

    tabla = analisis.resumenNulos(df, "prueba")

    assert list(tabla.index) == ["a"]
    assert tabla.loc["a", "nulos"] == 1


def test_detectarOutliersIQREncuentraValorExtremo(analisis):
    df = pd.DataFrame({"valor": [10, 11, 9, 10, 12, 1000]})

    outliers = analisis.detectarOutliersIQR(df, "valor")

    assert len(outliers) == 1
    assert outliers["valor"].iloc[0] == 1000


def test_graficarDistribucionVentasGuardaArchivoPng(tmp_path):
    analisisConSalida = AnalisisExploratorio(directorio_salida=str(tmp_path))
    df = pd.DataFrame({"Weekly_Sales": [100.0, 200.0, 150.0, 300.0]})

    analisisConSalida.graficarDistribucionVentas(df, guardar=True)

    assert (tmp_path / "distribucion_weekly_sales.png").exists()
