"""Tests for data loading functionality."""

import pytest
import dask.dataframe as dd
from adv_data_processing.loading import (
    load_data,
    load_from_csv,
    load_from_excel,
    load_from_json,
    load_from_parquet,
    load_from_html,
    load_from_url,
    load_from_sql,
    load_from_s3,
    load_from_api,
    load_from_csv_chunked,
    load_from_json_chunked,
    load_from_sql_chunked
)

@pytest.fixture
def sample_csv_file(tmp_path):
    data = "A,B,C\n1,2,3\n4,5,6\n7,8,9"
    file_path = tmp_path / "sample.csv"
    file_path.write_text(data)
    return str(file_path)

@pytest.fixture
def sample_excel_file(tmp_path):
    data = {"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]}
    file_path = tmp_path / "sample.xlsx"
    pd.DataFrame(data).to_excel(file_path, index=False)
    return str(file_path)

@pytest.fixture
def sample_json_file(tmp_path):
    data = [{"A": 1, "B": 2, "C": 3}, {"A": 4, "B": 5, "C": 6}, {"A": 7, "B": 8, "C": 9}]
    file_path = tmp_path / "sample.json"
    file_path.write_text(json.dumps(data))
    return str(file_path)

@pytest.fixture
def sample_parquet_file(tmp_path):
    data = {"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]}
    file_path = tmp_path / "sample.parquet"
    pd.DataFrame(data).to_parquet(file_path)
    return str(file_path)

@pytest.fixture
def sample_html_file(tmp_path):
    data = """
    <table>
        <tr><th>A</th><th>B</th><th>C</th></tr>
        <tr><td>1</td><td>2</td><td>3</td></tr>
        <tr><td>4</td><td>5</td><td>6</td></tr>
        <tr><td>7</td><td>8</td><td>9</td></tr>
    </table>
    """
    file_path = tmp_path / "sample.html"
    file_path.write_text(data)
    return str(file_path)

def test_load_from_csv(sample_csv_file):
    df = load_from_csv(sample_csv_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_from_excel(sample_excel_file):
    df = load_from_excel(sample_excel_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_from_json(sample_json_file):
    df = load_from_json(sample_json_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_from_parquet(sample_parquet_file):
    df = load_from_parquet(sample_parquet_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_from_html(sample_html_file):
    df = load_from_html(sample_html_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_data_csv(sample_csv_file):
    df = load_data(sample_csv_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_data_excel(sample_excel_file):
    df = load_data(sample_excel_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_data_json(sample_json_file):
    df = load_data(sample_json_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_data_parquet(sample_parquet_file):
    df = load_data(sample_parquet_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_load_data_html(sample_html_file):
    df = load_data(sample_html_file)
    assert isinstance(df, dd.DataFrame)
    assert df.shape[0].compute() == 3

def test_edge_case_empty_file(tmp_path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("")
    with pytest.raises(ValueError):
        load_data(str(empty_file))

def test_edge_case_invalid_format(tmp_path):
    invalid_file = tmp_path / "invalid.txt"
    invalid_file.write_text("Invalid content")
    with pytest.raises(ValueError):
        load_data(str(invalid_file))
