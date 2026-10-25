# CropIntel

**Satellite-based multi-farm monitoring and automated field reporting**  
An AfriScan geospatial software project.

This repository contains **selected original Python source files**, an Apache Airflow DAG snapshot, and actual reports generated for the *CropIntel Sene East Demo Farm*, a 5.0 ha maize demonstration field in Bono East, Ghana.

The reports are examples of the software output, not field-validated diagnoses. Farm data are described as demonstration data. No customer records or authentication credentials are included.

![Field report](docs/sample-reports/overview.png)

## Example farm outputs

| Output | View |
| --- | --- |
| Full field report | [View image](docs/sample-reports/overview.png) |
| Crop condition | [View image](docs/sample-reports/crop-condition.png) |
| Field trends | [View image](docs/sample-reports/field-trends.png) |
| Irrigation | [View image](docs/sample-reports/irrigation.png) |
| Nutrition indicators | [View image](docs/sample-reports/nutrition.png) |
| Water and climate | [View image](docs/sample-reports/water-and-climate.png) |

The example reports were produced on **25 October 2026**. The latest optical image listed in the trend report was **19 October 2026**. A report date is not necessarily the satellite observation date.

## Selected Python files

These are selected files from the real CropIntel codebase, not invented demonstration scripts.

| File | Purpose |
| --- | --- |
| `src/extract_data.py` | Satellite indicator extraction and processing |
| `src/gap_fill_ndvi.py` | Handling missing observations in NDVI time series |
| `src/gee_helpers.py` | Google Earth Engine initialization and helper functions |
| `src/farm_loader.py` | Reading and validating farm boundaries |
| `src/transform_data.py` | Data processing and indicator transformation |
| `src/validate_outputs.py` | Checking generated outputs |
| `src/dashboard_maps.py` | Mapping routines used in dashboard reports |
| `dags/cropintel_monitoring_dag.py` | Airflow pipeline orchestration snapshot |

**Scope:** This public selection is **not a stand-alone runnable copy** of the platform. Other private modules, configuration, datasets and the deployment environment are required. The scripts are retained from the project snapshot and have not been rewritten as mock implementations.

## Pipeline and cloud deployment

CropIntel was deployed on an **Oracle Cloud Infrastructure Ubuntu ARM64 virtual machine** using **Docker Compose, Apache Airflow and PostgreSQL**. Tasks validate farm inputs, retrieve satellite and weather information, transform indicators, validate results and create field reports.

![Successful Airflow graph](docs/airflow/successful-dag-run.png)

![Airflow schedule](docs/airflow/schedule-dashboard.png)

The screenshots show a **successful run on 25 October 2026**, completing in approximately **27 minutes**. The Airflow schedule dashboard displayed a **14-day interval**.

**Version note:** The included DAG is an earlier source snapshot that describes a weekly schedule. The deployment screenshots show a later biweekly configuration. Do not use the snapshot alone to infer the current production schedule.

## Interpretation and limitations

CropIntel uses optical and radar observations along with weather information to help prioritize field inspection. NDVI reflects canopy greenness, NDMI is an optical moisture-related index, and NDRE is a red-edge vegetation index. None directly diagnoses nutrient deficiency, disease or required irrigation.

Satellite observation gaps, processing assumptions, geolocation errors and crop growth stage affect interpretation. The sample outputs are not presented as measured field-validation accuracy.

## Privacy and commercialization

This public project includes only selected source files and demonstration output images. Full commercial decision rules, private configuration, access credentials and customer datasets are intentionally excluded. Usage of satellite and cloud services in commercial settings is subject to their terms and applicable licensing.

## Author

Philip Atta Afriyie — CropIntel / AfriScan  
Website — https://afriscangeo.com
