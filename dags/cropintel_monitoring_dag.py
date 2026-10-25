"""Weekly CropIntel pipeline: validate, extract, transform, verify and report."""

from datetime import timedelta
import os
from pathlib import Path
import subprocess
import sys

import pendulum
from airflow.sdk import dag, task


HOME = Path(
    os.getenv("CROPINTEL_HOME", "/opt/airflow/cropintel")
)

SRC = HOME / "src"


def run_script(script_name, *args):
    """Run one CropIntel Python script with immediate log output."""

    script_path = SRC / script_name

    print(f"Starting {script_name}", flush=True)

    subprocess.run(
        [sys.executable, "-u", str(script_path), *args],
        check=True,
        env=os.environ.copy(),
        timeout=60 * 60 * 2,
    )

    print(f"Finished {script_name}", flush=True)


@dag(
    dag_id="cropintel_weekly_monitoring",
    description="Weekly CropIntel farm-monitoring pipeline",
    # Run every Monday at 06:00 UTC. Each run checks the actual Sentinel-1
    # and Sentinel-2 acquisition dates before claiming a fresh observation.
    schedule="0 6 * * 1",
    start_date=pendulum.datetime(2026, 1, 1, 6, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(hours=2),
    },
    tags=[
        "cropintel",
        "ghana",
        "earth-engine",
        "weekly-monitoring",
    ],
)
def cropintel_monitoring():
    """Define the ordered weekly monitoring workflow and its dependencies."""

    @task
    def validate_inputs():
        """Validate the farm registry, polygon files and optional field records."""
        run_script("validate_inputs.py")

    @task
    def validate_environment():
        """Confirm that the configured Google Earth Engine project is accessible."""
        run_script("validate_environment.py")

    @task
    def capture_run_datetime():
        """Capture one actual UTC execution time for the complete report."""
        run_datetime = pendulum.now("UTC").isoformat()
        print(f"CropIntel report reference time: {run_datetime}", flush=True)
        return run_datetime

    @task
    def extract_s2_regional(run_datetime: str):
        """Extract Sentinel-2 from public COGs instead of Earth Engine.

        Runs only when CROPINTEL_S2_SOURCE is "stac". In "gee" mode it does
        nothing, because extract_data.py then computes Sentinel-2 itself and
        running both would make two scripts write the same rasters and the
        same clear-view state file.
        """
        # Read the same setting config.py uses. docker-compose.yaml sets it.
        source = os.getenv("CROPINTEL_S2_SOURCE", "gee").strip().lower()
        if source != "stac":
            # Earth Engine mode: finish successfully without doing anything,
            # so the downstream task still runs.
            print(
                f"CROPINTEL_S2_SOURCE is {source!r}; Sentinel-2 stays in "
                "extract_data.py (Earth Engine). Skipping.",
                flush=True,
            )
            return
        # Same run_datetime as extract_observed_data, so the newest history
        # window ends at the same moment in both scripts.
        run_script("extract_s2_regional.py", "--datetime", run_datetime)

    @task
    def extract_observed_data(run_datetime: str):
        """Extract observed satellite indicators for the actual report time."""
        run_script("extract_data.py", "--datetime", run_datetime)

    @task
    def extract_rain_forecast(run_datetime: str):
        """Extract the configured short-range rainfall forecast for every farm."""
        run_script(
            "extract_rain_forecast.py",
            "--datetime",
            run_datetime,
        )

    @task
    def extract_climate_water(run_datetime: str):
        """Extract ERA5-Land temperature, heat and ET0 climate background."""
        run_script(
            "extract_climate_water.py",
            "--datetime",
            run_datetime,
        )

    @task
    def validate_raw_outputs():
        """Verify that all required raw extraction files were produced."""
        run_script("validate_outputs.py", "raw")

    @task
    def transform():
        """Transform raw observations into conservative monitoring indicators."""
        run_script("transform_data.py")

    @task
    def validate_processed_outputs():
        """Verify the processed tables required by the reporting scripts."""
        run_script("validate_outputs.py", "processed")

    @task
    def create_crop_dashboard():
        """Render the farm-level crop-condition dashboard."""
        run_script("visualize_crop_dashboard.py")

    @task
    def create_water_climate_dashboard():
        """Render the water and weather monitoring dashboard."""
        run_script("visualize_water_climate_dashboard.py")

    @task
    def create_nutrition_dashboard():
        """Render the crop-nutrition attention dashboard."""
        run_script("visualize_nutrition_dashboard.py")

    @task
    def create_irrigation_dashboard():
        """Render the irrigation assessment dashboard."""
        run_script("visualize_irrigation_dashboard.py")

    @task
    def create_full_dashboard():
        """Render the combined farm-management summary dashboard."""
        run_script("visualize_full_dashboard.py")

    inputs_validated = validate_inputs()
    environment_validated = validate_environment()
    report_datetime = capture_run_datetime()

    sentinel2 = extract_s2_regional(report_datetime)
    observed = extract_observed_data(report_datetime)
    forecast = extract_rain_forecast(report_datetime)
    climate = extract_climate_water(report_datetime)

    raw_outputs_validated = validate_raw_outputs()
    analysis = transform()
    processed_outputs_validated = validate_processed_outputs()

    crop_dashboard = create_crop_dashboard()
    water_dashboard = create_water_climate_dashboard()
    nutrition_dashboard = create_nutrition_dashboard()
    irrigation_dashboard = create_irrigation_dashboard()
    full_dashboard = create_full_dashboard()

    inputs_validated >> environment_validated
    environment_validated >> report_datetime

    # extract_data.py reads s2_current.csv and s2_timeseries.csv in "stac"
    # mode, so Sentinel-2 must finish first. Both receive the same
    # report_datetime, which keeps their 10-day windows identical.
    sentinel2 >> observed

    [observed, forecast, climate] >> raw_outputs_validated

    raw_outputs_validated >> analysis
    analysis >> processed_outputs_validated

    processed_outputs_validated >> [
        crop_dashboard,
        water_dashboard,
        nutrition_dashboard,
        irrigation_dashboard,
        full_dashboard,
    ]


cropintel_monitoring()
