import subprocess

import os
from pathlib import Path
from monai.bundle import run
import click
import mlflow

import shutil


os.environ["nnUNet_raw"] =""
os.environ["nnUNet_preprocessed"]=""
os.environ["nnUNet_results"]=""


@click.command()
@click.option("--bundle-dir", type=str)
@click.option("--minio-url", type=str)
@click.option("--bucket-name", type=str)
@click.option("--input-folder", type=str)
@click.option("--output-folder", type=str)
def run_minio_inference(minio_url, bucket_name,input_folder ,output_folder,bundle_dir):

    subprocess.run(["/workspace/mc", "alias", "set", "minio", minio_url, os.environ["MINIO_ACCESS_KEY"], os.environ["MINIO_SECRET_KEY"]])
    subprocess.run(["/workspace/mc", "cp", "--recursive", f"minio/{bucket_name}/{input_folder}", "/"])

    bundle_path = str(Path(bundle_dir))
    os.environ['PYTHONPATH'] = bundle_path

    step = "inference"

    run(
        run_id=step,
        meta_file=f"{bundle_path}/configs/metadata.json",
        config_file=f"{bundle_path}/configs/{step}.yaml",
        bundle_root=bundle_path,
        data_dir="/"+input_folder,
        output_dir="/output",
        logging_file=f"{bundle_path}/configs/logging.conf",

    )
    subprocess.run(["/workspace/mc", "cp", "--recursive", "/output",f"minio/{bucket_name}/{output_folder}",])


def download_bundle(dst_folder,BUNDLE_NAME, MLFLOW_MODEL_NAME, MLFLOW_MODEL_VERSION):
    #jwt_payload = jwt.decode(os.environ["MLFLOW_TRACKING_TOKEN"],options={"verify_signature": False})
    #mlflow.set_tag("user",jwt_payload["preferred_username"])
    mlflow.pytorch.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}",dst_path=dst_folder)
    shutil.unpack_archive(Path(dst_folder).joinpath(f"extra_files/{BUNDLE_NAME}.tar.gz"),dst_folder)


@click.command()
@click.option("--bundle-dir", type=str)
@click.option("--input-folder", type=str)
@click.option("--output-folder", type=str)
def run_inference(bundle_dir, output_folder, input_folder):


    bundle_path = str(Path(bundle_dir))
    os.environ['PYTHONPATH'] = bundle_path

    
    step = "inference"

    run(
        run_id=step,
         meta_file=f"{bundle_path}/configs/metadata.json",
               config_file=f"{bundle_path}/configs/{step}.yaml",
                bundle_root=bundle_path,
                data_dir= input_folder,
                outputd_ir = output_folder,
                logging_file= f"{bundle_path}/configs/logging.conf",


    )


if __name__ == "__main__":
    #run_inference()
    run_minio_inference()

