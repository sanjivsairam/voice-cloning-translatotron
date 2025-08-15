from celery import Celery
from app.t2_pipeline import run_pipeline
import os

celery = Celery("tasks", broker="redis://redis:6379/0")

@celery.task
def process_translation(input_path, target_lang, task_id):
    output_path = f"outputs/{task_id}.wav"
    run_pipeline(input_path, output_path, target_lang)
    return output_path

def add_translation_task(input_path, target_lang):
    task_id = os.path.basename(input_path).split(".")[0]
    process_translation.delay(input_path, target_lang, task_id)
    return task_id

# app/t2_pipeline.py
def run_pipeline(input_path, output_path, target_lang):
    # Placeholders for actual Translatotron2 + OpenVoice
    import shutil
    shutil.copy(input_path, output_path)
    # Replace with real processing logic
