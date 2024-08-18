import json
import logging
import os
from typing import List

import gradio as gr
import pandas as pd
import requests

# Define the URL for the FastAPI service
BASE_URL = os.environ.get("API_BASE_URL", default="http://app:8000")


def upload_csv(file):
    filename = file.name
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/csv')}
        response = requests.post(f"{BASE_URL}/upload-csv/", files=files)
        if response.status_code == 200:
            return response.json()['message']
        else:
            return response.json()['detail']


def update_manual_tags(query, *manual_tags):
    if query == "":
        raise gr.Error("Please enter a search query")
    manual_tags = [tag for tag in manual_tags if tag.strip()]
    response = requests.post(f"{BASE_URL}/update_manual/",  params={'query': query}, json=manual_tags)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return response.json()['detail']

def search_names(query, k=5, reduced=False):
    if query == "":
        raise gr.Error("Please enter a search query")

    if k < 1:
        raise gr.Error("Number of results should be at least 1")

    response = requests.get(f"{BASE_URL}/search/", params={'query': query, 'k': k, 'reduced': reduced})
    if response.status_code == 200:
        data = response.json()
        semantic_results = data['match']['semantic']
        typo_results = data['match']['typo']

        semantic_df = pd.DataFrame(semantic_results)
        typo_df = pd.DataFrame(typo_results)

        return semantic_df, typo_df
    else:
        # raise gr.Error(response.json()['detail'])
        return response.json()['detail'], pd.DataFrame()


def delete_collection():
    response = requests.delete(f"{BASE_URL}/purge/")
    if response.status_code == 200:
        return response.json()['message']
    else:
        return response.json()['detail']


def check_task_status():
    response = requests.get(f"{BASE_URL}/task-status/")
    if response.status_code == 200:
        resp_str = json.dumps(response.json(), indent=4)
        return resp_str
    else:
        return "Something bad happened"


def create_pca_collection(num_components: int):
    if num_components < 1:
        raise gr.Error("Number of components should be at least 1")
    response = requests.post(f"{BASE_URL}/create_reduced_collection/", params={"num_components": num_components})
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return response.json()['detail']


# Gradio interface
with gr.Blocks(css="body { font-family: Arial, sans-serif; } footer { visibility: hidden }") as demo:
    gr.Markdown(
        "<h3 style='text-align: center;'>Tag Search API</h3>")

    with gr.Tab("Search"):
        num_components = gr.Number(label="Number of components for PCA reduction", elem_classes="tab-container")
        create_button = gr.Button("Create PCA reduction collection", elem_classes="tab-container")
        with gr.Row():
            message = gr.Textbox(label="message", interactive=False)
            variance = gr.Number(label="explained variance", interactive=False)
            error = gr.Number(label="reconstruction error", interactive=False)
        create_button.click(fn=create_pca_collection, inputs=num_components, outputs=[message, variance, error])

        with gr.Row():
            query = gr.Textbox(label="Search Query", placeholder="Enter tag to search", elem_classes="tab-container")
            k = gr.Number(label="Number of Results", value=5, elem_classes="tab-container")
        search_button = gr.Button("Search", elem_classes="tab-container")

        with gr.Row():
            with gr.Column():
                semantic_table = gr.Dataframe(label="Semantic Matches", interactive=False, elem_classes="tab-container")
                typo_table = gr.Dataframe(label="Typo Matches", interactive=False, elem_classes="tab-container",visible=True)
                search_button.click(fn=search_names, inputs=[query, k], outputs=[semantic_table, typo_table])
                query.submit(fn=search_names, inputs=[query, k], outputs=[semantic_table, typo_table])

            with gr.Column():
                reduced = gr.Checkbox(label="Reduced", value=True, visible=False)
                semantic_table2 = gr.Dataframe(label="Semantic Matches for reduced vectors", interactive=False, elem_classes="tab-container")
                typo_table2 = gr.Dataframe(label="Typo Matches for reduced vectors", interactive=False, elem_classes="tab-container",  visible=True)
                search_button.click(fn=search_names, inputs=[query, k, reduced], outputs=[semantic_table2, typo_table2])
                query.submit(fn=search_names, inputs=[query, k, reduced], outputs=[semantic_table2, typo_table2])

    with gr.Tab("Update manual tags"):
        with gr.Row():
            query = gr.Textbox(label="Query", placeholder="Enter tag to update", elem_classes="tab-container")
            # manual_tags = gr.List(label="List of tags", value=[['asd', 'ds']], elem_classes="tab-container")
            with gr.Column():
                manual_tags = [gr.Textbox(label=f"Tag {i + 1}", placeholder=f"Enter tag {i + 1}") for i in range(5)]
        update_button = gr.Button("Update", elem_classes="tab-container")
        output = gr.Textbox(label="Manual tags", interactive=False, elem_classes="tab-container")
        update_button.click(fn=update_manual_tags, inputs=[query, *manual_tags], outputs=output)
        query.submit(fn=update_manual_tags, inputs=[query, *manual_tags], outputs=output)

    with gr.Tab("Upload CSV"):
        with gr.Accordion("Information", open=False):
            gr.Markdown("""## Info
                        - the csv file must be **comma separated**
                        - the csv file **should have a 'name' column**, anything else will be discarded
                        - processing time depends on the number of records, you can check the status with the button below
                        """)
        csv_file = gr.File(label="Upload CSV File", type='filepath', elem_classes="tab-container")
        upload_button = gr.Button("Upload", elem_classes="tab-container")
        upload_status = gr.Textbox(label="Upload Status", interactive=False, elem_classes="tab-container")

        upload_button.click(fn=upload_csv, inputs=csv_file, outputs=upload_status)

        status_button_upload = gr.Button("Check Task Status", elem_classes="tab-container")
        status_display_upload = gr.Textbox(label="Task Status", interactive=False, elem_classes="tab-container")
        status_button_upload.click(fn=check_task_status, outputs=status_display_upload)

        delete_button = gr.Button("Delete Collection", elem_classes="tab-container")
        delete_status = gr.Textbox(label="Delete Status", interactive=False, elem_classes="tab-container")
        delete_button.click(fn=delete_collection, outputs=delete_status)

# Launch the interface
demo.launch(server_name="0.0.0.0")
