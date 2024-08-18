import time
import requests
import psutil
from create_dummy_tags import tags_df
from query import Query, BatchQuery

FASTAPI_URL = "http://localhost:8111"


def benchmark_upload_csv(filename):
    start_time = time.time()
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/csv')}
        response = requests.post(f"{FASTAPI_URL}/upload-csv/", files=files)
    end_time = time.time()
    if response.status_code == 200:
        response = response.json()['message']
    else:
        response = response.json()['detail']
    return response, end_time - start_time

def benchmark_search(query, num_searches=100):
    total_time = 0
    for _ in range(num_searches):
        start_time = time.time()
        response = requests.get(FASTAPI_URL + "/search/", params={"query": query, "k": 5})
        # print(f"response: {response.json()}")
        end_time = time.time()
        total_time += end_time - start_time
    return total_time / num_searches


def benchmark_batch_search(query: Query, num_searches=100):
    start_time = time.time()
    queries = [query]*num_searches
    responses = requests.post(FASTAPI_URL + "/batch/", json=[query.model_dump() for query in queries])
    # print(f"responses: {responses.json()}")
    end_time = time.time()
    total_time = end_time - start_time
    return total_time / num_searches

# def measure_resource_usage(duration):
#     cpu_usage = []
#     mem_usage = []
#     for _ in range(duration):
#         cpu_usage.append(psutil.cpu_percent(interval=1))
#         mem_usage.append(psutil.virtual_memory().percent)
#     return cpu_usage, mem_usage


if __name__ == "__main__":

    print("Benchmarking Upload csv operation...")
    try:
        filename = "dummy_tags.csv"
        response, upload_time = benchmark_upload_csv(filename)
        print(f"Upload csv: {response}, Time taken: {upload_time:.4f} seconds")
    except FileNotFoundError as e:
        print(e)

    query = {"query": "I am here", "k": 5}
    print("Benchmarking Search Operation...")
    avg_search_time = benchmark_search(query["query"], 100)
    print(f"Average Search Time: {avg_search_time:.4f} seconds")

    print("Benchmarking Batch Search Operation...")
    avg_batch_time = benchmark_batch_search(Query(**query), 100)
    print(f"Average Batch Search Time per item: {avg_batch_time:.4f} seconds")

    # print("Measuring Resource Usage...")
    # cpu_usage, mem_usage = measure_resource_usage(test_duration)
    # print(f"Average CPU Usage: {sum(cpu_usage) / len(cpu_usage):.2f}%")
    # print(f"Average Memory Usage: {sum(mem_usage) / len(mem_usage):.2f}%")
