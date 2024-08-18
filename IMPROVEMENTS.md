# Tag Search API w/ UI

## Thoughts

### 1. Warm-Up: Benchmarking

A benchmark script benchmark_script.py has been created to test the performance

What metrics can be the most important?

GPU is used to speed up the application that's why the most logical metrics to measure the performance
improvement would be tracking time needed to perform the repetitive operations many times or batch operations.   

### 2. Main part-1: Manual Tag Overrides

The logic to store and return manual tags if they are present has been incorporated. Manual tags are stored
in the collection with the prefix 'manual_' as a payload parameter and the endpoint to create them is a post 
endpoint "/update_manual/".

### 3. Main part-2: Size Optimization

The PCA technique has been implemented to reduce vector size and the resulted reduced vectors are stored
in the collection with the prefix 'reduced_'.

How would you measure the performance of the reduced vectors?
- Explained variance of the initial set of vectors
- Reconstruction error
- Possible comparison with the result for original vectors (if the result of semantic matches 
for original vectors  taken as ground truth it is possible to elaborate some heuristic to calculate the difference
but in its essence it will be something similar to the reconstruction error)

### 3. Tear-Down: Improvements and Optimizations

Code optimizations / Solution optimizations

- Precalculate the most reasonable number of components for PCA
- Add batching and asynchronous processing
- Pre-compute and store nearest neighbors for common queries
- Cache frequently requested query results
- Add monitoring    

Maintenance improvements
 
- Documentation
- API Versioning to track breaking changes
- Monitoring and Logging
- Testing
- Data validation
- CI Pipeline,  
- Set up automated backups of the Qdrant database, Auto-Scaling

Deployment strategies
    
- Kubernetes clusters
- Amazon ECS, EKS
- CI/CD Pipeline


Any other changes you would implement if given more time

- Improve Gradio interface look
- Add checkboxes to get manual tags
- When manual tags are present for a search query and if a number of desired matches to be found is greater than a number of corresponding manual tags
it would be nice to merge the list of manual tags with found semantic tags in the corresponding order to get the required 
search number provided.
- Introduce feedback score from a user based on the tags received. 
- Precalculate results for the most popular queries   