# Tag Search API w/ UI

## Overview

Welcome to the Tag Search API project! 👋

The objective is to develop a search API for a website that allows users to enter any free-text query and return the closest matches.
This functionality ensures that users can find what they are looking for even if they don't know the exact name of the tag, and it also accommodates typos in the search queries.

The current codebase implements a method that searches for close matches based on both the structure and semantic meaning of the word(s).
The `.csv` file (that you can generate yourself) contains the existing tags on the website, and the task is to match user queries with these tags.

## Submission Instructions

Please work in a private repository so the changes are easily trackable.

1. Make an initial commit with the original provided codebase.
2. Implement the required changes and improvements.
3. Push your changes to the private repository.
4. Share access to the repository with us.

Rules:

- You don't have to write extensive documentation
- You can use any libraries or tools you think are necessary
- You can modify the existing codebase as you see fit
- You can add new files or directories

**Thank you for your participation and good luck with your assignment! ❤️**

## Data

You can use the `create_dummy_tags.py` file to generate sample tags.

---

## Tasks

### 1. Warm-Up: Benchmarking

- Create a benchmark script with which we can test the difference between using a GPU and CPU
- What metrics can be the most important?

### 2. Main part-1: Manual Tag Overrides

In certain scenarios, we would like to manually define the closest representations of a tag within the tag list.
For example, we may want to define that "Moon Mission" is closely related to "Moon" and "Apollo 11".
This means that when someone searches for "Moon Mission", the solution should return matches that are closest to the defined synonyms (such as "Moon" and "Apollo 11") and not the original query.

Originally, without any modification to the solution, if the user searched for "Moon Mission", they would get results based on semantic similarity.
We want to override this behavior with manually defined synonyms.

Your task is to incorporate this in the solution and to create an additional API endpoint that allows us to override certain tags with their closest representations.

### 3. Main part-2: Size Optimization

Right now, the embedding vectors are too big for us. We would like to reduce them to a smaller size and use the reduced vectors for the search.
Implement/use a technique that accomplishes this.

You can select any technique you think is suitable for this task.

How would you measure the performance of the reduced vectors?

### 3. Tear-Down: Improvements and Optimizations

While the codebase provides a functional proof of concept, it needs enhancements to be production-ready.
Please provide your thoughts on (in a `IMPROVEMENTS.md` file):

- Code optimizations / Solution optimizations
- Maintenance improvements
- Deployment strategies
- Any other changes you would implement if given more time
