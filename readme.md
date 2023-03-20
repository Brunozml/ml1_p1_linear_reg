# Machine Learning: Linear Regression
## Sentiment analysis
#### introduction

This **project** is my first ML endevour. It was inspired by MIT's [Machine Learning with Python](https://learning.edx.org/course/course-v1:MITx+6.86x+1T2023/) course.

> Most of the code was written by me; some was provided by the MIT staff

objective: design a binary classifier to use for sentiment analysis of product reviews.

I'll make use of the perceptron algorithm, the average perceptron algorithm, and the Pegasos algorithm.

----
# Data Example

| Review (x)                                                                                               | Label (y) |
|----------------------------------------------------------------------------------------------------------|-----------|
| "*Nasty No flavor. The candy is just red, No flavor. Just plan and chewy. I would never buy them again*" | -1        |
| "*YUMMY! You would never guess that they're sugar-free and it's so great that you can eat them pretty much guilt free! i was so impressed that i've ordered some for myself (w dark chocolate) to take to the office. These are just EXCELLENT!*"                                                                                                     | 1         |

File structure
---
The archive contains the data files in the Data directory, along with the following python files:

1. `project1.py` contains the main functions: the perceptron algorithm (and its variations), Pegasos, and Classifier Accuracy (among others)
2. `main.py` takes care of the execution of the functions in `project1.py`. Most of these are commented out .
3. `test.py` is a script which runs tests on a few of the methods you will implement. Most of these were written by MIT staff.
4. `utils.py` contains utility functions for loading data.  Most of these were written by MIT staff.



---
## Task list
- [x] Write the readme file
- [ ] use python debugger `pdb` effectively
- [ ] Feature engineering: explore using the length of the text, ALL-CAP WORDS, 
   and word embeddings as features to improve performance of algorithm.
- [ ]



```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```
