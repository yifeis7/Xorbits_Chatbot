# Use LangChain & Xorbits to perform scientific computations with simple prompts and customized data

[Xorbits](https://doc.xorbits.io/en/latest/), an open-source computing framework, is revolutionizing the way data science and machine learning workloads are scaled. It simplifies every step of the process, from data loading and preprocessing to tuning, training, and model serving. By integrating with LangChain, Xorbits provides a seamless experience for users who want to parallelize their Pandas and Numpy operations on their own data. This integration allows Large Language Models (LLMs) to leverage the distributed and multi-threading features of Xorbits, offering users an all-inclusive solution for their data-related needs.

One of the key features of this integration is the Xorbits Pandas and Numpy Agent. This groundbreaking tool transforms the way users interact with custom dataframes and ndarrays. It leverages the power of natural language prompts, providing users with the ability to perform intricate data operations without requiring specialized programming skills. This agent is designed to make data manipulation more accessible and intuitive, enabling users to input their data and communicate with it using natural language, thereby facilitating efficient data analysis, transformation, and visualization.

In this comprehensive guide, we will delve into a multitude of examples showcasing how to use the Xorbits Pandas and Numpy Agent with two popular libraries in data analysis: Pandas and Numpy. Pandas is a software library for Python, providing flexible data structures designed to facilitate working with "relational" or "labeled" data. Numpy is another library for the Python programming language that adds support for large, multi-dimensional arrays and matrices, along with a collection of high-level mathematical functions to operate on these arrays.

The examples provided in this guide are meticulously designed to showcase the power, versatility, and user-friendly nature of the Xorbits Pandas and Numpy Agent. By the end of this guide, you will have a clear understanding of how this tool can revolutionize your data manipulation and analysis processes, making them more efficient and insightful. Regardless of your expertise level in data analysis, this guide will equip you with a powerful new tool to enhance your data workflow.

## Xorbits Pandas and Numpy Agent

The Xorbits Pandas and Numpy Agent is an innovative tool that enables users to manipulate custom dataframes and ndarrays using natural language prompts. With this powerful agent, users can easily perform complex data operations without the need for specialized programming skills. Simply input your data and use natural language to interact with it, allowing you to quickly and efficiently analyze, transform, and visualize your data. Whether you're a data scientist or a business analyst, the Xorbits Pandas and Numpy Agent is an indispensable tool for streamlining your data workflow and unlocking valuable insights.

Here are some typical examples:

### Pandas

```Python
import xorbits.pandas as pd
from langchain.agents import create_xorbits_agent
from langchain.llms import OpenAI
data = pd.read_csv("titanic.csv")
agent = create_xorbits_agent(OpenAI(temperature=0), data, verbose=True)
```

Output:

```Python
agent.run(
    "Show the number of people whose age is greater than 30 and fare is between 30 and 50 , and pclass is either 1 or 2"
)
```

```
> Entering new  chain...
Thought: I need to filter the dataframe to get the desired result
Action: python_repl_ast
Action Input: data[(data['Age'] > 30) & (data['Fare'] > 30) & (data['Fare'] < 50) & ((data['Pclass'] == 1) | (data['Pclass'] == 2))].shape[0]
Observation: 20
Thought: I now know the final answer
Final Answer: 20

> Finished chain.
'20'
```

```Python
agent.run("Group the data by sex and find the average age for each group")
```

Output:

```
> Entering new  chain...
Thought: I need to group the data by sex and then find the average age for each group
Action: python_repl_ast
Action Input: data.groupby('Sex')['Age'].mean()

Observation: Sex
female    27.915709
male      30.726645
Name: Age, dtype: float64
Thought: I now know the average age for each group
Final Answer: The average age for female passengers is 27.92 and the average age for male passengers is 30.73.

> Finished chain.
'The average age for female passengers is 27.92 and the average age for male passengers is 30.73.'
```

### Numpy

#### One-dimensional array

```Python
import xorbits.numpy as np
from langchain.agents import create_xorbits_agent
from langchain.llms import OpenAI
arr = np.array([1, 2, 3, 4, 5, 6])
agent = create_xorbits_agent(OpenAI(temperature=0), arr, verbose=True)
```

```Python
agent.run(
    "Reshape the array into a 2-dimensional array with 2 rows and 3 columns, and then transpose it"
)
```

Output:

```
> Entering new  chain...
Thought: I need to reshape the array and then transpose it
Action: python_repl_ast
Action Input: np.reshape(data, (2,3)).T

Observation: [[1 4]
 [2 5]
 [3 6]]
Thought: I now know the final answer
Final Answer: The reshaped and transposed array is [[1 4], [2 5], [3 6]].

> Finished chain.
'The reshaped and transposed array is [[1 4], [2 5], [3 6]].'
```

```Python
agent.run(
    "Reshape the array into a 2-dimensional array with 3 rows and 2 columns and sum the array along the first axis"
)
```

Output:

```
> Entering new  chain...
Thought: I need to reshape the array and then sum it
Action: python_repl_ast
Action Input: np.sum(np.reshape(data, (3,2)), axis=0)

Observation: [ 9 12]
Thought: I now know the final answer
Final Answer: The sum of the array along the first axis is [9, 12].

> Finished chain.
'The sum of the array along the first axis is [9, 12].'
```

#### Two-dimensional array

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
agent = create_xorbits_agent(OpenAI(temperature=0), arr, verbose=True)
```

```python
agent.run("calculate the covariance matrix")
```

Output:

```
> Entering new  chain...
Thought: I need to use the numpy covariance function
Action: python_repl_ast
Action Input: np.cov(data)

Observation: [[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
Thought: I now know the final answer
Final Answer: The covariance matrix is [[1. 1. 1.], [1. 1. 1.], [1. 1. 1.]].

> Finished chain.
'The covariance matrix is [[1. 1. 1.], [1. 1. 1.], [1. 1. 1.]].'
```

```Python
agent.run("compute the U of Singular Value Decomposition of the matrix")
```

Output:

```
> Entering new  chain...
Thought: I need to use the SVD function
Action: python_repl_ast
Action Input: U, S, V = np.linalg.svd(data)
Observation: 
Thought: I now have the U matrix
Final Answer: U = [[-0.70710678 -0.70710678]
 [-0.70710678  0.70710678]]

> Finished chain.
'U = [[-0.70710678 -0.70710678]\n [-0.70710678  0.70710678]]'
```

## Summary

LangChain is a powerful framework that allows developers to build applications powered by language models. It enables the interaction between language models and personalized data sources, opening up new possibilities for querying and analyzing data. With its modular components, chains, and agents, LangChain provides customized workflows for specific tasks. The integration of Xorbits Pandas and Numpy agents further enhances data manipulation capabilities. 