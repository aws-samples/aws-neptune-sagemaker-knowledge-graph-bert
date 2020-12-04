# Fine-tune a Tensorflow 2 BERT model for a custom named entity recognition and build a knowledge graph of extracted entities

## Background and motivation

Knowledge graphs are a powerful way to represent and connect items with properties. For example connecting products according to their characteristics, connecting online news articles according to the subjects they talk about, etc. This enables to search, compare or recommend items by their properties effectively

As a consequence, our customers often ask us to build internal search and comparison engines running on graph databases. However, data is often a limiting factor: while text descriptions are often available for all items, properties are regularly missing or inconsistent. Those properties can be however extracted in a standardize manner through named entity recognition.

In this repository we present how to build a custom named entity recognition, by fine-tuning BERT on Tensorflow 2 with Keras using the [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) and then build and populate a knowledge graph of items through these extracted properties using [Amazon Neptune](https://aws.amazon.com/neptune/) 

Amazon SageMaker is a fully managed service that provides developers and data scientists with the ability to build, train, and deploy machine learning (ML) models quickly. Amazon SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high-quality models. The SageMaker Python SDK provides open source APIs and containers that make it easy to train and deploy models in Amazon SageMaker with several different machine learning and deep learning frameworks. We use an Amazon SageMaker Notebook Instance for running the code. For information on how to use Amazon SageMaker Notebook Instances, see the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html).

Amazon Neptune is a fast, reliable, fully managed graph database service that makes it easy to build and run applications that work with highly connected datasets. The core of Neptune is a purpose-built, high-performance graph database engine. This engine is optimized for storing billions of relationships and querying the graph with milliseconds latency. Neptune powers graph use cases such as recommendation engines, fraud detection, knowledge graphs, drug discovery, and network security.

## Quick start

### Named Entity Recognition with Tensorflow 2 Bert on SageMaker 

You can train and deploy the custom named entity recognition by following [this](notebooks/ner-bert-keras-sagemaker.ipynb) notebook on [Amazon SageMaker](https://aws.amazon.com/sagemaker/).

### Neptune cluster

You can either create a Neptune cluster by following [Amazon Neptune documentation](https://docs.aws.amazon.com/neptune/latest/userguide/get-started-create-cluster.html) or use **Launch Stack** below to launch a cloudformation stack in a region of your choice. 

* Create a Neptune cluster with a stack:

Region name	| Region code	| Launch 
--- | --- | --- 
US East (N. Virginia)	| us-east-1	| Launch Stack (Replace Launch stack with steps in this [blogpost](https://aws.amazon.com/blogs/devops/construct-your-own-launch-stack-url/) 
US West (Oregon)	| us-west-2	| Launch Stack
Europe (Ireland)	| eu-west-1	| Launch Stack

* Launch Neptune workbench/Notebook, launch a stack

Region name	| Region code	| Launch 
--- | --- | --- 
US East (N. Virginia)	| us-east-1	| Launch Stack (Replace Launch stack with steps in this [blogpost](https://aws.amazon.com/blogs/devops/construct-your-own-launch-stack-url/) 
US West (Oregon)	| us-west-2	| Launch Stack
Europe (Ireland)	| eu-west-1	| Launch Stack

Once, you have created your Neptune cluster and workbench/notebook, you can follow the instructions in [this](notebooks/knowledge-graph-neptune.ipynb) notebook.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Authors
- Viktor Malesevic (https://github.com/ViktorMalesevic)
- Fatema Alkhanaizi (https://github.com/Fatema)
- Othmane Hamzaoui (https://github.com/Othmane796)

