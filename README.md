# Fine tune a Tensorflow 2 BERT model for named entity recognition and build a knowledge graph of extracted entities

## Background and motivation

Knowledge graphs are a powerful way to represent and connect items with properties. For example connecting products according to their characteristics, connecting online news articles according to the subjects they talk about, etc. This enables to search, compare or recommend items by their properties effectively

As a consequence, our customers often ask us to build internal search and comparison engines running on graph databases. However, data is often a limiting factor: while text descriptions are often available for all items, properties are regularly missing or inconsistent. Those properties can be however extracted in a standardize manner through named entity recognition.

In this repository we present how to build a custom named entity recognition, by fine-tuning BERT on Tensorflow 2 with Keras using the [SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/) and then build and populate a knowledge graph of items through these extracted properties using [Amazon Neptune](https://aws.amazon.com/neptune/) 

Amazon SageMaker is a fully managed service that provides developers and data scientists with the ability to build, train, and deploy machine learning (ML) models quickly. Amazon SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high-quality models. The SageMaker Python SDK provides open source APIs and containers that make it easy to train and deploy models in Amazon SageMaker with several different machine learning and deep learning frameworks. We use an Amazon SageMaker Notebook Instance for running the code. For information on how to use Amazon SageMaker Notebook Instances, see the [AWS documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/nbi.html).

Amazon Neptune is a fast, reliable, fully managed graph database service that makes it easy to build and run applications that work with highly connected datasets. The core of Neptune is a purpose-built, high-performance graph database engine. This engine is optimized for storing billions of relationships and querying the graph with milliseconds latency. Neptune powers graph use cases such as recommendation engines, fraud detection, knowledge graphs, drug discovery, and network security.

## Quick start

### Named Entity Recognition with Tensorflow 2 Bert on SageMaker 

To train and deploy the custom named entity recognition on sagemaker follow the in notebooks/ner-bert-keras-sagemaker.ipynb

### Neptune cluster

Follow the steps and cloudformation steps avaialbe in Amazon Neptune documentation to start up cluster or use one of the following link to get the cfn stack:

- For Neptune Cluster: https://s3.amazonaws.com/aws-neptune-customer-samples/v2/cloudformation-templates/neptune-base-stack.json. This will also create a VPC in you account that Neptune Cluster will be running in, if you want to use your own VPC setup you will need to edit the CFN accordingly. 
- For Neptune workbench/Notebook: https://s3.amazonaws.com/aws-neptune-customer-samples/v2/cloudformation-templates/neptune-sagemaker-notebook-stack.json.

Once these steps are over, follow the instructions in notebooks/knowledge-graph-neptune.ipynb

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

