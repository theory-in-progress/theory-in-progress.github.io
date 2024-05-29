---
author: ["Jyotirmay Khavasi"]
title: "About my Technical Skills"
date: "2024-05-29"
description: "About my Technical proficiency"
summary: "About my Technical proficiency"
tags: ["markdown"]
categories: ["tech", "skills", "proficiency"]
series: ["My-Posts"]
ShowToc: true
TocOpen: true
---

## Describe a machine learning project you have worked on. What algorithms and tools did you use, and what was the outcome?

I have worked on many projects spanning from use cases of LLMs Retrival Augmented Generation to Core Deep Learning and GPU distrubuted training of text, images and documents.

One of the most impactful machine learning projects I worked on was the development of a table detection pipeline using OpenCV for my role at Wolters Kluwer. This project required an in-depth understanding of computer vision techniques and the ability to create a comprehensive pipeline from data preprocessing to deployment.

### Project Overview

The primary goal was to accurately detect and extract tabular data from PDFs, which involved several steps:

1. **Data Augmentation and Preprocessing**: I used various image processing techniques to enhance the quality of the input data. This included operations like scaling, rotation, and noise addition to improve the robustness of the model.
2. **Table Detection**: For detecting tables within the document images, I employed OpenCV’s object detection and image segmentation capabilities. This step involved the use of contour extraction and morphological operations to identify the table boundaries accurately.
3. **Entity Extraction**: Once the tables were detected, the next step was to extract the individual cells and their contents. This was achieved by employing bounding boxes and kernels, leveraging coordinate geometry to precisely segment and reconstruct the table’s structure.

### Tools and Technologies

- **OpenCV**: The core library used for image processing and computer vision tasks.
- **Python**: The primary programming language for scripting and implementing the pipeline.
- **Hugging Face Optimum**: Utilized for converting models to ONNX format, which significantly reduced evaluation time.
- **Distributed Data Parallel (DDP)**: Used for efficient training and evaluation of the models across multiple GPUs, which helped in handling large datasets and minimizing idle time during processing.

### Outcome

The implementation of this pipeline led to a substantial improvement in the accuracy and efficiency of table detection and data extraction processes. Specifically:

- **Performance Improvement**: The optimized models showed a 4% improvement across all relevant metrics.
- **Efficiency Gains**: By converting models to the ONNX format, I achieved a 40% reduction in evaluation time for clients.
- **Scalability**: The use of multi-GPU processing allowed for faster training and evaluation, making the solution scalable for larger datasets.

<!-- This project not only honed my skills in computer vision and deep learning but also demonstrated my ability to deliver production-ready solutions that significantly enhance operational efficiency. -->
---

## Which programming languages are you proficient in (e.g., Python, R, Java)? Please provide examples of how you have used them in past projects.

I am proficient in several programming languages, with Python being my primary language. Here's a detailed overview of how I've utilized Python and other languages in various projects:

### Python

**Experience**: Extensive use during internships, open-source contributions, and academic projects.

1. **Google Summer of Code**:
   - **Project**: Contributing to the PyTorch-Ignite library.
   - **Details**: I worked on improving deep learning models and developing reinforcement learning models for the OpenCarRacing-Gym environment of OpenAI. This required a deep understanding of Python's data structures, object-oriented programming, and advanced features.
   - **Tasks**:
     - Enhanced CI/CD pipelines using GitHub workflows.
     - Implemented Docker containerization for easier deployment.
     - Wrote event filters for the engine and doctests for various metrics.

2. **Wolters Kluwer Internship**:
   - **Project**: Optimizing production-deployed models and developing an OpenCV-based table detection pipeline.
   - **Details**: Utilized Python extensively for data augmentation, object detection, image segmentation, and deploying models in production environments.
   - **Tasks**:
     - Converted models to ONNX format using Hugging Face Optimum.
     - Implemented a multi-GPU processing solution for efficient model training and evaluation.

### C++

**Experience**: Used in systems programming and backend development.

1. **Project**: Developing parsers, shells, and loadable kernel modules.
   - **Details**: Wrote parsers and shells to improve system interactions and enhance functionality. Developed kernel modules to extend operating system capabilities.
   - **Tasks**:
     - Implemented code generators for efficient compilation processes.
     - Ensured system stability and performance through rigorous testing and debugging.

### SQL

**Experience**: Utilized for data manipulation and database management.

1. **Project**: Data Science projects and internships.
   - **Details**: Used SQL for querying and managing data in relational databases.
   - **Tasks**:
     - Wrote complex SQL queries for data extraction and analysis.
     - Managed MySQL databases in conjunction with Python-based applications.

### Tools and Technologies

- **Git**: Proficient in using version control systems for collaborative development.
- **Docker**: Experienced in containerizing applications to ensure consistency across environments.
- **OpenCV**: Utilized for computer vision tasks, such as image processing and object detection.
- **PyTorch**: Extensive use for deep learning projects, including model training and evaluation.
- **Hugging Face Optimum**: Used for model optimization and deployment.

### Examples of Use

- **Python**: Developing reinforcement learning models, optimizing production models, creating data pipelines, and contributing to open-source projects.
- **C++**: Writing system-level code for parsers, shells, and kernel modules to enhance system functionalities.
- **Golang**: Building distributed systems that handle concurrent processes efficiently.
- **SQL**: Extracting and managing data within large-scale data science projects.

Overall, my proficiency in these languages has enabled me to tackle a wide range of technical challenges, from low-level system programming to high-level data science and machine learning applications.

---

## Describe your work experience in NLP

### Work Experience in Natural Language Processing (NLP)

My experience in Natural Language Processing (NLP) spans multiple projects and roles, where I have leveraged my skills to develop and optimize various NLP models and applications.

### Internship at Wolters Kluwer

During my time as a Data Science Intern at Wolters Kluwer, I worked on optimizing production-deployed models, which included Vision Encoder-Decoder and Layout Transformer models. A significant aspect of this role involved improving the models’ performance on tasks that required accurate text extraction and transformation from complex document structures. I utilized advanced NLP techniques to enhance the models' ability to understand and process text within different layouts, ensuring more accurate and efficient information retrieval and processing.

### Google Summer of Code @ PyTorch-Ignite

As an open-source contributor during my Google Summer of Code tenure, I contributed to the PyTorch-Ignite library, where I developed templates for Reinforcement Learning using algorithms like DQN and Advantage Actor Critic. Although primarily focused on reinforcement learning, this experience required a deep understanding of NLP techniques for processing and interpreting textual data used in training the models. Additionally, I implemented enhancements to CI/CD pipelines and Docker containerization, which improved the deployment and scalability of NLP models.

### Research Internship at HCL Technologies

At HCL Technologies, I worked on a project focused on extracting rules from textual data using NLP. I developed a custom BERT-based architecture for Named Entity Recognition (NER), achieving a recall of 88%. This involved preprocessing textual data, training the model to identify and classify entities accurately, and refining the model to improve its performance. Furthermore, I employed Random Forests for classifying sentences into 'Rule' or 'Not Rule' categories, achieving a 92% accuracy. I also created custom parse trees to convert the recognized rules into logical mathematical expressions, demonstrating my ability to handle complex NLP tasks and integrate them into larger systems.

### Technical Skills and Tools

Throughout these experiences, I have developed proficiency in using various NLP tools and libraries, including:

- **BERT**: For tasks like named entity recognition and text classification.
- **Hugging Face Transformers**: For leveraging pre-trained models and fine-tuning them for specific NLP tasks.
- **Python**: Extensive use of Python for scripting, model development, and integration.
- **PyTorch**: For building and training deep learning models.
- **OpenCV**: For combining NLP with computer vision tasks, particularly in document processing applications.

<!-- These roles have equipped me with a solid foundation in NLP, enabling me to tackle complex text processing challenges and contribute effectively to advancing the field. -->

---

## Describe your experience training large language models like BERT, GPT, Llama, etc?

### Experience Training Large Language Models

I have extensive experience working with large language models (LLMs) such as BERT, GPT-2, and LLaMA, focusing on fine-tuning these models for specific applications in classification tasks. My work has involved training these models on custom datasets for multi-class and multi-label classification, leveraging advanced techniques to optimize performance and efficiency.

### Fine-Tuning at Wolters Kluwer

During my tenure as a Data Science Intern at Wolters Kluwer, I was involved in fine-tuning large language models to enhance their performance on various NLP tasks. One of my key responsibilities was optimizing production-deployed models, including Vision Encoder-Decoder and Layout Transformer models. I fine-tuned these models using custom datasets to improve their accuracy in text extraction and classification tasks. By employing techniques such as data augmentation and batch classification, I was able to achieve significant improvements in model performance metrics.

### Fine-Tuning BERT at HCL Technologies

As a Research Intern at HCL Technologies, I worked extensively with BERT for Named Entity Recognition (NER) and text classification tasks. I fine-tuned BERT on custom datasets to improve its ability to accurately identify and classify entities within textual data. This involved preprocessing the data, setting up the training environment, and using advanced techniques to enhance model accuracy. I achieved a recall of 88% for NER tasks and employed Random Forests alongside BERT to classify sentences into 'Rule' or 'Not Rule' categories with a 92% accuracy rate.

### Training Techniques and Methodologies

In addition to fine-tuning specific models, I have implemented data distributed parallel training methods to train large language models on GPUs. This approach involves distributing the data and model computations across multiple GPUs, significantly accelerating the training process and allowing for the handling of large datasets. Some of the techniques and tools I have used include:

- **Data Distributed Parallel (DDP) Training**: Implemented synchronized multi-GPU processing to minimize idle time and maximize efficiency during training.
- **Hugging Face Transformers**: Leveraged pre-trained models from the Hugging Face library and fine-tuned them on domain-specific data.
- **PyTorch**: Used PyTorch as the primary framework for developing and training models, benefiting from its flexibility and robustness.
- **ONNX**: Converted models to ONNX format to optimize and reduce evaluation time, ensuring quicker inference times for deployed models.

### Key Achievements

- **Wolters Kluwer**: Achieved a 4% improvement across all metrics for production-deployed models through fine-tuning and optimization techniques.
- **HCL Technologies**: Enhanced BERT's NER capabilities to achieve an 88% recall and implemented a hybrid model combining BERT and Random Forests for sentence classification, achieving 92% accuracy.

Overall, my experience with large language models encompasses a thorough understanding of model fine-tuning, distributed training methodologies, and practical implementation to solve complex NLP problems effectively.

I have extensive experience working with large language models (LLMs) such as BERT, and GPT-2, focusing on fine-tuning these models for specific applications in classification tasks. My work has involved training these models on custom datasets for multi-class and multi-label classification, leveraging advanced techniques to optimize performance and efficiency.

As a Data Science Intern at Wolters Kluwer, I was involved in fine-tuning large language models to enhance their performance on various NLP tasks. I was involved in fine-tuning large language models to enhance their performance on various NLP tasks. My primary project revolved around training a layout model based on document understanding, utilizing a Swin Transformer. This model processes language tokens and classifies them into different classes, leveraging OCR (Optical Character Recognition) for initial data extraction. The OCR step was crucial in converting scanned documents into machine-readable text, which then served as input for the Swin Transformer model.

I also worked on the classification of large language models such as BERT and GPT-2. I fine-tuned these models on custom datasets for multi-class and multi-label classification tasks, ensuring they could accurately handle domain-specific language and classification challenges. This process involved meticulous data preparation, model tuning, and performance evaluation to achieve the desired accuracy and robustness.

As a Research Intern at HCL Technologies, I worked extensively with BERT for Named Entity Recognition (NER) and text classification tasks. I fine-tuned BERT on custom datasets to improve its ability to accurately identify and classify entities within textual data. I achieved a recall of 88% for NER tasks and employed Random Forests alongside BERT to classify sentences into 'Rule' or 'Not Rule' categories with a 92% accuracy rate.

In addition to fine-tuning specific models, I have implemented data distributed parallel training methods to train large language models on GPUs. This approach involves distributing the data and model computations across multiple GPUs, significantly accelerating the training process and allowing for the handling of large datasets. Some of the techniques and tools I have used include:

- Data Distributed Parallel (DDP) Training: Implemented synchronized multi-GPU processing to minimize idle time and maximize efficiency during training.
- Hugging Face Transformers: Leveraged pre-trained models from the Hugging Face library and fine-tuned them on domain-specific data.
- PyTorch: Used PyTorch as the primary framework for developing and training models, benefiting from its flexibility and robustness.
- ONNX: Converted models to ONNX format to optimize and reduce evaluation time, ensuring quicker inference times for deployed models.

My experience with large language models involves a thorough understanding of model fine-tuning, distributed training methodologies, and practical implementation to solve complex NLP problems effectively.

---

## Experience in Prompt Engineering, Information Retrieval, and Retrieval-Augmented Generation

I have substantial experience working with prompt engineering and retrieval-augmented generation (RAG) using tools like Llama-Index and Langchain. My work in these areas involves embedding data, efficient storage and retrieval, and enhancing the performance of large language models (LLMs) through context-based query resolution.

### Embedding and Retrieval with Llama-Index and Langchain

In my projects, I have embedded data from various sources, such as PDFs, and stored the embedded data in a vector store. This allows for efficient information retrieval based on user queries. By embedding user queries, I can retrieve the top-matching results from the vector store, providing relevant context for the LLM to generate accurate responses. This method ensures that the responses are not only precise but also include proper citations of the sources, enhancing the reliability and traceability of the information provided.

### Experimentation with Indexers and Retrieval Methods

To optimize the embedding and retrieval processes, I have experimented with different indexers for indexing data and various retrieval methods for fetching the most relevant information. This experimentation helps in fine-tuning the performance of the system, ensuring that the most relevant context is provided to the LLM for generating answers. My approach typically involves:

- **Indexing**: Utilizing different indexing techniques to efficiently organize and store embedded data.
- **Retrieval**: Testing various retrieval algorithms to ensure the highest accuracy in matching user queries with the stored data.
- **Prompt Engineering**: Crafting and refining prompts to ensure that the LLM can generate contextually accurate and relevant responses.

### Practical Applications and Achievements

By leveraging Llama-Index and Langchain, I have successfully implemented systems that can handle complex information retrieval tasks and augment the capabilities of large language models. These systems are designed to provide detailed, accurate answers with appropriate citations, making them valuable for various applications such as academic research, customer support, and automated documentation.

In summary, my work with prompt engineering and retrieval-augmented generation has enabled me to build sophisticated systems that enhance the performance and accuracy of large language models. By embedding data, optimizing retrieval methods, and refining prompts, I ensure that the LLMs can deliver precise and reliable information tailored to user queries.

---

## Experience with Distributed Systems and Deep Learning?

I have a robust background in both distributed systems and deep learning, leveraging these technologies to develop, train, and deploy sophisticated machine learning models at scale. My experience spans various aspects of these fields, from optimizing model training to implementing efficient data processing pipelines.

### Distributed Systems

#### Data Distributed Parallel (DDP) Training

In my role at Wolters Kluwer, I implemented Data Distributed Parallel (DDP) training to optimize the performance of large language models. By distributing the training process across multiple GPUs, I significantly reduced training time and enhanced the scalability of the models. This involved synchronizing the workloads across GPUs, minimizing idle times, and ensuring efficient utilization of hardware resources.

### Deep Learning

#### Model Training and Optimization

My deep learning experience includes training various models such as BERT, GPT-2, and custom neural networks. I have fine-tuned these models for specific tasks, such as text classification, named entity recognition, and image processing. This work often involved hyperparameter tuning, data augmentation, and implementing advanced training techniques to improve model accuracy and robustness.

#### PyTorch and TensorFlow

I am proficient in using deep learning frameworks like PyTorch and TensorFlow. These tools are essential for developing and training neural networks, and I have utilized them to build and deploy models for various applications. My projects often involve customizing network architectures, optimizing training loops, and integrating these models into larger systems.

#### Distributed Deep Learning

Combining my knowledge of distributed systems and deep learning, I have implemented distributed deep learning solutions to handle large-scale training tasks. By parallelizing the training process and distributing the data across multiple nodes, I could train models more efficiently and effectively handle large datasets. This approach ensures that deep learning models can be trained quickly without compromising accuracy or performance.

### Key Projects and Achievements

- **Table Detection Pipeline**: Developed a table detection pipeline using OpenCV for document understanding tasks. This involved detecting tables in scanned documents, extracting tabular data using geometric algorithms, and training a model to improve detection accuracy.
- **Vision Encoder-Decoder Models**: At Wolters Kluwer, I optimized vision encoder-decoder models for document processing tasks. This included converting models to the ONNX format to reduce evaluation time and implementing synchronized multi-GPU processing for efficient model training.
- **Reinforcement Learning**: As part of the Google Summer of Code, I developed reinforcement learning models using DQN and Advantage Actor Critic algorithms. This project involved parallel processing for environment simulation and optimizing reinforcement learning workflows.

<!-- Overall, my experience with distributed systems and deep learning encompasses a wide range of technical skills and practical applications. By leveraging advanced techniques and tools, I have successfully developed and deployed scalable, efficient, and high-performance machine learning solutions. -->