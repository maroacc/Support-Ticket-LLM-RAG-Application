# Architecture Documentation

This document describes the architecture choices for the intelligent support system.
Since a real production solution requires a great amount of time to implement, this document describes both the
ideal and the implemented solution, explaining the design choices and compromises.

## Ideal Architecture

The ideal architecture is a system that can respond to the following requirements :
- Support of training and versioning of machine learning models
- Support of real-time inference

For the first point, we need to store historical ticket data and transform it to feed it
into ML models and of course store the machine learning models. Since this will be done by AI Engineers / Data Scientists during the day who do not need ultra-fast responses, or it can be done in batches
at night to retrain production models with new data, the architecture does not need very fast responses, but it does need to be able to manage
very large volumes of data.

On the other hand, for inference we do need quick response times, but we do not manage large amounts of data.
However, the amount of new tickets per day is +500, which makes around 1 ticket per minute. That is not very high concurrency, 
so that means we don't have to chose ultra performant solutions for real time. However, if the volume of tickets augmented,
then we might need to adapt and look for other solutions. 

The chosen schema is therefore the following : 

![schema](Documentation\images\Architecture.png)

Disclaimer : The particular technologies chosen are an example. However, it is impossible to
properly choose tools without knowing budget limitations, the compostion of the team, technologies that are already being
used in the company, security and data protection requirements, etc. In a real use case,
all these variables should be analyzed before making a decision.


### Ticket System
This is the support ticket application where the tickets are created.

### Operational Database

This is where the ticket system stores its data. It stores all the ticket and client information
needed for the application to run.

### Data Lake
Historical data is stored raw in a Data Lake to be later used for model training. In addition, everyday
new ticket data is collected from the Operational Database and stored in For this technology
I chose Databricks because it can handle very large volumes of data, it is a ML first technology and it is very flexible and is integrated with MLFlow. I compared it to Snowflake but
Snowflake is more for structured data (although it also supports semi-structured), it is better for streaming with ____ . Snowflake is also more limited for ML. The compromise is that
it is harder to set up and maintain.
Also in the data lake is where the data quality controls start, with data stewards having access to it to
explore raw data.

### Offline feature store

There needs to be place where we store the features needed to train the model and also for analytic purposes. This is processed and clean data.
For this I chose the databricks feature store because of its integration with databricks. However there are other popular feature store
solutions like fever and tecton. Feast is open source and doesnt have vendor lock in but it needs more infrastructure management. tecton is__.


This information will be accessed both by models during training and by BI tools
for analytic purposes.

### Model Registry and Experiment tracking

This component does 2 things. On the one hand it stores all the information related to the training 
of models, like the metrics and parameters. And on the other hand it stores the different models stating their versions. 
It also marks what model is the production model and what models belong to the test environment.

### Api Server
The ticket system will call the api server to get suggested solutions and similar tickets for the new ticket created. 
I assume all information needed for the model inference will be provided in the ticket. However, if this wasnt the 
case it would maybe have to connect to the operational database to get some information. Or for example if there were calculated fields it could connect
to the online feature store to access them quickly.
For the solution I chose uvicorn + fast api because it is very easy to set up and maintain, and it is vquite a small API that will only be accessed by the ticket system for
+500 tickets a day, so a backend in python is good enough. Even if it scaled we could just use more uvicorn workers. and fast API is very fast and has automatic documentation.

### Online feature store
This is where we store data that is gonna be needed by our intelligent system for inference in real time. It could store precalculated features needed for the ML model like previous_tickets, 
but we don't use it for the moment we don't use it for our model. So it's gonna store precalculated data needed for the retrieval system. It is gonna store the precalculated embeddings for the
semantic search and it is going to store the knowledge graph. 
For these solutions, I am assuming that the team is big enough and has the knowledge to maintain a Graph database and a vector database. The reason to choose these solutions when we
already have an operational database and a data lake is that these databases are optimized for this type of data. For example, a Graph database like Neo4j will first of all be faster to call
to find all relationships than postgre and a lot better than mongoDB with all the joins, which is very necessary to find real time data. The disadvantage is that it will require synchronizing with the operational database for
it to have all the updated info. And Neo4j is relatively simple to maintain because it can be in the cloud. If the teams is not ready to maintain it, they could just call the operational database and if it too slow, they can show the proposed solutions only to the agent and not to the person
creating the ticket, so it would have more time to get all the data and would only be semi real-time. For the vector database, i would highly recommend it because it will be a lot faster to calculate the cosine simmilarity
between all the tickets. And since at every call we would have to calculate the cosine simmilarity with the whole database and this will only increase with time.  
For vector database we can choose pgvector if we are already using posgre and will only use it for this because it requires kinimal installation and maintenance. Or if not ____.
If we needed to precompute features then I would add redis to the mix to store information like for example the amount of times a user has created a ticket in the last month that can be used in real time to precipute.

### Orchestration
The orchestration to retrain the model everynight will be done in Airflow
because it can be very easily integrated with python ad because_________.

    

ADD CI/CD DEPLOYMENT!!!!! AND CLOUD PROVIDER!!!