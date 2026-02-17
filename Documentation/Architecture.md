# Architecture Documentation

This document describes the architecture choices for the intelligent support system.
Since a real production solution requires a great amount of time to implement, this document describes both the
ideal and the implemented solution, explaining the design choices and compromises.

## Ideal Architecture

The ideal architecture is a system that can respond to the following requirements :
- Support of training and versioning of machine learning models
- Support of real-time inference

For the first point, we need to store historical ticket data and transform it to feed it
into ML models and of course store the machine learning models. Since this can be done in batches
at night, the architecture does not need very fast responses, but it does need to be able to manage
very large volumes of data.

On the other hand, for inference we do need quick response times, but we do not manage large amounts of data. The
amount of new tickets per day is +500, which makes around 1 ticket per minute. That is not very high concurrency, 
so that means we don't have to chose ultra performant solutions for real time. However, if the volume of tickets augmented,
then we might need to adapt and look for other solutions. 

The chosen schema is therefore the following : 

IMAGE

### Operational Database

This is where the ticket system currently stores its operational data. Like for example in PostgreSQL or MongoDB.


### Data Lake
Historical data is stored raw in a Data Lake. This would correspond to the Bronze layer. For this technology
I chose Databricks because __________________. I compared it to Snowflake but ___________.

### Offline feature store

There needs to be place where we store the features needed to train the model and also for analytic purposes. This is processed and clean data.
For this I chose dbt because ______________.

In dbt we first do some data quality to verify that the data is not polluted. This is the silver layer. Then we transform our data
to extract or calculate our features and we arrive to the gold layer (?). 
In this feature store we will store :
- Transformed tickets (with only features we need and calculated embeddings)
- Embeddings for RAG semantic search in the retrieval system
- Metrics like resolution stats for result prioritization in the retrival system 

This information will be accessed both by models during training and by BI tools
for analytic purposes.

### Model Registry and Experiment tracking

This component does 2 things. On the one hand it stores all the information related to the training of models,
like ___________. And on the other hand it stores the different models stating their versions. It also marks what model is the production model and what models belong to the test environment.

### Api Server
The ticket system will call the api server to get suggested solutions and similar tickets for the new ticket created. 
Most information needed will be provided by the ticket, however some information will not be provided by the ticket system. For example, the customer tier. This information will be obtained directly from the
operational database that the ticket system uses. I considered using redis to store online features but in the end decided not to because the model today uses mostly
information already in the ticket, so it is a huge overhead for some fields that are not even important. If in the future the model evolves and needs other precomputed fields, then sure we will use redis.
Uvicorn + Fast API because _____________________-
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


### Orchestration
The orchestration to retrain the model everynight will be done in Airflow
because it can be very easily integrated with python ad because_________.

    

ADD CI/CD DEPLOYMENT!!!!! AND CLOUD PROVIDER!!!