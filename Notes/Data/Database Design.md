2 key elements for dimensional modelling:

- Facts are the numerical data in a data warehouse
- Dimensions are the metadata (that is, data explaining some other data) attached to the fact variables. 

Both facts and dimensions are equally important for generating actionable insights from a data set.


Different types of Schemas in DW:
- Star Schema
- SnowFlake Schema
- Galaxy Schema
- Star Cluster Schema

OLAP vs OLTP
- OLAP: Online Analytical Processing
    - OLAP (On-line Analytical Processing) is characterized by relatively low volume of transactions. Queries are often very complex and involve aggregations. For OLAP systems a response time is an effectiveness measure. OLAP applications are widely used by Data Mining techniques. In OLAP database there is aggregated, historical data, stored in multi-dimensional schemas (usually star schema).  For example, a bank storing years of historical records of check deposits could use an OLAP database to provide reporting to business users. 

- OLTP: Online Transactional Processing
    - OLTP (On-line Transaction Processing) is characterized by a large number of short on-line transactions (INSERT, UPDATE, DELETE). The main emphasis for OLTP systems is put on very fast query processing, maintaining data integrity in multi-access environments and an effectiveness measured by number of transactions per second. In OLTP database there is detailed and current data, and schema used to store transactional databases is the entity model (usually 3NF). 


SETL (Select Extract Transform Load)

    - Select: Identification of the data that you want to analyse
    - Extract: Connecting to the particular data source and pulling out the data
    - Transform: Modifying the extracted data to standardise it
    - Load: Pushing the data into the data warehouse

