---
published: true
---



Usually, the most efficient way to interact with databases is by addressing specific queries. Standard operations (**CRUD**) on databases are usually done through processes called **SQL** (Structured Query Language) queries. 
 
I am talking here about relational databases (**RDBMS**) where records can be created, read, updated, and deleted (CRUD). Each single query adressed to a SQL server database requires a certain amount of time before the response is returned from the server. It is trivial that for each query that we write to the database, we want to make it the most computational efficient way. 
There are a number of standard tuning tricks and techniques for query execution processes to make them efficient. Among others:

1- **Proper creation of indexes**: 
This is by far the most used and the best way to better SQL queries performance. Queries without indexes at all are more likely to be slow. Database indexing is actually a development task and the sensitiveinformation behind indexing resides in the way the data querying happens. 

 
2- **Retrieving ONLY needed data:**
    **- restrict columns:**  
![png](/images/sql_opt1.PNG)
the above request for instance is used to retrieve all records and all columns of a specific table (here Persons) with a salary less than 6000. Most of the times, we just really need specific columns, and it is more efficient to retrieve just what we need:
![png](/images/sql_opt2.PNG)  
We are able here to read only the first_name, last_name and role for persons with salary less than 6000. This surely saves some computation time.   
    **- restrict rows**:   
![png](/images/sql_opt3.PNG)  
there will be cases and this is more often, where instead of retrieving all the rows or records in a specific table, we would like to limit the read data to a specific number of rows. 
The query is returning 50 lines from the table Employee. Rows limitation in query execution has a tremendous effect in database querying optimization. 

     
3- **Use more 'joins' instead than 'correlated subqueries'**:
In some when writing sql queries, the results of some parts of code depend on some other parts of the code. Correlated queries are just that. They depend on what is called outer queries. To illustrate, the following query: 
![png](/images/sql_opt4.PNG)               
tells the sql server to retrieve the list of employees who belong to the IT department. Because the correlated subquery (the one within brackets) is executed for every record of the outer query, it will take more time to return the results (very inefficient). A better option would be to use **JOINS**. The intuition behind 'joins' is simple. Let us reconsider the previous query; In case of 5000 employees in the Employee table, using a join like this:
![png](/images/sql_opt5.PNG)                      
might seem more efficient than applying the correlated subquery to find the employees who belong to the IT department. The correlated subquery will have to run 5000 times before the results are read. Although correlated queries might be a better choice sometimes, we may want to avoid using them when attempting to retrieve many rows.       


4- **No use of Wildcard characters at the start of 'LIKE' clause**:  	
The following query:
![png](/images/sql_opt6.PNG)                   
aims at returning all the employees which contain 'jo' in their first name. The **%** character at the start of the '**LIKE**' clause is inhibiting the ability of the SQL server to use indexing (in case there is). The problem here is that the server has no information about the beginning of the name column, it would have again to scan through each and every row in the table. This may lead to longer query execution. To reduce the execution time, use the % character at the end of the 'LIKE' clause as the following
![png](/images/sql_opt7.PNG)           
tells the server to read all the employees with first names starting with 'jo'.



The above techniques are old as SQL is, and they do work effectively in helping optimizing queries' executions. However, in this talk we would like to cover a strictly different approach for sql database query optimization. It is going to be about an artificially intelligent type of machine learning called Reinforcement learning.
