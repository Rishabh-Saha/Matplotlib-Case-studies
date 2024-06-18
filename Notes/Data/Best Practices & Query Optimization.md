Best Practices:

1. Comment your code by using a hyphen (-) for a single line and (/* ... */) for multiple lines of code.
2. Always use table aliases when your query involves more than one source table.
3. Assign simple and descriptive names to columns and tables.
4. Write SQL keywords in upper case and the names of columns, tables and variables in lower case.
5. Always use column names in the 'order by' clause, instead of numbers.
6. Maintain the right indentation for different sections of a query.
7. Use new lines for different sections of a query.
8. Use a new line for each column name.
9. Use the SQL Formatter or the MySQL Workbench Beautification tool (Ctrl+B).

------------

Indexing

The command for creating an index is as follows:

CREATE INDEX index_name
ON table_name (column_1, column_2, ...);
 

The command for adding an index is as follows:

ALTER TABLE table_name
ADD INDEX index_name(column_1, column_2, ...);
 

The command for dropping an index is as follows:

ALTER TABLE table_name
DROP INDEX index_name;
 
------------

Clustered Index vs	Non-Clustered Index
1. This is mostly the primary key of the table.	1. This is a combination of one or more columns of the table.
2. It is present within the table.	2. The unique list of keys is present outside the table.
3. It does not require a separate mapping.	3. The external table points to different sections of the main table.
4. It is relatively faster.	4. It is relatively slower.


---------
Order of SQL Query Execution

FROM (including JOIN)
    WHERE
        GROUP BY
            HAVING
                WINDOW functions
                    SELECT
                        DISTINCT
                            ORDER BY
                                LIMIT and OFFSET

Few more points to remember:
- Use inner joins wherever possible to avoid having any unnecessary rows in the resultant table.
- Apply all the required filters to get only the required data values from multiple tables.
- Index the columns that are frequently used in the WHERE clause.
- Avoid using DISTINCT while using the GROUP BY clause, as it slows down query processing.
- Avoid using SELECT * as much as possible. Select only the required columns.
- Use the ORDER BY clause only if it is absolutely necessary, as it is processed late in a query.
- Avoid using LIMIT and OFFSET as much as possible. Instead, apply appropriate filters using the WHERE clause.

-----------------------

Joins vs nested queries: 

Executing a statement with the 'join' clause creates a join index, which is an internal indexing structure. This makes it more efficient than a nested query. However, a nested query would perform better than a join while querying data from a distributed database.
