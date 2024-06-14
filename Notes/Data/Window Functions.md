different types of rank functions:

RANK(): Rank of the current row within its partition, with gaps
DENSE_RANK(): Rank of the current row within its partition, without gaps
PERCENT_RANK(): Percentage rank value, which always lies between 0 and 1

The syntax for writing the window functions are as follows:

RANK() OVER (
  PARTITION BY <expression>[{,<expression>...}]
  ORDER BY <expression> [ASC|DESC], [{,<expression>...}]
)

RANK() can be replaced by DENSE_RANK() OR PERCENT_RANK()


'row number' function for the following use cases:

- To determine the top 10 selling products out of a large variety of products
- To determine the top three winners in a car race
- To find the top five areas in different cities in terms of GDP growth

ROW_NUMBER() OVER (
  PARTITION BY <expression>[{,<expression>...}]
  ORDER BY <expression> [ASC|DESC], [{,<expression>...}]
)

Using the same window to define multiple 'over' clauses

WINDOW window_name AS (window_spec)
  [, window_name AS (window_spec)] ...

After group by and before Order by


 'lead' and 'lag' functions are as follows:

LEAD(expr[, offset[, default]])
  OVER (Window_specification | Window_name) 
 

LAG(expr[, offset[, default]])
  OVER (Window_specification | Window_name)