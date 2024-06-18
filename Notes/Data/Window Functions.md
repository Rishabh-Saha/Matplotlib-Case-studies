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


--------
User Defined Functions (UDFs)

DELIMITER $$

CREATE FUNCTION function_name(func_parameter1, func_parameter2, ...)
  RETURN datatype [characteristics]
/*      func_body      */
  BEGIN
    <SQL Statements>
    RETURN expression;
END ; $$

DELIMITER ;

CALL function_name;


You need to specify the Deterministic keyword to ensure that the output is the same for the same input values. This is disabled in MySQL by default.

---------------
Stored Procedures

DELIMITER $$

CREATE PROCEDURE Procedure_name (<Paramter List>)
BEGIN
  <SQL Statements>
END $$

DELIMITER ;

CALL Procedure_name;
------------------

UDF vs 	Stored Procedure
1. It supports only the input parameter, not the output.	1. It supports input, output and input-output parameters.
2. It cannot call a stored procedure.	2. It can call a UDF.
3. It can be called using any SELECT statement.	3. It can be called using only a CALL statement.
4. It must return a value.	4. It need not return a value.
5. Only the 'select' operation is allowed.	5. All database operations are allowed.
