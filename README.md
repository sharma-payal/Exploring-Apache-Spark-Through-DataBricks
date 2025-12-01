Introduction to PySpark in Databricks: Exploratory Data Analysis (EDA) Lab

Welcome to your first lab using PySpark in the Databricks environment! In this lab, you will learn how to conduct Exploratory Data Analysis (EDA) using PySpark. The focus will be on basic PySpark functions, understanding the Spark DataFrame API, and getting hands-on with common EDA tasks.

We’ll use Databricks, a popular cloud-based platform for Spark. In this lab, you will:

Set up your Databricks workspace
Load a dataset into Spark
Explore the dataset using basic PySpark functions
Perform exploratory data analysis (EDA) using PySpark
Visualize data using built-in functions and libraries
Step 1: Setting Up Your Databricks Workspace

Log in to Databricks Community Edition (if you don’t have an account, create one at databricks.com Links to an external site.)
Create a new notebook:
Once logged in, go to the Workspace section and create a new notebook.
Give your notebook a meaningful name, for example: "ALY 6110 PySpark Lab".
Set the language to Python.
(If Needed) Create a new Resource and select the default cluster. Wait for it to attach.
Question: What are the advantages of using a cloud-based platform like Databricks for working with Spark?
Question: How does Spark's cluster computing model differ from traditional single-machine data processing?
Step 2: Setting Up Spark Session

Before performing any analysis, you need to create a Spark session. This will allow you to run Spark code in the Databricks environment. This step is usually automatic in Databricks, but we will initialize it for clarity.

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("ALY6110_EDA_Lab").getOrCreate()
Step 3: Loading the Dataset

For this lab, we'll use the a population and housing sales dataset. 

     # Load sample data

     df = spark.read.csv("/databricks-datasets/samples/population-vs-price/data_geo.csv", header=True, inferSchema=True)

     df.show()
Question: What are the key differences between a Spark DataFrame and a Pandas DataFrame in Python?

Step 4:Basic Cleaning of the Dataset

df = df.dropna()

# Rename the columns

df = df.withColumnRenamed("2014 Population estimate", "Population").withColumnRenamed("2015 median sales price", "Sales Price")

# Show the DataFrame with renamed columns

df.show()
Question:  In the above code snipped, we remove all rows that are missing data.  Explain the challenges of this decisions in a big data set.

Step 5: Perform Basic Data Operations: 

Perform basic data operations such as filtering, grouping, and aggregating data using Spark DataFrames.

df.printSchema()
# Filter data     

filtered_df = df.filter(df['State'] == 'California')     

filtered_df.show()    
from pyspark.sql.functions import sum
total_population = df.agg(sum("Population").alias("Total_Population"))

total_population.show()
 

from pyspark.sql.functions import round, avg, col,sum

# Group and aggregate data    

grouped_df = df.groupBy("State").agg(

    round(avg(col("Sales Price")), 2).alias("Average_Sales"),

    sum(col("Population")).alias("Population")

)

grouped_df.show()

 

Question: How do these operations in Spark compare to similar operations in Python or R?

 

from pyspark.sql.functions import round

# Add a new column

transformed_df = df.withColumn("Price_Squared", df["Sales Price"] * df["Sales Price"])

# Assuming you want to round to 2 decimal places

transformed_df =transformed_df.withColumn("Price_Squared", round(col("Price_Squared"), 2))

transformed_df.show()
 

 Question: What are the benefits of using Spark for large-scale data transformations?

# Describe the data     

df.describe().show()     

     

# Calculate correlation     

correlation = df.stat.corr("Population", "Sales Price")     

print(f"Correlation between Population and Price: {correlation}") 
