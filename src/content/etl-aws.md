# ETL Data Pipeline on AWS

## Why ETL Pipelines Matter for ML/AI

Machine learning models are only as good as their data. **ETL (Extract, Transform, Load)** pipelines are the backbone of data engineering—they move data from sources, clean and transform it, and load it into destinations where models can consume it. In production ML systems, data pipelines run continuously, handling everything from user event streams to training dataset refreshes.

AWS provides a rich ecosystem for building ETL pipelines: S3 for storage, Lambda for transformation, Glue for orchestration, Athena for querying, and Step Functions for workflow coordination.

---

## ETL Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    ETL Pipeline on AWS                                │
│                                                                       │
│  EXTRACT                    TRANSFORM                    LOAD         │
│  ───────                    ─────────                    ────         │
│                                                                       │
│  ┌───────────┐             ┌───────────────┐           ┌───────────┐ │
│  │ Database  │             │  AWS Lambda   │           │  S3       │ │
│  │ (RDS/     │─Kinesis──► │  or Glue ETL  │──►        │  (Data    │ │
│  │  DynamoDB)│             │               │           │   Lake)   │ │
│  └───────────┘             └───────────────┘           └─────┬─────┘ │
│                                                                       │
│  ┌───────────┐             ┌───────────────┐           ┌─────▼─────┐ │
│  │ API Logs  │             │  Spark on     │           │ Redshift │ │
│  │ (CloudWatch)─► Lambda ─►│  EMR          │──►        │ Athena   │ │
│  └───────────┘             └───────────────┘           └───────────┘ │
│                                                                       │
│  ┌───────────┐             ┌───────────────┐           ┌───────────┐ │
│  │ CSV Files │             │  Glue         │           │ Sagemaker│ │
│  │ (Upload)  │─► S3 ────► │  Transform    │──►        │ Feature  │ │
│  └───────────┘             │  (PySpark)    │           │ Store    │ │
│                            └───────────────┘           └───────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## AWS Services for ETL

| Service | Role | When to Use |
|---------|------|-------------|
| **S3** | Data lake storage | Raw data, transformed data, model artifacts |
| **AWS Glue** | Serverless ETL | Scheduled data transformation jobs |
| **Lambda** | Event-driven compute | Small, quick transformations |
| **Kinesis** | Stream processing | Real-time data ingestion |
| **Step Functions** | Workflow orchestration | Complex multi-step pipelines |
| **Athena** | Query engine | SQL queries on S3 data |
| **CloudWatch** | Monitoring & logs | Pipeline health, error alerts |

---

## Building an ETL Pipeline: S3 → Lambda → S3

### Step 1: S3 Bucket Structure

```
my-ml-data-lake/
├── raw/
│   ├── users/
│   │   └── 2024-01-15.json
│   ├── transactions/
│   │   └── 2024-01-15.csv
│   └── events/
│       └── 2024-01-15.parquet
├── processed/
│   ├── user_features/
│   └── training_data/
├── models/
│   └── sentiment-v1/
└── predictions/
    └── 2024-01-15/
```

### Step 2: Lambda Transformation Function

```python
# lambda_function.py
import json
import boto3
import pandas as pd
import io
from datetime import datetime

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Triggered when new data lands in raw/ bucket.
    Transforms and saves to processed/ bucket.
    """
    # Extract bucket and key from S3 event
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']

        print(f"Processing: s3://{bucket}/{key}")

        # 1. EXTRACT: Read data from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response['Body'].read().decode('utf-8')

        # Parse JSON events
        events = [json.loads(line) for line in data.strip().split('\n')]

        # 2. TRANSFORM: Clean and feature engineer
        df = pd.DataFrame(events)

        # Data cleaning
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.dropna(subset=['user_id', 'event_type'])

        # Feature engineering
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour'] = df['timestamp'].dt.hour
        df['text_length'] = df['text'].str.len()
        df['has_emoji'] = df['text'].str.contains(r'[^\w\s]', regex=True).astype(int)

        # Filter: keep only valid events
        df = df[df['text_length'] > 0]
        df = df[df['text_length'] <= 10000]

        # 3. LOAD: Save transformed data
        output_bucket = 'my-ml-data-lake'
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
        output_key = f'processed/events/{date_str}/transformed.parquet'

        # Write to buffer
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        s3.put_object(
            Bucket=output_bucket,
            Key=output_key,
            Body=parquet_buffer.getvalue()
        )

        print(f"Saved: s3://{output_bucket}/{output_key}")
        print(f"Rows processed: {len(df)}")

    return {
        'statusCode': 200,
        'body': json.dumps(f'Processed {len(events)} events')
    }
```

---

## Glue ETL Jobs with PySpark

```python
# glue_etl_job.py (AWS Glue script)
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job

# Initialize Glue context
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glue_context = GlueContext(sc)
spark = glue_context.spark_session
job = Job(glue_context)
job.init(args['JOB_NAME'], args)

# ── EXTRACT ──
# Read from S3 (CSV, JSON, Parquet)
raw_data = spark.read.parquet("s3://my-ml-data-lake/raw/events/2024-01-*")

# Read from Glue Data Catalog
users_df = glue_context.create_dynamic_frame.from_catalog(
    database="ml_database",
    table_name="users"
).toDF()

# ── TRANSFORM ──
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Clean data
cleaned = raw_data.filter(
    (F.col("text").isNotNull()) &
    (F.length(F.col("text")) > 0)
)

# Feature engineering
transformed = cleaned.withColumn(
    "text_length", F.length("text")
).withColumn(
    "word_count", F.size(F.split("text", " "))
).withColumn(
    "avg_word_length", F.col("text_length") / F.col("word_count")
).withColumn(
    "hour_of_day", F.hour("timestamp")
).withColumn(
    "is_weekend", F.dayofweek("timestamp").isin([1, 7]).cast("int")
)

# Aggregation: user-level features
user_features = transformed.groupBy("user_id").agg(
    F.count("*").alias("total_events"),
    F.avg("text_length").alias("avg_text_length"),
    F.countDistinct("hour_of_day").alias("active_hours"),
    F.sum("is_weekend").alias("weekend_events"),
    F.collect_list("text").alias("all_texts")
)

# ── LOAD ──
user_features.write.mode("overwrite").parquet(
    "s3://my-ml-data-lake/processed/user_features/"
)

# Write to Glue Data Catalog for Athena querying
user_features.write.mode("overwrite").format("parquet").saveAsTable(
    "ml_database.user_features"
)

job.commit()
```

---

## Step Functions: Orchestrating Multi-Step Pipelines

```python
# Define pipeline as Step Functions state machine
import boto3

stepfunctions = boto3.client('stepfunctions')

definition = {
    "Comment": "ML Data Pipeline",
    "StartAt": "ExtractRawData",
    "States": {
        "ExtractRawData": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123456789:function:extract-data",
            "Next": "TransformData"
        },
        "TransformData": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123456789:function:transform-data",
            "Next": "ValidateData"
        },
        "ValidateData": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123456789:function:validate-data",
            "Next": "ChoiceState"
        },
        "ChoiceState": {
            "Type": "Choice",
            "Choices": [{
                "Variable": "$.validation.valid",
                "BooleanEquals": True,
                "Next": "TrainModel"
            }, {
                "Variable": "$.validation.valid",
                "BooleanEquals": False,
                "Next": "AlertFailure"
            }],
            "Default": "AlertFailure"
        },
        "TrainModel": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123456789:function:trigger-training",
            "Next": "SuccessState"
        },
        "AlertFailure": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-east-1:123456789:function:send-alert",
            "End": True
        },
        "SuccessState": {
            "Type": "Succeed"
        }
    }
}
```

Visual representation:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Extract     │───►│ Transform   │───►│  Validate   │
│ Raw Data    │    │ Data        │    │  Data       │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                                        ┌────▼────┐
                                        │ Choice  │
                                        └────┬────┘
                                    Valid? │    │ Invalid
                                    ┌──────▼──┐  ┌──▼──────────┐
                                    │ Train   │  │ Alert       │
                                    │ Model   │  │ Failure     │
                                    └────┬────┘  └─────────────┘
                                         │
                                    ┌────▼────┐
                                    │ Success │
                                    └─────────┘
```

---

## Athena: Query Data in S3 with SQL

```sql
-- Query transformed data directly from S3
SELECT
    user_id,
    total_events,
    avg_text_length,
    active_hours
FROM ml_database.user_features
WHERE total_events > 10
ORDER BY avg_text_length DESC
LIMIT 100;

-- Compute training data statistics
SELECT
    COUNT(*) as total_samples,
    AVG(text_length) as avg_length,
    PERCENTILE(text_length, 0.5) as median_length,
    MIN(text_length) as min_length,
    MAX(text_length) as max_length
FROM ml_database.training_data;
```

---

## Monitoring & Data Quality

```python
# Lambda function for data quality checks
def validate_data(df):
    """Validate data quality before loading."""
    results = {
        'valid': True,
        'checks': [],
        'errors': []
    }

    # Check 1: No null user_ids
    null_users = df['user_id'].isna().sum()
    if null_users > 0:
        results['valid'] = False
        results['errors'].append(f'{null_users} null user_ids')

    # Check 2: No duplicate entries
    duplicates = df.duplicated(subset=['user_id', 'timestamp']).sum()
    if duplicates > 0:
        results['valid'] = False
        results['errors'].append(f'{duplicates} duplicate entries')

    # Check 3: Data freshness
    latest = df['timestamp'].max()
    if (pd.Timestamp.now() - latest).days > 1:
        results['errors'].append(f'Stale data: latest is {latest}')

    # Check 4: Distribution check
    text_lengths = df['text_length']
    if text_lengths.mean() < 5 or text_lengths.mean() > 5000:
        results['errors'].append(f'Abnormal text length distribution')

    return results
```

---

## Exercises

### Exercise 1: Design a Pipeline
Design an ETL pipeline that processes user reviews from a PostgreSQL database, creates features, and stores them for model training.

**Solution:**
```
1. EXTRACT: AWS DMS (Database Migration Service) to replicate
   PostgreSQL changes to S3 (CDC - Change Data Capture)

2. TRANSFORM: AWS Glue job (PySpark) that:
   - Reads raw reviews from S3
   - Cleans text (remove HTML, normalize)
   - Computes features: word_count, sentiment_keywords, readability_score
   - Joins with user demographics table

3. LOAD:
   - Save processed features to S3 (Parquet, partitioned by date)
   - Register in Glue Data Catalog
   - Trigger model training via SNS notification

4. MONITOR: CloudWatch alarms for:
   - Pipeline failure
   - Data volume anomalies
   - Feature distribution drift
```

### Exercise 2: PySpark Transformation
Write PySpark code to compute daily active user features from event logs.

**Solution:**
```python
from pyspark.sql import functions as F

daily_features = events_df.withColumn("date", F.to_date("timestamp")).groupBy(
    "user_id", "date"
).agg(
    F.count("*").alias("daily_events"),
    F.countDistinct("event_type").alias("unique_event_types"),
    F.min("timestamp").alias("first_event"),
    F.max("timestamp").alias("last_event"),
    F.avg("text_length").alias("avg_text_length")
)
```

### Exercise 3: Error Handling
How would you handle a scenario where the raw data format changes unexpectedly?

**Solution:**
1. **Schema validation**: Use Glue Schema Registry to enforce schemas
2. **Try-catch in Lambda**: Catch parsing errors, move bad records to S3 error bucket
3. **Dead Letter Queue**: Failed records go to DLQ for manual review
4. **Alerting**: CloudWatch alarm triggers SNS notification
5. **Graceful degradation**: Process valid records, quarantine invalid ones
6. **Schema evolution**: Use Parquet format with schema evolution support

---

## Key Takeaways

1. **ETL** moves data from sources through transformations to ML-ready destinations
2. **S3** is the foundation of data lakes on AWS—store raw, processed, and model data
3. **Lambda** handles event-driven, small transformations; **Glue** handles large-scale batch ETL
4. **Step Functions** orchestrate multi-step pipelines with error handling and branching
5. **Data quality checks** are essential—never assume input data is clean or consistent
