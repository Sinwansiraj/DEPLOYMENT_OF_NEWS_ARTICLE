"""
Central configuration for the News Categorization System.
All environment variables and constants are managed here.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()   # reads .env into os.environ automatically

@dataclass
class ModelConfig:
    """Model training and inference configuration."""
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 4
    max_length: int = 128
    batch_size: int = 8
    num_epochs: int = 2
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    local_model_dir: str = "model_artifacts"
    s3_model_key: str = "models/news-categorizer/model.tar.gz"


@dataclass
class AWSConfig:
    """AWS service configuration."""
    region: str = os.getenv("AWS_REGION","ap-south-2")
    s3_bucket: str = os.getenv("S3_BUCKET", "your-news-categorizer-bucket")
    aws_access_key: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")


@dataclass
class DBConfig:
    """PostgreSQL RDS configuration."""
    host: str = os.getenv("DB_HOST", "your-rds-endpoint.amazonaws.com")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "news_categorizer")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "your_password")


# Label mapping for AG News
LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology",
}

MODEL_CONFIG = ModelConfig()
AWS_CONFIG = AWSConfig()
DB_CONFIG = DBConfig()