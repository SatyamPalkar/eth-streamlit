__version__ = "0.3.0"

from . import features
from . import io
from . import metrics
from . import utils
from . import preprocessing
from . import models
# New crypto modules for AT3
from . import crypto_data
from . import crypto_models
from . import crypto_api

# Existing weather prediction functions
from .models import (
    predict_rain_smart,
    predict_precipitation_smart,
    get_api_status,
    get_health_status,
    get_model_info,
    get_project_info,
    get_health_check,
    predict_rain_by_date,
    predict_precipitation_by_date
)

# New crypto prediction functions
from .crypto_data import (
    get_ethereum_data,
    preprocess_crypto_data,
    CryptoDataFetcher
)

from .crypto_models import (
    EthereumPredictor,
    train_ethereum_model,
    predict_ethereum_price
)

from .crypto_api import (
    app as crypto_api_app
)

__all__ = [
    # Existing modules
    "metrics",
    "io",
    "utils",
    "features",
    "preprocessing",
    "models",
    # Existing weather functions
    "predict_rain_smart",
    "predict_precipitation_smart",
    "get_api_status",
    "get_health_status",
    "get_model_info",
    "get_project_info",
    "get_health_check",
    "predict_rain_by_date",
    "predict_precipitation_by_date",
    # New crypto modules
    "crypto_data",
    "crypto_models", 
    "crypto_api",
    # New crypto functions
    "get_ethereum_data",
    "preprocess_crypto_data",
    "CryptoDataFetcher",
    "EthereumPredictor",
    "train_ethereum_model",
    "predict_ethereum_price",
    "crypto_api_app"
]